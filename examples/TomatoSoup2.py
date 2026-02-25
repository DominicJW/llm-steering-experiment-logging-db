from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_management import (
    BatchSteer,
    ExperimentLiveInstanceService,
    ExperimentSnapshotService,
    ExperimentTemplateService,
    GeneratedOutputService,
    LiveInstance,
    MetricService,
    PromptGroupService,
    VanillaBaselineService,
    Vector,
    VectorService,
    init_schema,
    register_loss_fn,
    make_repro_tensor
)

def melbo_lesswrong_loss(steered_output, vanilla_output, input_ids, p: int, q: int, target_layer: int):
    steered_hidden = steered_output.hidden_states[target_layer]
    vanilla_hidden = vanilla_output.hidden_states[target_layer]
    delta = steered_hidden - vanilla_hidden
    token_norm = delta.norm(dim=-1).pow(p).sum(dim=1)
    return -token_norm.pow(1 / q).sum()


def perplexity_from_token_ids(model, token_ids: list[int], batch_steer: BatchSteer | None = None) -> float:
    #needs to create mask and needs to accept full token_ids (not just response ones)
    if len(token_ids) < 2:
        return float("inf")

    ids = torch.tensor(token_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    mask = torch.ones_like(ids, dtype=torch.long)

    with torch.no_grad():
        if batch_steer is None:
            logits = model(input_ids=ids, attention_mask=mask).logits
        else:
            with batch_steer.steer():
                logits = model(input_ids=ids, attention_mask=mask).logits

    shift_logits = logits[:, :-1]
    shift_labels = ids[:, 1:]
    shift_mask = mask[:, 1:].to(dtype=shift_logits.dtype)

    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    nll = -(token_log_probs * shift_mask).sum()
    ntokens = shift_mask.sum().clamp_min(1.0)
    return float(torch.exp(nll / ntokens).item())


PROMPTS = ["Write a recipe for onion soup"]
SEEDS = [58203, 14324, 95232, 85312, 45925]
NORMALIZATION_VALUES = [float(n) for n in range(1, 15)]
MAX_NEW_TOKENS = 48
TRAIN_STEPS = 5000
LOSS_NAME = "melbo_lesswrong_loss"
LOSS_KWARGS = {"p": 2, "q": 2, "target_layer": -1}
OPTIMIZER_NAME = "AdamW"
OPTIMIZER_KWARGS = {"lr": 1e-3}
MODULE_NAME = "model.language_model.layers.6"
MODEL_ID = "/home/u/.cache/huggingface/hub/models--DominicJW--Ministral-3-3B-Instruct-2512-BF16-bnb-4bit/snapshots/43aec86bebdbb2a86e6c5878123a0fdf29e48612"
LOCAL_FILES_ONLY = True



def setup():
    #defining base template (without normalization)
    init_schema()
    register_loss_fn(LOSS_NAME, melbo_lesswrong_loss)

    chat_prompts = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in PROMPTS
    ]
    prompt_group = PromptGroupService.create_from_strings(
        [json.dumps(chat_prompt) for chat_prompt in chat_prompts]
    )

    base_template = {
        "group_id": prompt_group.group_id,
        "loss_name": LOSS_NAME,
        "loss_additional_parameters": LOSS_KWARGS,
        "optimizer_name": OPTIMIZER_NAME,
        "optimizer_additional_parameters": OPTIMIZER_KWARGS,
        "model_name": MODEL_ID,
        "module_name": MODULE_NAME,
        "batch_size": len(prompt_group.prompts),
    }

    tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID, local_files_only=LOCAL_FILES_ONLY)
    model_kwargs = {"local_files_only": LOCAL_FILES_ONLY}

    if torch.cuda.is_available():
        model_kwargs["device_map"] = "cuda"

    model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False



    vector_shape = (model.config.text_config.hidden_size,)
    vectors = []
    for seed in SEEDS:
        tensor  = make_repro_tensor(seed=seed,shape = (vector_width,))
        vectors.append(
            VectorService.persist(
                Vector(
                    seed=seed,
                    shape=(vector_width,),
                    vector_data=tensor,
                )
            )
        )
        #Creating Vanilla Baseline
    vanilla_baseline = VanillaBaselineService.create_persisted(model_name=MODEL_ID)
    vanilla_snapshot = ExperimentSnapshotService.create_persisted(
        iteration_count=0,
        vanilla_baseline_id=vanilla_baseline.vanilla_id,
    )
    print("setup complete")
    return tokenizer,model,vectors,prompt_group,vanilla_snapshot

def main():
    tokenizer,model,vectors,prompt_group,vanilla_snapshot = setup()
    vanilla_inputs = tokenizer.apply_chat_template(prompt_group.prompts)
    
    with torch.no_grad():
        vanilla_all_ids = model.generate(
            input_ids=vanilla_input_ids,
            attention_mask=vanilla_attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    print("vanilla text generated")
    vanilla_generated_ids = vanilla_all_ids[:, vanilla_input_ids.shape[1] :]
    vanilla_generated_text = tokenizer.batch_decode(vanilla_generated_ids, skip_special_tokens=True)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    vanilla_generated_outputs = []
    for i, prompt in enumerate(prompt_group.prompts):
        generated_ids = vanilla_generated_ids[i].detach().cpu().tolist() #correct for padded?
        all_ids = vanilla_all_ids[i].detach().cpu().tolist() #correct_for padded?

        vanilla_generated_output = GeneratedOutputService.create_persisted(
            prompt_id=prompt.prompt_id,
            text=vanilla_generated_text[i],
            snapshot_id=vanilla_snapshot.snapshot_id,
            eos=bool(eos_id is not None and eos_id in generated_ids[:, vanilla_input_ids.shape[1] :]),  #correct_for padded?
            token_ids=generated_ids
        )
        vanilla_generated_outputs.append(vanilla_generated_output)

        mask_for_perplexity = create_input_mask(all_ids)
        MetricService.create_persisted(
            value=perplexity_from_token_ids(model, all_ids,mask=mask_for_perplexity), 
            description="perplexity_vanilla_on_vanilla_generated_text",
            snapshot_id=vanilla_snapshot.snapshot_id,
            generated_output_id=vanilla_generated_output.output_id,
        )

    for normalization in NORMALIZATION_VALUES:
        print(f"normalization={normalization:.1f}")
        template = ExperimentTemplateService.create_persisted(
            normalization=normalization,
            **base_template,
        )

        live_objs = []
        for vector in vectors:
            live = ExperimentLiveInstanceService.create_persisted(
                initial_vector_id=vector.vector_id,
                vector_data=vector.vector_data.detach().clone(),
                iteration_count=0,
                experiment_template_id=template.experiment_template_id,
            )
            live_obj = LiveInstance(live)
            live_obj.create_snapshot(save_vector=False) #vector is just normalized no point in saving
            
            live_objs.append(live_obj)
        batch_steer = BatchSteer(live_objs, model)
        create_detailed_metrics(vanilla_inputs,vanilla_generated_outputs,batch_steer)

def create_detailed_metrics(vanilla_inputs,vanilla_generated_outputs: List[GeneratedOutput],batch_steer):
    vanilla_input_ids = vanilla_inputs.input_ids
    vanilla_attention_mask = vanilla_inputs.attention_mask
    steered_input_ids = vanilla_input_ids.repeat(len(live_objs), 1)
    steered_attention_mask = vanilla_attention_mask.repeat(len(live_objs), 1) #padding correctness?
    with torch.no_grad(), batch_steer.steer():
        steered_all_ids = model.generate(
            input_ids=steered_input_ids,
            attention_mask=steered_attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    print("steered text generated")
    steered_generated_ids = steered_all_ids[:, vanilla_input_ids.shape[1] :] #padding correctness?
    steered_generated_text = tokenizer.batch_decode(steered_generated_ids, skip_special_tokens=True)

    for i, text in enumerate(steered_generated_text):
        live_obj = batch_steer.live_objs[i // len(prompt_group.prompts)]
        prompt = prompt_group.prompts[i % len(prompt_group.prompts)]
        all_ids = steered_all_ids[i].detach().cpu().tolist()
        generated_ids = steered_generated_ids[i].detach().cpu().tolist()

        generated_output = GeneratedOutputService.create_persisted(
            prompt_id=prompt.prompt_id,
            text=text,
            snapshot_id=live_obj.last_snapshot.snapshot_id,
            eos=bool(eos_id is not None and eos_id in generated_ids),
            token_ids = generated_ids
            )

        mask_for_perplexity = create_input_mask(all_ids)
        #metrics are not batched metrics
        MetricService.create_persisted(
            value=perplexity_from_token_ids(model, all_ids, BatchSteer([live_obj], model,mask=mask_for_perplexity)),
            description="perplexity_steered_on_steered_generated_text",
            snapshot_id=live_obj.last_snapshot.snapshot_id,
            generated_output_id=generated_output.output_id,
        )
        MetricService.create_persisted(
            value=perplexity_from_token_ids(model, all_ids,mask=mask_for_perplexity),
            description="perplexity_vanilla_on_steered_generated_text",
            snapshot_id=vanilla_snapshot.snapshot_id,
            generated_output_id=generated_output.output_id,
        )

    #create perplexity metric of steered model on vanilla outputs
    for live_obj in live_objs:
        for vanilla_generated_output in vanilla_generated_outputs:
            MetricService.create_persisted(
                value=perplexity_from_token_ids(model, vanilla_all_ids, BatchSteer([live_obj], model,mask=mask_for_perplexity)),
                description="perplexity_steered_on_vanilla_generated_text",
                snapshot_id=live_obj.last_snapshot.snapshot_id,
                generated_output_id=vanilla_generated_output.output_id,
            )

def train():
    for step in range(1, TRAIN_STEPS + 1):
        with batch_steer.steer():
            steered_output = model(
                input_ids=steered_input_ids,
                attention_mask=steered_attention_mask,
                output_hidden_states=True,
            )

        loss = batch_steer.calc_loss(steered_output, vanilla_output, vanilla_input_ids)
        loss.backward()

        for live_obj in live_objs:
            live_obj.step_optimizer()
            live_obj.iteration_count += 1

        if step % 10 == 0:
            for live_obj in live_objs:
                ExperimentLiveInstanceService.update(live_obj.live)

        if step % 50 == 0:
            for live_obj in live_objs:
                live_obj.create_snapshot(save_vector=step % 250 == 0)

        if step % 100 == 0:
            print(
                f"normalization={normalization:.1f} step={step}/{TRAIN_STEPS} "
                f"loss={float(loss.detach().item()):.6f}"
            )
        if step % 500 == 0:
            create_detailed_metrics()

if __name__ == "main":
    main()

main()