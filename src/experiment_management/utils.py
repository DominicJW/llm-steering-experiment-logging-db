import io
import json
from typing import Any, Callable, Dict, Tuple, Type, Union

import torch
from torch import optim

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    return buf.getvalue()

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    buf = io.BytesIO(b)
    buf.seek(0)
    return torch.load(buf)

def make_repro_tensor(
    shape: Union[Tuple[int, ...], torch.Size],
    seed: int = 42,
    device: str = "cpu",
    dtype=torch.float32,
    **kwargs
) -> torch.Tensor:
    cpu_state = torch.get_rng_state()
    cuda_state = None
    if device.startswith("cuda"):
        cuda_state = torch.cuda.get_rng_state_all()
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    x = torch.randn(shape, device=device, dtype=dtype, **kwargs)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return x

def slice_batch_output(output, sl: slice):
    output_type = type(output)


    logits = getattr(output, "logits", None)
    hidden_states = getattr(output, "hidden_states", None)
    attentions = getattr(output, "attentions", None)
    #mistral only
    image_hidden_states = getattr(output, "image_hidden_states", None)
    image_hidden_states = None
    if not (logits is None):
        logits = logits[sl]
    if not (hidden_states is None):
        hidden_states = tuple(layer[sl] for layer in hidden_states) 
    

    return output_type(
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        image_hidden_states=image_hidden_states,
    )


loss_registry: Dict[str, Callable] = {}

def register_loss_fn(name: str, function: Callable):
    loss_registry[name] = function


def loss_factory(name: str, **kwargs):
    loss_fn = loss_registry[name]
    return lambda steered_output, vanilla_output, input_ids: loss_fn(
        steered_output, vanilla_output, input_ids, **kwargs
    )
    
    
optimizer_registry: Dict[str, Type[optim.Optimizer]] = {"AdamW": optim.AdamW}

def optimizer_factory(name: str, **kwargs):
    optimizer_class = optimizer_registry[name]
    return lambda parameter_list: optimizer_class(parameter_list, **kwargs)
