import torch,io
def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    buf.seek(0)
    return buf.getvalue()

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    buf = io.BytesIO(b)
    buf.seek(0)
    return torch.load(buf)

def make_repro_tensor(shape: Union(Tuple[int, ...],torch.Size), seed: int = 42, device: str = "cpu", dtype=torch.float32, **kwargs) -> torch.Tensor:
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

def slice_batch_output(output,sl:slice):
    output_type = type(output)
    attr_list = ["logits", "hidden_states", "attentions", "image_hidden_states"] #for mistral only

    logits = getattr(output,"logits")
    hidden_states = getattr(output,"hidden_states")
    attentions = getattr(output,"attentions")
    image_hidden_states = getattr(output,"image_hidden_states")
    
    if not (logits is None):
        logits = logits[sl]
    if not (hidden_states is None):
        hidden_states = tuple(layer[sl] for layer in hidden_states) 

    return output_type(logits=logits,hidden_states=hidden_states,attentions=attentions,image_hidden_states=image_hidden_states)

