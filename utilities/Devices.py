import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Pick a torch device, honouring an explicit string or auto-detecting."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
