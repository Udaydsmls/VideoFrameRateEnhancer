import torch

from utilities.Devices import resolve_device


def test_resolve_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_falls_back_to_cpu_when_no_accelerator(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_mps = type("MPS", (), {"is_available": staticmethod(lambda: False)})
    monkeypatch.setattr(torch.backends, "mps", fake_mps, raising=False)
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")
