import pytest
import torch

from CreatingModel import available_architectures, build_model
from CreatingModel.Losses import L1SSIMLoss


@pytest.mark.parametrize("name", ["unet", "diffusion", "transformer", "mamba"])
def test_models_round_trip_shape(name: str) -> None:
    model = build_model(name)
    model.eval()
    f1 = torch.rand(1, 3, 32, 32)
    f2 = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = model(f1, f2)
    assert out.shape == (1, 3, 32, 32)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1).all()


def test_registry_lists_all_architectures() -> None:
    assert set(available_architectures()) == {"unet", "diffusion", "transformer", "mamba"}


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError):
        build_model("not-a-real-arch")


def test_l1_ssim_loss_is_lower_for_perfect_match() -> None:
    target = torch.rand(2, 3, 16, 16)
    loss = L1SSIMLoss()
    assert loss(target.clone(), target).item() < loss(torch.rand_like(target), target).item()


def test_diffusion_accepts_noise_level() -> None:
    model = build_model("diffusion")
    f1 = torch.rand(2, 3, 32, 32)
    f2 = torch.rand(2, 3, 32, 32)
    out = model(f1, f2, noise_level=torch.rand(2))
    assert out.shape == f1.shape
