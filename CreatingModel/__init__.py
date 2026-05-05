from typing import Callable

from torch import nn

from CreatingModel.UNetModel import UNetInterpolator
from CreatingModel.DiffusionModel import DiffusionInterpolator
from CreatingModel.TransformerModel import TransformerInterpolator
from CreatingModel.MambaModel import MambaInterpolator

ModelFactory = Callable[..., nn.Module]

_REGISTRY: dict[str, ModelFactory] = {
    "unet": UNetInterpolator,
    "diffusion": DiffusionInterpolator,
    "transformer": TransformerInterpolator,
    "mamba": MambaInterpolator,
}


def available_architectures() -> list[str]:
    """List the names accepted by :func:`build_model`."""
    return sorted(_REGISTRY)


def build_model(name: str, **kwargs) -> nn.Module:
    """Construct a model by registry name."""
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture '{name}'. Available: {available_architectures()}"
        ) from exc
    return factory(**kwargs)


__all__ = [
    "DiffusionInterpolator",
    "MambaInterpolator",
    "TransformerInterpolator",
    "UNetInterpolator",
    "available_architectures",
    "build_model",
]
