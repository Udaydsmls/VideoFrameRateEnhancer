from pathlib import Path

import torch

from utilities.Checkpoints import (
    Checkpoint,
    checkpoint_path,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _make_checkpoint(arch: str, epoch: int) -> Checkpoint:
    return Checkpoint(
        architecture=arch,
        state_dict={"weight": torch.tensor([float(epoch)])},
        epoch=epoch,
        metadata={"note": f"epoch {epoch}"},
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    path = checkpoint_path(tmp_path, "unet", 7)
    save_checkpoint(_make_checkpoint("unet", 7), path)
    loaded = load_checkpoint(path)
    assert loaded.architecture == "unet"
    assert loaded.epoch == 7
    assert torch.equal(loaded.state_dict["weight"], torch.tensor([7.0]))
    assert loaded.metadata == {"note": "epoch 7"}


def test_find_latest_picks_highest_epoch(tmp_path: Path) -> None:
    for epoch in (3, 1, 9, 5):
        save_checkpoint(_make_checkpoint("unet", epoch), checkpoint_path(tmp_path, "unet", epoch))
    latest = find_latest_checkpoint(tmp_path, "unet")
    assert latest is not None and latest.name == "unet_epoch0009.pt"


def test_find_latest_filters_architecture(tmp_path: Path) -> None:
    save_checkpoint(_make_checkpoint("unet", 1), checkpoint_path(tmp_path, "unet", 1))
    save_checkpoint(_make_checkpoint("transformer", 9), checkpoint_path(tmp_path, "transformer", 9))
    assert find_latest_checkpoint(tmp_path, "unet").name == "unet_epoch0001.pt"
    assert find_latest_checkpoint(tmp_path, "mamba") is None


def test_find_latest_ignores_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "weights_unet.pt").write_text("x")
    save_checkpoint(_make_checkpoint("unet", 2), checkpoint_path(tmp_path, "unet", 2))
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None and latest.name == "unet_epoch0002.pt"


def test_find_latest_missing_directory(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path / "nowhere") is None
