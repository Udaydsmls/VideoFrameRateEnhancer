from pathlib import Path

from CreatingModel import build_model
from CreatingModel.TrainingModel import Trainer
from ImageOperations.GenerateFrames import generate_video_frames
from utilities.Checkpoints import Checkpoint, checkpoint_path, save_checkpoint
from utilities.Config import Config


def _config(tmp_path: Path, frames_root: Path) -> Config:
    return Config(
        root=tmp_path,
        videos=tmp_path / "videos",
        frames=frames_root,
        interpolated_frames=tmp_path / "interpolated",
        enhanced_videos=tmp_path / "enhanced",
        checkpoints=tmp_path / "checkpoints",
        architecture="unet",
        scale_factor=1.0,
        image_size=(32, 32),
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        validation_split=0.0,
        num_workers=0,
        device="cpu",
    )


def test_trainer_runs_one_epoch(tmp_path: Path, frame_directory: Path) -> None:
    config = _config(tmp_path, frame_directory)
    config.ensure_directories()
    history = Trainer(config).fit()
    assert len(history) == 1
    assert checkpoint_path(config.checkpoints, "unet", 1).is_file()


def test_generate_video_frames_writes_intermediates(tmp_path: Path, frame_directory: Path) -> None:
    config = _config(tmp_path, frame_directory)
    config.ensure_directories()
    save_checkpoint(
        Checkpoint(
            architecture="unet",
            state_dict=build_model("unet").state_dict(),
            epoch=1,
            metadata={},
        ),
        checkpoint_path(config.checkpoints, "unet", 1),
    )

    counts = generate_video_frames(
        frames_dir=config.frames,
        interpolated_frames_dir=config.interpolated_frames,
        architecture="unet",
        checkpoints_dir=config.checkpoints,
        device="cpu",
    )
    assert counts["clip"] > 0
    assert sorted((config.interpolated_frames / "clip").glob("*_interp.jpg"))
