import sys
from pathlib import Path

from CreatingModel import available_architectures
from CreatingModel.TrainingModel import train_model
from FolderOperations.DataFlow import start_data_flow
from ImageOperations.GenerateFrames import generate_video_frames
from VideoOperations.EnhanceVideos import enhance_videos_frame_rate
from utilities.Checkpoints import find_latest_checkpoint
from utilities.Config import Config, load_config
from utilities.Prompts import ask_choice, ask_from_set, ask_yes_no


def _print_banner() -> None:
    print("=" * 80)
    print("Welcome to the Video Frame Rate Enhancer.")
    print("=" * 80)
    print("\nIf you need to change paths or hyperparameters, edit setup.json first.")
    input("Press Enter to continue...")


def _stage_data_flow(config: Config) -> None:
    if not ask_yes_no("\nDo you want to run the data flow (extract frames)? (y/n): "):
        print("Skipping data flow.")
        return
    print("\n[1/4] Extracting frames...")
    start_data_flow(config.videos, config.frames, scale_factor=config.scale_factor)


def _stage_model(config: Config) -> tuple[bool, Path | None]:
    """Returns ``(should_continue, checkpoint_override)``."""
    while True:
        choice = ask_choice(
            "\n[2/4] Model:",
            [
                "Train a new model",
                "Use the latest checkpoint",
                "Provide a path to a checkpoint",
                "Train a new model and exit",
                "Continue training the latest checkpoint",
            ],
        )
        if choice == 1:
            print("Training a new model...")
            train_model(config, resume=False)
            return True, None
        if choice == 2:
            print("Using the latest checkpoint.")
            return True, None
        if choice == 3:
            entered = input("Enter absolute path to a checkpoint: ").strip()
            if entered:
                return True, Path(entered)
            print("No path provided, please try again.")
            continue
        if choice == 4:
            print("Training a new model...")
            train_model(config, resume=False)
            print("Exiting after training.")
            return False, None
        if choice == 5:
            print("Continuing training the latest checkpoint...")
            train_model(config, resume=True)
            return True, None


def _stage_generate(config: Config, checkpoint_override: Path | None) -> bool:
    if checkpoint_override is None:
        latest = find_latest_checkpoint(config.checkpoints, config.architecture)
        if latest is None:
            print(f"No checkpoint found for '{config.architecture}'. Exiting.")
            return False
        print(f"Using checkpoint: {latest}")
        checkpoint_override = latest

    print("\n[3/4] Generating intermediate frames...")
    counts = generate_video_frames(
        frames_dir=config.frames,
        interpolated_frames_dir=config.interpolated_frames,
        architecture=config.architecture,
        checkpoints_dir=config.checkpoints,
        checkpoint=checkpoint_override,
        device=config.device,
    )
    if not counts:
        print("No videos to interpolate; check your frames directory.")
        return False
    for name, n in counts.items():
        print(f"  {name}: {n} interpolated frames")
    return True


def _stage_assemble(config: Config) -> bool:
    print("\n[4/4] Assembling enhanced videos...")
    outputs = enhance_videos_frame_rate(
        config.interpolated_frames, config.enhanced_videos, config.videos
    )
    if not outputs:
        print("No matching source videos found; nothing was assembled.")
        return False
    for name, path in outputs.items():
        print(f"  {name} -> {path}")
    return True


def main() -> int:
    _print_banner()

    config = load_config()
    config.architecture = ask_from_set(
        "Architecture", available_architectures(), default=config.architecture
    )
    config.ensure_directories()

    _stage_data_flow(config)

    keep_going, checkpoint_override = _stage_model(config)
    if not keep_going:
        return 0

    if not _stage_generate(config, checkpoint_override):
        return 1
    if not _stage_assemble(config):
        return 1

    print("\nAll operations completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
