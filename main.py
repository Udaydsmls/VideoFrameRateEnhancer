import sys
from pathlib import Path

from CreatingModel import available_architectures
from CreatingModel.TrainingModel import train_model
from FolderOperations.DataFlow import start_data_flow
from ImageOperations.GenerateFrames import generate_video_frames
from VideoOperations.EnhanceVideos import enhance_videos_frame_rate
from utilities.Checkpoints import find_latest_checkpoint
from utilities.Config import load_config


def _ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def _ask_choice(prompt: str, options: list[str]) -> int:
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    while True:
        answer = input("Enter your choice: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        print(f"Please enter a number from 1 to {len(options)}.")


def _ask_architecture(default: str) -> str:
    options = available_architectures()
    print(f"Available architectures: {', '.join(options)}")
    while True:
        answer = input(f"Architecture [{default}]: ").strip().lower()
        if not answer:
            return default
        if answer in options:
            return answer
        print(f"Unknown architecture. Choose one of: {', '.join(options)}")


def main() -> int:
    print("=" * 80)
    print("Welcome to the Video Frame Rate Enhancer.")
    print("=" * 80)
    print("\nIf you need to change paths or hyperparameters, edit setup.json first.")
    input("Press Enter to continue...")

    config = load_config()
    config.architecture = _ask_architecture(config.architecture)
    config.ensure_directories()

    if _ask_yes_no("\nDo you want to run the data flow (extract frames)? (y/n): "):
        print("\n[1/4] Extracting frames...")
        start_data_flow(config.videos, config.frames, scale_factor=config.scale_factor)
    else:
        print("Skipping data flow.")

    checkpoint_override: Path | None = None
    while True:
        choice = _ask_choice(
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
            break
        if choice == 2:
            print("Using the latest checkpoint.")
            break
        if choice == 3:
            entered = input("Enter absolute path to a checkpoint: ").strip()
            if entered:
                checkpoint_override = Path(entered)
                break
            print("No path provided, please try again.")
            continue
        if choice == 4:
            print("Training a new model...")
            train_model(config, resume=False)
            print("Exiting after training.")
            return 0
        if choice == 5:
            print("Continuing training the latest checkpoint...")
            train_model(config, resume=True)
            break

    if checkpoint_override is None:
        latest = find_latest_checkpoint(config.checkpoints, config.architecture)
        if latest is None:
            print(f"No checkpoint found for '{config.architecture}'. Exiting.")
            return 1
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
        return 1
    for name, n in counts.items():
        print(f"  {name}: {n} interpolated frames")

    print("\n[4/4] Assembling enhanced videos...")
    outputs = enhance_videos_frame_rate(
        config.interpolated_frames, config.enhanced_videos, config.videos
    )
    if not outputs:
        print("No matching source videos found; nothing was assembled.")
        return 1
    for name, path in outputs.items():
        print(f"  {name} -> {path}")

    print("\nAll operations completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
