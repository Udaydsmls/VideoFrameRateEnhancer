import sys
import FolderOperations.DataFlow as df
import CreatingModel.TrainingModel as tm
import ImageOperations.GenerateFrames as gen
import utilities.utils as utils
import VideoOperations.InterpolatedImages as ii
import setup


def main():
    """ Running the Program."""
    print("=" * 80)
    print("Welcome to the Video Enhancement Pipeline!")
    print("=" * 80)
    print("\nBefore you begin, if you want to change any file paths, please update them in setup.json.")
    input("If everything is correct, just press Enter to continue...")

    setup.setup()
    paths = setup.get_paths()
    values = setup.get_values()

    vid_dir = paths["vid_dir"]
    frames_dir = paths["frames_dir"]
    intermediate_frames_dir = paths["intermediate_frames_dir"]
    scale_down_frames_dir = paths["scale_down_frames_dir"]
    input_train_frames_dir = paths["input_train_frames_dir"]
    output_train_frames_dir = paths["output_train_frames_dir"]
    input_training_dataset = paths["input_training_dataset"]
    output_training_dataset = paths["output_training_dataset"]
    enhanced_videos = paths["enhanced_videos"]

    batch_size = values["batch_size"]
    scale_down_factor = values["scale_down_factor"]

    while True:
        do_dataflow = input("\nDo you want to run the data flow? (y/n): ").strip().lower()
        if do_dataflow in ("y", "yes", "n", "no"):
            break
        print("Invalid input. Please enter 'y' or 'n'.")
    if do_dataflow in ("y", "yes"):
        print("\n[1/4] Starting Data Flow...")

        while True:
            create_training = input("Do you want to create a training dataset? (y/n): ").strip().lower()
            if create_training in ("y", "yes", "n", "no"):
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        create_training_flag = create_training in ("y", "yes")

        success = df.start_data_flow(
            vid_dir, frames_dir, scale_down_frames_dir,
            input_train_frames_dir, output_train_frames_dir,
            input_training_dataset, output_training_dataset,
            batch_size, scale_down_factor, create_training_dataset=create_training_flag
        )
        if not success:
            sys.exit("Data flow failed. Exiting.")
        print("Data flow completed successfully.")
    else:
        print("Skipping data flow as per user selection.")

    model_path = ""

    while True:
        print("\nWould you like to:")
        print("  1. Train a new model")
        print("  2. Use the existing model")
        print("  3. Give path to an existing model")
        print("  4. Train a new model and exit")
        print("  5. Continue Training on previous model")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            print("\n[2/4] Training the Model...")
            tm.train_model()
            break
        elif choice == "2":
            print("\nSkipping model training. Using the existing model.")
            break
        elif choice == "3":
            print("\nSkipping model training. Give path to an existing model.")
            model_path = input("Enter absolute path to an existing model: ")
            break
        elif choice == "4":
            print("\n[2/4] Training the Model...")
            tm.train_model()
            print("\nExiting...")
            return
        elif choice == "5":
            print("\n[2/4] Continue training the Model...")
            tm.train_model(True)
            break
        else:
            print("Invalid input. Please enter a number from 1 to 5.")

    print("\n[3/4] Loading the Model...")
    if model_path == "":
        model_path = utils.load_latest_model()
    if not model_path:
        print("No model found. Exiting.")
        sys.exit(1)
    print(f"Model loaded successfully: {model_path}")

    print("\n[4/4] Generating Video Frames...")
    gen.generate_video_frames(scale_down_frames_dir, model_path, intermediate_frames_dir)
    print("Video frames generated successfully.")

    print("\nEnhancing video frame rate...")
    ii.enhance_videos_frame_rate(scale_down_frames_dir, enhanced_videos)
    print("Video enhancement completed successfully.")

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()
