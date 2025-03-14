
# Video Frame Rate Enhancer Pipeline



https://github.com/user-attachments/assets/f8f5117b-76b9-4375-b722-87f0403a8e67



## Overview
This project implements a terminal-based video enhancement pipeline that processes video files through several stages: data flow management, model training (or utilizing an existing model), frame generation, and video frame rate enhancement. It offers a user-friendly interface with clear instructions and options for training a new model or reusing an existing one.

The main objective of this program is to double the frame rate of any given input video.
## Features
- **User-Friendly Terminal UI:** Guides you through the setup and process.
- **Configurable Paths:** Easily update file paths via `setup.json`.
- **Flexible Model Options:** Choose to train a new model or use the latest available model.
- **Multi-Step Pipeline:** Processes video files, generates frames, and enhances video quality.

## Prerequisites
- A Virtual Environment with Python 3.10 
- All necessary dependencies listed in the project's `requirements.txt`
- Correct directory structure and file paths as defined in `setup.json`

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Udaydsmls/VideoFrameRateEnhancer.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd VideoFrameRateEnhancer
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Before running the pipeline, check the `setup.json` file to ensure all file paths are correct:
- `root`
- `metadata`
- `vid_dir`
- `frames_dir`
- `intermediate_frames_dir`
- `scale_down_frames_dir`
- `input_train_frames_dir`
- `output_train_frames_dir`
- `input_training_dataset`
- `output_training_dataset`
- `mean_std_file`
- `enhanced_videos`

If no changes are needed, simply press Enter when prompted at the start of the execution.

## Usage
Run the main script from your terminal:
```bash
python main.py
```
During execution, you'll see:
- A welcome message with instructions on checking/updating `setup.json`.
- A prompt to choose whether to train a new model or use an existing one.
- Step-by-step status messages indicating the progress of data flow, model training, frame generation, and video enhancement.

## Pipeline Workflow
1. **Data Flow Initialization:**  
   The pipeline begins by processing video files and preparing frames for further operations.
   
2. **Model Training Option:**  
   - **Train a New Model:** If you choose to train a new model, the script will initiate the training process.
   - **Use Existing Model:** Alternatively, you can opt to skip training and use the most recently saved model.
   
3. **Frame Generation:**  
   The script generates video frames using the specified or trained model.
   
4. **Video Frame Rate Enhancement:**  
   Finally, the video frame rate is enhanced to produce smoother video output.

## Modules Overview
- **FolderOperations.DataFlow:** Handles data ingestion and processing of video files.
- **CreatingModel.TrainingModel:** Manages the training process for new models.
- **ImageOperations.GenerateFrames:** Generates video frames from the processed data.
- **utilities.utils:** Contains utility functions, including model loading.
- **VideoOperations.InterpolatedImages:** Enhances video quality by increasing the frame rate.
- **setup:** Loads configuration parameters from `setup.json`.

## License
This project is licensed under the [MIT License](LICENSE).
