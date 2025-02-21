# from GenerateFrames_old import generating_frames
# from InterpolatedImages import create_video_from_frames, get_video_frame_rate
# from MovingBackFiles_old import move_back_files
# import os
#
# def create_video(input_dir, model_path, output_dir, video_name, video_dir, img_height, img_width, num_channels):
#     os.makedirs(output_dir, exist_ok=True)
#     generating_frames(input_dir, model_path, output_dir, img_height, img_width, num_channels)
#     move_back_files(input_dir, output_dir, output_dir)
#     frame_rate = get_video_frame_rate(video_dir)
#     create_video_from_frames(output_dir, video_name, frame_rate)
