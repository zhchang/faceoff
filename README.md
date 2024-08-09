# Faceoff: Video Face Swapping Tool

## Project Overview
Faceoff is a Python-based tool that allows users to swap faces in videos. It uses computer vision techniques to detect faces in each frame of a video and replace them with a specified face from an input image.

## Current State
- The core functionality for face detection and swapping is implemented.
- The tool can process videos and apply face swapping using a provided image.
- Currently, the tool is not specifically optimized for Disney princess faces.

## Usage Instructions
1. Ensure you have Python installed along with the required libraries (cv2, dlib, numpy).
2. Clone this repository to your local machine.
3. Download the shape predictor file (`shape_predictor_68_face_landmarks.dat`) and place it in the project directory.
4. Run the script using the following command:
   ```
   python main.py <input_video_path> <input_photo_path> <output_video_path>
   ```
   Where:
   - `<input_video_path>` is the path to your input video file
   - `<input_photo_path>` is the path to the image containing the face you want to swap
   - `<output_video_path>` is the path where the processed video will be saved

## Requirements
- Python 3.x
- OpenCV (cv2)
- dlib
- numpy
- A clear, frontal-facing image of the face to be swapped (e.g., a Disney princess image)

## Pending Tasks
1. Obtain and integrate a Disney princess image for face swapping.
2. Optimize face detection and swapping for cartoon-style faces.
3. Implement error handling for various edge cases (e.g., no faces detected, multiple faces).
4. Enhance performance for processing longer videos.
5. Add support for multiple face swapping options.

## Note
This tool is currently in development. The face swapping functionality works best with clear, front-facing images. Future updates will focus on improving compatibility with Disney princess-style faces and overall performance enhancements.
