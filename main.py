import cv2
import dlib
import numpy as np
import argparse
import logging

# TODO: Pending Integration - Disney Princess Image
# This script will be updated to include functionality for changing faces
# in videos to Disney princess faces. The integration is pending user
# authorization to obtain and use Disney princess images.
#
# Future updates will include:
# 1. Loading and preprocessing of Disney princess images
# 2. Adapting face detection and landmark detection for cartoon-style faces
# 3. Enhancing face swapping algorithm for smoother integration with animated features

def detect_face(image):
    """
    Detect a face in the given image using dlib's frontal face detector.

    This function uses dlib's pre-trained frontal face detector to identify faces
    in the input image. It's designed to work best with frontal faces but can also
    detect faces at different angles to some extent.

    Args:
        image (numpy.ndarray): Input image in BGR color format (as read by OpenCV).

    Returns:
        dlib.rectangle: Detected face rectangle. This object contains the
        coordinates of the bounding box around the detected face.

    Raises:
        ValueError: If no face is detected in the input image. This can happen
        if the image doesn't contain a face or if the face is not clearly visible.

    Note:
        This function returns only the first detected face. If multiple faces
        are present in the image, only the first one (usually the largest or
        most prominent) will be returned.
    """
    # Create an instance of dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Apply the detector to the image
    # The second argument (1) is the number of times to upsample the image
    # Upsampling can help detect smaller faces but increases processing time
    faces = detector(image, 1)

    # Check if any faces were detected
    if len(faces) == 0:
        raise ValueError("No face detected in the input photo")

    # Return the first detected face
    return faces[0]

def get_face_landmarks(image, face):
    """
    Get facial landmarks for a detected face using dlib's shape predictor.

    This function uses a pre-trained facial landmark predictor to identify
    68 specific points on a face, such as the corners of the eyes, nose, and mouth.

    Args:
        image (numpy.ndarray): Input image containing the face.
        face (dlib.rectangle): Detected face rectangle from the image.

    Returns:
        numpy.ndarray: Array of 68 facial landmark coordinates as (x, y) pairs.

    Note:
        This function requires the "shape_predictor_68_face_landmarks.dat" file,
        which should be present in the same directory as the script.
    """
    # Load the pre-trained facial landmark predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Predict the 68 facial landmarks
    landmarks = predictor(image, face)

    # Convert the landmark points to a numpy array of (x, y) coordinates
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def apply_face_swap(src_image, src_face, dst_image, dst_face):
    """
    Apply face swapping between source and destination images.

    Args:
        src_image (numpy.ndarray): Source image containing the face to be swapped.
        src_face (dlib.rectangle): Detected face in the source image.
        dst_image (numpy.ndarray): Destination image where the face will be swapped.
        dst_face (dlib.rectangle): Detected face in the destination image.

    Returns:
        numpy.ndarray: Image with the face swapped, or original destination image if swap fails.
    """
    try:
        logging.debug("Starting face swap process")

        # Step 1: Get facial landmarks for both source and destination faces
        src_landmarks = get_face_landmarks(src_image, src_face)
        dst_landmarks = get_face_landmarks(dst_image, dst_face)

        # Ensure landmarks were successfully detected
        if len(src_landmarks) == 0 or len(dst_landmarks) == 0:
            logging.error("Failed to detect face landmarks")
            raise ValueError("Failed to detect face landmarks")

        logging.debug(f"Detected landmarks - Source: {len(src_landmarks)}, Destination: {len(dst_landmarks)}")

        # Step 2: Create a mask for the source face
        src_mask = np.zeros(src_image.shape[:2], dtype=np.float64)
        cv2.fillConvexPoly(src_mask, cv2.convexHull(src_landmarks), 1)
        logging.debug("Created source face mask")

        # Step 3: Get the bounding rectangle of the destination face
        (x, y, w, h) = cv2.boundingRect(dst_landmarks)
        logging.debug(f"Destination face bounding rectangle: x={x}, y={y}, w={w}, h={h}")

        # Step 4: Ensure the bounding rectangle is within the image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, dst_image.shape[1] - x)
        h = min(h, dst_image.shape[0] - y)
        logging.debug(f"Adjusted bounding rectangle: x={x}, y={y}, w={w}, h={h}")

        # Step 5: Calculate the center point for seamless cloning
        center_x = x + w // 2
        center_y = y + h // 2
        center = (center_x, center_y)
        logging.debug(f"Seamless cloning center point: {center}")

        # Step 6: Resize source image if it's too small
        if src_image.shape[0] < h or src_image.shape[1] < w:
            scale_factor = max(h / src_image.shape[0], w / src_image.shape[1])
            new_size = (int(src_image.shape[1] * scale_factor), int(src_image.shape[0] * scale_factor))
            src_image = cv2.resize(src_image, new_size, interpolation=cv2.INTER_LINEAR)
            src_mask = cv2.resize(src_mask, new_size, interpolation=cv2.INTER_LINEAR)
            logging.debug(f"Resized source image to {new_size}")

        # Step 7: Crop the source image and mask to match the destination face size
        src_face_img = src_image[max(0, center_y - h//2):min(src_image.shape[0], center_y + h//2),
                                 max(0, center_x - w//2):min(src_image.shape[1], center_x + w//2)]
        src_face_mask = src_mask[max(0, center_y - h//2):min(src_mask.shape[0], center_y + h//2),
                                 max(0, center_x - w//2):min(src_mask.shape[1], center_x + w//2)]

        # Ensure the cropped images have the correct dimensions
        if src_face_img.shape[:2] != (h, w):
            src_face_img = cv2.resize(src_face_img, (w, h), interpolation=cv2.INTER_LINEAR)
            src_face_mask = cv2.resize(src_face_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            logging.debug(f"Resized cropped face image to {src_face_img.shape[:2]}")

        # TODO: Add Disney princess image processing logic
        # This section will be implemented once we receive authorization and the Disney princess image.
        # Potential steps may include:
        # - Adjusting the color palette to match the Disney style
        # - Enhancing facial features to resemble a cartoon character
        # - Applying additional filters or effects to create a more animated look

        # Step 8: Apply seamless cloning to swap the faces
        logging.debug("Applying seamless cloning")
        output = cv2.seamlessClone(
            src_face_img, dst_image, (src_face_mask * 255).astype(np.uint8), center, cv2.NORMAL_CLONE
        )

        logging.debug("Face swap completed successfully")
        return output
    except Exception as e:
        logging.error(f"Error in apply_face_swap: {str(e)}")
        return dst_image  # Return the original image if face swap fails

def process_video(video_path, photo_path, output_path):
    """
    Process a video by applying face swapping to each frame.

    This function reads a video file, detects faces in each frame, and applies face swapping
    using a provided photo. The resulting video is then saved to the specified output path.

    Args:
        video_path (str): Path to the input video file.
        photo_path (str): Path to the input photo file containing the face to be swapped.
        output_path (str): Path to save the output video file.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    logging.info(f"Starting video processing: {video_path}")

    # Open the video file and read the photo
    video = cv2.VideoCapture(video_path)
    photo = cv2.imread(photo_path)

    # Check if the video file was successfully opened
    if not video.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        raise ValueError("Could not open video file")

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Detect the face in the input photo
    try:
        src_face = detect_face(photo)
        logging.info("Successfully detected face in input photo")
    except ValueError as e:
        logging.error(f"Error detecting face in input photo: {str(e)}")
        return

    # Initialize the face detector for video frames
    detector = dlib.get_frontal_face_detector()

    # Process each frame of the video
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            logging.info("Reached end of video")
            break  # End of video

        frame_count += 1
        if frame_count % 100 == 0:
            logging.info(f"Processing frame {frame_count}/{total_frames}")

        # Detect faces in the current frame
        faces = detector(frame)
        logging.debug(f"Detected {len(faces)} faces in frame {frame_count}")

        # Apply face swapping to each detected face
        for i, face in enumerate(faces):
            logging.debug(f"Applying face swap to face {i+1} in frame {frame_count}")
            frame = apply_face_swap(photo, src_face, frame, face)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    video.release()
    out.release()
    logging.info(f"Video processing completed. Output saved to: {output_path}")

def main():
    """
    Main function to parse command-line arguments and process the video.
    """
    parser = argparse.ArgumentParser(description="Face swap in video")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("photo", help="Path to the input photo file")
    parser.add_argument("output", help="Path to the output video file")
    parser.add_argument("--log", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)", default="INFO")
    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        logging.info("Starting video processing")
        process_video(args.video, args.photo, args.output)
        logging.info("Video processing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()

# Test cases:
# 1. Valid input:
#    python main.py input_video.mp4 input_photo.jpg output_video.mp4
#
# 2. Invalid video file:
#    python main.py nonexistent_video.mp4 input_photo.jpg output_video.mp4
#
# 3. Invalid photo file:
#    python main.py input_video.mp4 nonexistent_photo.jpg output_video.mp4
#
# 4. No face in input photo:
#    python main.py input_video.mp4 no_face_photo.jpg output_video.mp4
#
# 5. No faces in video:
#    python main.py no_face_video.mp4 input_photo.jpg output_video.mp4
#
# 6. Multiple faces in video:
#    python main.py multiple_faces_video.mp4 input_photo.jpg output_video.mp4
#
# 7. Very small input photo:
#    python main.py input_video.mp4 small_photo.jpg output_video.mp4
#
# 8. High-resolution video:
#    python main.py high_res_video.mp4 input_photo.jpg output_video.mp4
#
# 9. Low-resolution video:
#    python main.py low_res_video.mp4 input_photo.jpg output_video.mp4
#
# 10. Different video formats (e.g., .avi, .mov):
#     python main.py input_video.avi input_photo.jpg output_video.mp4
#     python main.py input_video.mov input_photo.jpg output_video.mp4
