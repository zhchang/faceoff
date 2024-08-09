import cv2
import dlib
import numpy as np
import argparse

def detect_face(image):
    """
    Detect a face in the given image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        dlib.rectangle: Detected face rectangle.

    Raises:
        ValueError: If no face is detected in the input image.
    """
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the input photo")
    return faces[0]

def get_face_landmarks(image, face):
    """
    Get facial landmarks for a detected face.

    Args:
        image (numpy.ndarray): Input image.
        face (dlib.rectangle): Detected face rectangle.

    Returns:
        numpy.ndarray: Array of facial landmark coordinates.
    """
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(image, face)
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
        src_landmarks = get_face_landmarks(src_image, src_face)
        dst_landmarks = get_face_landmarks(dst_image, dst_face)

        if len(src_landmarks) == 0 or len(dst_landmarks) == 0:
            raise ValueError("Failed to detect face landmarks")

        src_mask = np.zeros(src_image.shape[:2], dtype=np.float64)
        cv2.fillConvexPoly(src_mask, cv2.convexHull(src_landmarks), 1)

        (x, y, w, h) = cv2.boundingRect(dst_landmarks)

        # Ensure the bounding rectangle is within the image bounds
        if x < 0 or y < 0 or x + w > dst_image.shape[1] or y + h > dst_image.shape[0]:
            print("ROI out of bounds, skipping face swap for this frame")
            return dst_image

        x = max(0, x)
        y = max(0, y)
        w = min(w, dst_image.shape[1] - x)
        h = min(h, dst_image.shape[0] - y)

        # Calculate center point, ensuring it's within the image bounds
        center_x = min(max(x + w // 2, 0), dst_image.shape[1] - 1)
        center_y = min(max(y + h // 2, 0), dst_image.shape[0] - 1)
        center = (center_x, center_y)

        # Check if the source image is large enough for the face swap
        if src_image.shape[0] < h or src_image.shape[1] < w:
            print("Source image too small for face swap, skipping this frame")
            return dst_image

        output = cv2.seamlessClone(
            src_image, dst_image, src_mask.astype(np.uint8) * 255, center, cv2.NORMAL_CLONE
        )

        return output
    except Exception as e:
        print(f"Error in apply_face_swap: {str(e)}")
        return dst_image  # Return the original image if face swap fails

def process_video(video_path, photo_path, output_path):
    """
    Process a video by applying face swapping to each frame.

    Args:
        video_path (str): Path to the input video file.
        photo_path (str): Path to the input photo file containing the face to be swapped.
        output_path (str): Path to save the output video file.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    video = cv2.VideoCapture(video_path)
    photo = cv2.imread(photo_path)

    if not video.isOpened():
        raise ValueError("Could not open video file")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        src_face = detect_face(photo)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return

    detector = dlib.get_frontal_face_detector()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        faces = detector(frame)
        for face in faces:
            frame = apply_face_swap(photo, src_face, frame, face)

        out.write(frame)

    video.release()
    out.release()

def main():
    """
    Main function to parse command-line arguments and process the video.
    """
    parser = argparse.ArgumentParser(description="Face swap in video")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("photo", help="Path to the input photo file")
    parser.add_argument("output", help="Path to the output video file")
    args = parser.parse_args()

    try:
        process_video(args.video, args.photo, args.output)
        print("Video processing completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

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
