import cv2
import numpy as np

def enhance_image(image):
    """
    Enhance the input image using the SCI model.
    
    Parameters:
    image (numpy.ndarray): Input low-light image to be enhanced.
    
    Returns:
    numpy.ndarray: Enhanced image.
    """
    # Placeholder for SCI enhancement logic
    enhanced_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Example operation
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
    return enhanced_image

def process_video(video_path):
    """
    Process a video file and enhance each frame using the SCI model.
    
    Parameters:
    video_path (str): Path to the input video file.
    """
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        enhanced_frame = enhance_image(frame)
        cv2.imshow('Enhanced Frame', enhanced_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()