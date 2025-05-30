def load_image(image_path):
    """Load an image from the specified path."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def save_image(image, save_path):
    """Save an image to the specified path."""
    import cv2
    cv2.imwrite(save_path, image)

def preprocess_image(image, size=(416, 416)):
    """Preprocess the image for YOLO model."""
    import cv2
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def postprocess_detections(detections, confidence_threshold=0.5):
    """Filter detections based on confidence threshold."""
    return [d for d in detections if d['confidence'] >= confidence_threshold]

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image."""
    import cv2
    for detection in detections:
        x, y, w, h = detection['box']
        label = detection['label']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image