import cv2
import numpy as np
import tensorflow as tf

class PersonDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = tf.saved_model.load(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image):
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)

        return self.process_detections(detections)

    def process_detections(self, detections):
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        detected_persons = []
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                detected_persons.append({
                    'box': boxes[i],
                    'score': scores[i],
                    'class': int(classes[i])
                })

        return detected_persons

    def draw_detections(self, image, detections):
        for detection in detections:
            box = detection['box']
            score = detection['score']
            class_id = detection['class']

            # Convert box coordinates from normalized to pixel values
            h, w, _ = image.shape
            ymin, xmin, ymax, xmax = (box * np.array([h, w, h, w])).astype(int)

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f'Class: {class_id}, Score: {score:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image