import unittest
from src.detection.yolo_v4.person_detector import PersonDetector
from src.detection.utils import preprocess_image

class TestPersonDetector(unittest.TestCase):

    def setUp(self):
        self.detector = PersonDetector()
        self.test_image = "path/to/test/image.jpg"  # Replace with actual test image path

    def test_preprocess_image(self):
        processed_image = preprocess_image(self.test_image)
        self.assertIsNotNone(processed_image)
        self.assertEqual(processed_image.shape[0], 416)  # Assuming YOLO expects 416x416 images
        self.assertEqual(processed_image.shape[1], 416)

    def test_detect_person(self):
        detections = self.detector.detect(self.test_image)
        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)  # Assuming there should be at least one detection

if __name__ == '__main__':
    unittest.main()