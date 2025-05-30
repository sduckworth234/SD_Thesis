import unittest
from src.camera.realsense_capture import capture_images
from src.detection.yolo_v4.person_detector import detect_persons
from src.enhancement.zero_dce.enhance import enhance_image
from src.slam.orbslam2_wrapper import initialize_slam

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.camera = capture_images()
        self.image = self.camera.get_image()
        self.enhanced_image = enhance_image(self.image)
        self.detections = detect_persons(self.enhanced_image)
        self.slam_system = initialize_slam()

    def test_image_capture(self):
        self.assertIsNotNone(self.image, "Image capture failed.")

    def test_image_enhancement(self):
        self.assertIsNotNone(self.enhanced_image, "Image enhancement failed.")
        self.assertNotEqual(self.image, self.enhanced_image, "Enhanced image should differ from original.")

    def test_person_detection(self):
        self.assertIsInstance(self.detections, list, "Detection results should be a list.")
        self.assertGreater(len(self.detections), 0, "No persons detected in the enhanced image.")

    def test_slam_initialization(self):
        self.assertIsNotNone(self.slam_system, "SLAM system initialization failed.")

if __name__ == '__main__':
    unittest.main()