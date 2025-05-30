import cv2
import numpy as np
import orbslam2

class ORB_SLAM2_Wrapper:
    def __init__(self, config_file):
        self.slam_system = orbslam2.System(config_file, orbslam2.Sensor.MONOCULAR, True)
        self.slam_system.initialize()

    def process_frame(self, image, timestamp):
        self.slam_system.track_monocular(image, timestamp)

    def shutdown(self):
        self.slam_system.shutdown()

    def get_map(self):
        return self.slam_system.get_map()

    def reset(self):
        self.slam_system.reset()