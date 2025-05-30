import rospy
from std_msgs.msg import String
from detection.yolo_v4.person_detector import PersonDetector
from camera.realsense_capture import RealSenseCapture

class DetectionNode:
    def __init__(self):
        rospy.init_node('detection_node', anonymous=True)
        self.detector = PersonDetector()
        self.camera = RealSenseCapture()
        self.pub = rospy.Publisher('detection_results', String, queue_size=10)
        rospy.loginfo("Detection Node Initialized")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            frame = self.camera.capture_frame()
            detections = self.detector.detect(frame)
            self.pub.publish(detections)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = DetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass