import rospy
from std_msgs.msg import String
from slam.orbslam2_wrapper import ORB_SLAM2
from camera.realsense_capture import RealSenseCapture

class SLAMNode:
    def __init__(self):
        rospy.init_node('slam_node', anonymous=True)
        self.slam_system = ORB_SLAM2()
        self.camera = RealSenseCapture()
        self.image_pub = rospy.Publisher('/slam/image', String, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz

    def run(self):
        while not rospy.is_shutdown():
            image = self.camera.capture_image()
            if image is not None:
                self.slam_system.process_image(image)
                self.image_pub.publish(image)
            self.rate.sleep()

if __name__ == '__main__':
    slam_node = SLAMNode()
    try:
        slam_node.run()
    except rospy.ROSInterruptException:
        pass