import rospy
from sensor_msgs.msg import Image
from camera.realsense_capture import RealSenseCapture

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=True)
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        self.camera = RealSenseCapture()

    def run(self):
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            image = self.camera.get_image()
            if image is not None:
                self.image_pub.publish(image)
            rate.sleep()

if __name__ == '__main__':
    try:
        camera_node = CameraNode()
        camera_node.run()
    except rospy.ROSInterruptException:
        pass