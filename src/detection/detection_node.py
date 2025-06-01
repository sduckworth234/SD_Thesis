#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class DetectionNode:
    def __init__(self):
        rospy.init_node('detection_node', anonymous=True)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Use simple HOG + SVM person detector as fallback
        rospy.loginfo("Initializing HOG person detector...")
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            rospy.loginfo("HOG person detector initialized successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize detector: {e}")
            return
        
        # Detection parameters
        self.confidence_threshold = 0.3
        
        # Publishers
        self.detection_pub = rospy.Publisher('/detection/detections', String, queue_size=10)
        self.visualization_pub = rospy.Publisher('/detection/image_with_detections', Image, queue_size=10)
        self.status_pub = rospy.Publisher('/detection/status', String, queue_size=10)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Detection node initialized successfully")
        
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Perform detection
            detections = self.detect_objects(cv_image)
            
            # Filter for persons only
            person_detections = self.filter_person_detections(detections)
            
            # Publish detection results
            self.publish_detections(person_detections)
            
            # Create visualization
            vis_image = self.draw_detections(cv_image.copy(), person_detections)
            
            # Publish visualization
            self.publish_visualization(vis_image)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"DETECTING - Found {len(person_detections)} persons"
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in detection callback: {e}")
            status_msg = String()
            status_msg.data = f"ERROR - {str(e)}"
            self.status_pub.publish(status_msg)
    
    def detect_objects(self, image):
        """Perform person detection using HOG + SVM"""
        # Detect people using HOG
        (rects, weights) = self.hog.detectMultiScale(image, 
                                                    winStride=(4, 4),
                                                    padding=(8, 8),
                                                    scale=1.05)
        
        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            # Convert to normalized coordinates
            height, width = image.shape[:2]
            normalized_box = [y/height, x/width, (y+h)/height, (x+w)/width]
            
            detections.append({
                'box': np.array(normalized_box),
                'score': float(weights[i]) if i < len(weights) else 0.5,
                'class': 1  # person class
            })
        
        return detections
    
    def filter_person_detections(self, detections):
        """Filter detections to keep only high confidence ones"""
        # For HOG detector, we'll use a different threshold
        confidence_threshold = 0.3
        
        person_detections = []
        for detection in detections:
            if detection['score'] >= confidence_threshold:
                person_detections.append(detection)
        
        return person_detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on the image"""
        h, w, _ = image.shape
        
        for detection in detections:
            box = detection['box']
            score = detection['score']
            
            # Convert normalized coordinates to pixel coordinates
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            
            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw label
            label = f'Person: {score:.2f}'
            cv2.putText(image, label, (xmin, ymin - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def publish_detections(self, detections):
        """Publish detection results as JSON string"""
        detection_data = {
            'timestamp': rospy.Time.now().to_sec(),
            'detections': []
        }
        
        for detection in detections:
            detection_data['detections'].append({
                'class': 'person',
                'confidence': float(detection['score']),
                'bounding_box': detection['box'].tolist()
            })
        
        msg = String()
        msg.data = str(detection_data)
        self.detection_pub.publish(msg)
    
    def publish_visualization(self, image):
        """Publish visualization image"""
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.visualization_pub.publish(img_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing visualization: {e}")

    def run(self):
        """Main run loop"""
        rospy.loginfo("Detection node running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        detection_node = DetectionNode()
        detection_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection node stopped")
    except Exception as e:
        rospy.logerr(f"Detection node failed: {e}")
