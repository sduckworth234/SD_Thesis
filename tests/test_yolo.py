#!/usr/bin/env python3
"""
Test script for YOLO v4 person detection functionality
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import sys

class YOLOTest:
    def __init__(self):
        rospy.init_node('yolo_test', anonymous=True)
        self.bridge = CvBridge()
        
        # YOLO configuration
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo')
        self.config_file = os.path.join(self.model_path, 'yolov4.cfg')
        self.weights_file = os.path.join(self.model_path, 'yolov4.weights')
        self.names_file = os.path.join(self.model_path, 'coco.names')
        
        # Load YOLO
        self.net = None
        self.output_layers = None
        self.classes = None
        self.load_yolo()
        
        # ROS subscriber
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        rospy.loginfo("YOLO test node initialized")
    
    def load_yolo(self):
        """Load YOLO model and configuration"""
        try:
            # Check if files exist
            if not os.path.exists(self.config_file):
                rospy.logerr(f"YOLO config file not found: {self.config_file}")
                return False
            
            if not os.path.exists(self.weights_file):
                rospy.logerr(f"YOLO weights file not found: {self.weights_file}")
                rospy.loginfo("Run: wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
                return False
            
            if not os.path.exists(self.names_file):
                rospy.logerr(f"YOLO names file not found: {self.names_file}")
                return False
            
            # Load YOLO
            rospy.loginfo("Loading YOLO model...")
            self.net = cv2.dnn.readNet(self.weights_file, self.config_file)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(self.names_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            rospy.loginfo("✓ YOLO model loaded successfully")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error loading YOLO: {e}")
            return False
    
    def detect_objects(self, image):
        """Detect objects in image using YOLO"""
        if self.net is None:
            return image, []
        
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        
        # Run detection
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw bounding boxes
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': [x, y, w, h]
                })
                
                # Draw bounding box and label
                color = (0, 255, 0) if label == 'person' else (255, 0, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image, detections
    
    def image_callback(self, msg):
        """Process incoming image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run detection
            result_image, detections = self.detect_objects(cv_image)
            
            # Count person detections
            person_count = sum(1 for d in detections if d['label'] == 'person')
            
            if person_count > 0:
                rospy.loginfo(f"Detected {person_count} person(s)")
            
            # Display result
            cv2.imshow("YOLO Detection", result_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def run_test(self):
        """Run YOLO test"""
        if self.net is None:
            rospy.logerr("YOLO model not loaded. Cannot run test.")
            return False
        
        rospy.loginfo("Starting YOLO detection test...")
        rospy.loginfo("Make sure camera is running: roslaunch realsense2_camera rs_camera.launch")
        
        # Test with sample image if available
        sample_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_images', 'test_person.jpg')
        
        if os.path.exists(sample_image_path):
            rospy.loginfo("Testing with sample image...")
            image = cv2.imread(sample_image_path)
            result_image, detections = self.detect_objects(image)
            
            person_count = sum(1 for d in detections if d['label'] == 'person')
            rospy.loginfo(f"Sample image test: {person_count} person(s) detected")
            
            cv2.imshow("Sample Detection", result_image)
            cv2.waitKey(3000)  # Show for 3 seconds
        
        # Wait for live camera feed
        rospy.loginfo("Waiting for camera feed...")
        rospy.spin()

def test_yolo_standalone():
    """Test YOLO without ROS"""
    rospy.loginfo("Testing YOLO standalone functionality...")
    
    try:
        # Test YOLO loading
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolo')
        config_file = os.path.join(model_path, 'yolov4.cfg')
        weights_file = os.path.join(model_path, 'yolov4.weights')
        
        if not os.path.exists(config_file) or not os.path.exists(weights_file):
            rospy.logerr("YOLO files not found. Run setup script first.")
            return False
        
        net = cv2.dnn.readNet(weights_file, config_file)
        rospy.loginfo("✓ YOLO model loaded successfully")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(dummy_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        outputs = net.forward(output_layers)
        rospy.loginfo("✓ YOLO inference test passed")
        
        return True
        
    except Exception as e:
        rospy.logerr(f"YOLO standalone test failed: {e}")
        return False

if __name__ == '__main__':
    try:
        # First test standalone functionality
        standalone_test = test_yolo_standalone()
        
        if standalone_test:
            # Test with ROS integration
            tester = YOLOTest()
            tester.run_test()
        else:
            rospy.logerr("Standalone YOLO test failed")
        
        cv2.destroyAllWindows()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test failed with error: {e}")
