#!/usr/bin/env python3
"""
Detection Node for YOLO v4 Person Detection
Subscribes to camera images and publishes detection results.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import json
import time
import threading

# YOLO v4 imports
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import detection
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    rospy.logwarn("PyTorch not available, using OpenCV DNN fallback")

class DetectionNode:
    def __init__(self):
        rospy.init_node('detection_node', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self.nms_threshold = rospy.get_param('~nms_threshold', 0.4)
        self.input_size = rospy.get_param('~input_size', 416)
        self.weights_path = rospy.get_param('~weights_path', 'models/yolo_v4/yolov4.weights')
        self.config_path = rospy.get_param('~config_path', 'models/yolo_v4/yolov4.cfg')
        self.classes_path = rospy.get_param('~classes_path', 'models/yolo_v4/coco.names')
        
        # Load class names
        self.load_classes()
        
        # Initialize detector
        self.load_model()
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        
        # Publishers
        self.detections_pub = rospy.Publisher('/detection/persons', Float32MultiArray, queue_size=10)
        self.annotated_pub = rospy.Publisher('/detection/annotated_image', Image, queue_size=1)
        self.markers_pub = rospy.Publisher('/detection/markers', MarkerArray, queue_size=10)
        self.stats_pub = rospy.Publisher('/detection/statistics', String, queue_size=10)
        
        # Performance monitoring
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        self.processing_times = []
        
        # Threading
        self.detection_lock = threading.Lock()
        self.latest_image = None
        self.image_timestamp = None
        
        rospy.loginfo("Detection Node Initialized")
    
    def load_classes(self):
        """Load COCO class names"""
        try:
            with open(self.classes_path, 'r') as f:
                self.classes = f.read().strip().split('\n')
        except FileNotFoundError:
            rospy.logwarn(f"Classes file not found: {self.classes_path}, using default COCO classes")
            self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
        
        # Find person class index
        self.person_class_id = 0  # Person is typically class 0 in COCO
        if 'person' in self.classes:
            self.person_class_id = self.classes.index('person')
    
    def load_model(self):
        """Load YOLO v4 model"""
        try:
            if TORCH_AVAILABLE:
                self.load_torch_model()
            else:
                self.load_opencv_model()
        except Exception as e:
            rospy.logerr(f"Failed to load detection model: {e}")
            self.use_dummy_detector()
    
    def load_torch_model(self):
        """Load PyTorch YOLO model"""
        # For simplicity, using a pre-trained model
        # In production, you would load custom YOLO v4 weights
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        
        rospy.loginfo("PyTorch detection model loaded")
    
    def load_opencv_model(self):
        """Load OpenCV DNN YOLO model"""
        try:
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            rospy.loginfo("OpenCV DNN YOLO model loaded")
        except Exception as e:
            rospy.logwarn(f"Failed to load OpenCV model: {e}")
            self.use_dummy_detector()
    
    def use_dummy_detector(self):
        """Fallback dummy detector for testing"""
        rospy.logwarn("Using dummy detector - no actual detection will be performed")
        self.net = None
        self.model = None
    
    def image_callback(self, msg):
        """Process incoming image messages"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.detection_lock:
                self.latest_image = cv_image.copy()
                self.image_timestamp = msg.header.stamp
            
            # Process detection in separate thread to avoid blocking
            threading.Thread(target=self.process_detection, args=(cv_image, msg.header.stamp)).start()
            
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")
    
    def process_detection(self, image, timestamp):
        """Process detection on image"""
        start_time = time.time()
        
        try:
            if hasattr(self, 'model') and self.model:
                detections = self.detect_torch(image)
            elif hasattr(self, 'net') and self.net:
                detections = self.detect_opencv(image)
            else:
                detections = self.detect_dummy(image)
            
            # Filter for person detections
            person_detections = [det for det in detections if det['class_id'] == self.person_class_id]
            
            # Publish results
            self.publish_detections(person_detections, timestamp)
            self.publish_annotated_image(image, person_detections, timestamp)
            self.publish_markers(person_detections, timestamp)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            self.detection_count += len(person_detections)
            
            # Publish statistics periodically
            if self.frame_count % 30 == 0:  # Every 30 frames
                self.publish_statistics()
                
        except Exception as e:
            rospy.logerr(f"Detection processing error: {e}")
    
    def detect_torch(self, image):
        """Perform detection using PyTorch model"""
        detections = []
        
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Process predictions
        for pred in predictions:
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score > self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    detections.append({
                        'class_id': int(label - 1),  # Convert to 0-indexed
                        'confidence': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return detections
    
    def detect_opencv(self, image):
        """Perform detection using OpenCV DNN"""
        detections = []
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Scale bounding box back to image dimensions
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'class_id': class_ids[i],
                    'confidence': confidences[i],
                    'bbox': [x, y, x + w, y + h]
                })
        
        return detections
    
    def detect_dummy(self, image):
        """Dummy detector for testing"""
        height, width = image.shape[:2]
        # Create a fake detection in the center of the image
        detections = [{
            'class_id': self.person_class_id,
            'confidence': 0.8,
            'bbox': [width//4, height//4, 3*width//4, 3*height//4]
        }]
        return detections
    
    def publish_detections(self, detections, timestamp):
        """Publish detection results as Float32MultiArray"""
        msg = Float32MultiArray()
        
        # Format: [num_detections, x1, y1, x2, y2, confidence, x1, y1, x2, y2, confidence, ...]
        data = [len(detections)]
        
        for det in detections:
            bbox = det['bbox']
            data.extend([bbox[0], bbox[1], bbox[2], bbox[3], det['confidence']])
        
        msg.data = data
        self.detections_pub.publish(msg)
    
    def publish_annotated_image(self, image, detections, timestamp):
        """Publish annotated image with bounding boxes"""
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Convert back to ROS message
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            annotated_msg.header.stamp = timestamp
            annotated_msg.header.frame_id = "camera_color_optical_frame"
            self.annotated_pub.publish(annotated_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish annotated image: {e}")
    
    def publish_markers(self, detections, timestamp):
        """Publish 3D markers for RViz visualization"""
        marker_array = MarkerArray()
        
        for i, det in enumerate(detections):
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = timestamp
            marker.ns = "person_detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position (approximate depth of 3 meters)
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            marker.pose.position.x = 3.0  # 3 meters in front
            marker.pose.position.y = -(center_x - 320) / 100.0  # Rough conversion
            marker.pose.position.z = -(center_y - 240) / 100.0
            
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 1.8  # Approximate person height
            
            # Color (green with alpha based on confidence)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = det['confidence']
            
            marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(marker)
        
        self.markers_pub.publish(marker_array)
    
    def publish_statistics(self):
        """Publish detection statistics"""
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        avg_processing_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0
        detection_rate = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        stats = {
            'frames_processed': self.frame_count,
            'detections_found': self.detection_count,
            'average_fps': avg_fps,
            'average_processing_time': avg_processing_time,
            'detection_rate': detection_rate,
            'elapsed_time': elapsed_time
        }
        
        stats_msg = String()
        stats_msg.data = json.dumps(stats)
        self.stats_pub.publish(stats_msg)
        
        rospy.loginfo(f"Detection Stats - FPS: {avg_fps:.2f}, Processing: {avg_processing_time*1000:.1f}ms, "
                     f"Detection Rate: {detection_rate:.2f}")

if __name__ == '__main__':
    try:
        node = DetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass