#!/usr/bin/env python3
"""
Enhancement ROS Node
ROS node for real-time low-light image enhancement using ZERO-DCE++ and SCI models.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError
import threading
import time
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from enhancement.zero_dce.enhance import ZeroDCEEnhancer
    from enhancement.sci.enhance import SCIEnhancer
except ImportError as e:
    rospy.logwarn(f"Could not import enhancement modules: {e}")
    # Fallback imports
    ZeroDCEEnhancer = None
    SCIEnhancer = None


class EnhancementNode:
    """
    ROS node for low-light image enhancement.
    
    Subscribes to camera images and publishes enhanced versions using
    ZERO-DCE++ and SCI models.
    """
    
    def __init__(self):
        """Initialize enhancement node."""
        rospy.init_node('enhancement_node', anonymous=True)
        
        # Parameters
        self.model_type = rospy.get_param('~model_type', 'zero_dce')  # 'zero_dce' or 'sci' or 'both'
        self.device = rospy.get_param('~device', 'auto')
        self.input_topic = rospy.get_param('~input_topic', '/camera/color/image_raw')
        self.queue_size = rospy.get_param('~queue_size', 1)
        self.publish_rate = rospy.get_param('~publish_rate', 30.0)
        
        # Model paths
        self.zero_dce_model_path = rospy.get_param('~zero_dce_model_path', None)
        self.sci_model_path = rospy.get_param('~sci_model_path', None)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize enhancement models
        self.zero_dce_enhancer = None
        self.sci_enhancer = None
        self._initialize_models()
        
        # Publishers
        self.publishers = {}
        if self.model_type in ['zero_dce', 'both'] and self.zero_dce_enhancer:
            self.publishers['zero_dce'] = rospy.Publisher(
                '/enhancement/zero_dce', Image, queue_size=self.queue_size)
        
        if self.model_type in ['sci', 'both'] and self.sci_enhancer:
            self.publishers['sci'] = rospy.Publisher(
                '/enhancement/sci', Image, queue_size=self.queue_size)
        
        # Performance monitoring publishers
        self.perf_pub = rospy.Publisher('/enhancement/performance', String, queue_size=1)
        self.fps_pub = rospy.Publisher('/enhancement/fps', Float32, queue_size=1)
        
        # Subscriber
        self.image_sub = rospy.Subscriber(
            self.input_topic, Image, self.image_callback, queue_size=self.queue_size)
        
        # Threading for performance
        self.processing_lock = threading.Lock()
        self.latest_image = None
        self.processing = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = []
        
        # Rate limiting
        self.rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo(f"Enhancement node initialized with model type: {self.model_type}")
        rospy.loginfo(f"Subscribed to: {self.input_topic}")
        rospy.loginfo(f"Publishing to: {list(self.publishers.keys())}")
    
    def _initialize_models(self):
        """Initialize enhancement models based on configuration."""
        try:
            if self.model_type in ['zero_dce', 'both']:
                if ZeroDCEEnhancer is not None:
                    rospy.loginfo("Initializing ZERO-DCE++ model...")
                    self.zero_dce_enhancer = ZeroDCEEnhancer(
                        model_path=self.zero_dce_model_path,
                        device=self.device
                    )
                    rospy.loginfo("ZERO-DCE++ model loaded successfully")
                else:
                    rospy.logwarn("ZERO-DCE++ enhancer not available")
            
            if self.model_type in ['sci', 'both']:
                if SCIEnhancer is not None:
                    rospy.loginfo("Initializing SCI model...")
                    self.sci_enhancer = SCIEnhancer(
                        model_path=self.sci_model_path,
                        device=self.device
                    )
                    rospy.loginfo("SCI model loaded successfully")
                else:
                    rospy.logwarn("SCI enhancer not available")
                    
        except Exception as e:
            rospy.logerr(f"Error initializing enhancement models: {e}")
            # Create fallback enhancers
            self._create_fallback_enhancers()
    
    def _create_fallback_enhancers(self):
        """Create simple fallback enhancers for testing."""
        class FallbackEnhancer:
            def enhance_image(self, image):
                # Simple contrast and brightness enhancement
                enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
                # Apply gamma correction
                gamma = 0.7
                lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, lookup_table)
                return enhanced
        
        if self.model_type in ['zero_dce', 'both'] and self.zero_dce_enhancer is None:
            rospy.logwarn("Using fallback ZERO-DCE++ enhancer")
            self.zero_dce_enhancer = FallbackEnhancer()
        
        if self.model_type in ['sci', 'both'] and self.sci_enhancer is None:
            rospy.logwarn("Using fallback SCI enhancer")
            self.sci_enhancer = FallbackEnhancer()
    
    def image_callback(self, msg):
        """
        Callback for incoming camera images.
        
        Args:
            msg: ROS Image message
        """
        with self.processing_lock:
            if not self.processing:
                self.latest_image = msg
                self.processing = True
                # Process in separate thread to avoid blocking
                threading.Thread(target=self._process_image, daemon=True).start()
    
    def _process_image(self):
        """Process the latest image with enhancement models."""
        try:
            start_time = time.time()
            
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            
            # Enhance with selected models
            enhanced_images = {}
            
            if self.model_type in ['zero_dce', 'both'] and self.zero_dce_enhancer:
                enhanced_images['zero_dce'] = self.zero_dce_enhancer.enhance_image(cv_image)
            
            if self.model_type in ['sci', 'both'] and self.sci_enhancer:
                enhanced_images['sci'] = self.sci_enhancer.enhance_image(cv_image)
            
            # Publish enhanced images
            for model_name, enhanced_img in enhanced_images.items():
                if model_name in self.publishers:
                    try:
                        enhanced_msg = self.bridge.cv2_to_imgmsg(enhanced_img, "bgr8")
                        enhanced_msg.header = self.latest_image.header
                        self.publishers[model_name].publish(enhanced_msg)
                    except CvBridgeError as e:
                        rospy.logwarn(f"CV Bridge error for {model_name}: {e}")
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep only last 100 measurements for rolling average
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.frame_count += 1
            
            # Publish performance metrics every 30 frames
            if self.frame_count % 30 == 0:
                self._publish_performance_metrics()
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
        
        finally:
            with self.processing_lock:
                self.processing = False
    
    def _publish_performance_metrics(self):
        """Publish performance metrics."""
        try:
            # Calculate metrics
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            if elapsed_time > 0:
                fps = self.frame_count / elapsed_time
                avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                
                # Publish FPS
                fps_msg = Float32()
                fps_msg.data = fps
                self.fps_pub.publish(fps_msg)
                
                # Publish detailed performance info
                perf_info = {
                    'fps': fps,
                    'avg_processing_time_ms': avg_processing_time * 1000,
                    'frames_processed': self.frame_count,
                    'model_type': self.model_type,
                    'device': self.device
                }
                
                perf_msg = String()
                perf_msg.data = str(perf_info)
                self.perf_pub.publish(perf_msg)
                
                rospy.loginfo(f"Enhancement FPS: {fps:.1f}, Avg processing time: {avg_processing_time*1000:.1f}ms")
                
        except Exception as e:
            rospy.logwarn(f"Error publishing performance metrics: {e}")
    
    def run(self):
        """Main run loop."""
        rospy.loginfo("Enhancement node started. Waiting for images...")
        
        try:
            while not rospy.is_shutdown():
                self.rate.sleep()
                
        except KeyboardInterrupt:
            rospy.loginfo("Enhancement node shutting down...")
        
        finally:
            # Final performance report
            if self.frame_count > 0:
                total_time = time.time() - self.start_time
                final_fps = self.frame_count / total_time if total_time > 0 else 0
                rospy.loginfo(f"Final performance: {final_fps:.1f} FPS, {self.frame_count} frames processed")


class EnhancementComparison:
    """
    Utility class for comparing different enhancement methods.
    """
    
    def __init__(self):
        """Initialize comparison node."""
        rospy.init_node('enhancement_comparison', anonymous=True)
        
        self.bridge = CvBridge()
        
        # Store images for comparison
        self.original_image = None
        self.zero_dce_image = None
        self.sci_image = None
        
        # Subscribers
        self.original_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.original_callback)
        self.zero_dce_sub = rospy.Subscriber('/enhancement/zero_dce', Image, self.zero_dce_callback)
        self.sci_sub = rospy.Subscriber('/enhancement/sci', Image, self.sci_callback)
        
        # Publisher for comparison image
        self.comparison_pub = rospy.Publisher('/enhancement/comparison', Image, queue_size=1)
        
        rospy.loginfo("Enhancement comparison node initialized")
    
    def original_callback(self, msg):
        """Store original image."""
        try:
            self.original_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._create_comparison()
        except CvBridgeError as e:
            rospy.logwarn(f"Error converting original image: {e}")
    
    def zero_dce_callback(self, msg):
        """Store ZERO-DCE enhanced image."""
        try:
            self.zero_dce_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._create_comparison()
        except CvBridgeError as e:
            rospy.logwarn(f"Error converting ZERO-DCE image: {e}")
    
    def sci_callback(self, msg):
        """Store SCI enhanced image."""
        try:
            self.sci_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._create_comparison()
        except CvBridgeError as e:
            rospy.logwarn(f"Error converting SCI image: {e}")
    
    def _create_comparison(self):
        """Create and publish comparison image."""
        if self.original_image is None:
            return
        
        images = [self.original_image]
        labels = ["Original"]
        
        if self.zero_dce_image is not None:
            images.append(self.zero_dce_image)
            labels.append("ZERO-DCE++")
        
        if self.sci_image is not None:
            images.append(self.sci_image)
            labels.append("SCI")
        
        if len(images) > 1:
            # Resize all images to same size
            target_height = min(img.shape[0] for img in images)
            resized_images = []
            
            for img in images:
                if img.shape[0] != target_height:
                    aspect_ratio = img.shape[1] / img.shape[0]
                    target_width = int(target_height * aspect_ratio)
                    resized = cv2.resize(img, (target_width, target_height))
                else:
                    resized = img
                resized_images.append(resized)
            
            # Create horizontal concatenation
            comparison = np.hstack(resized_images)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            color = (255, 255, 255)
            
            x_offset = 0
            for i, (img, label) in enumerate(zip(resized_images, labels)):
                # Calculate text position
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = x_offset + (img.shape[1] - text_size[0]) // 2
                text_y = 30
                
                # Add text with background
                cv2.rectangle(comparison, (text_x - 5, text_y - 25), 
                            (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(comparison, label, (text_x, text_y), font, font_scale, color, thickness)
                
                x_offset += img.shape[1]
            
            # Publish comparison image
            try:
                comparison_msg = self.bridge.cv2_to_imgmsg(comparison, "bgr8")
                self.comparison_pub.publish(comparison_msg)
            except CvBridgeError as e:
                rospy.logwarn(f"Error publishing comparison image: {e}")


def main():
    """Main function for enhancement node."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'comparison':
        # Run comparison node
        try:
            comparison_node = EnhancementComparison()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
    else:
        # Run enhancement node
        try:
            enhancement_node = EnhancementNode()
            enhancement_node.run()
        except rospy.ROSInterruptException:
            pass


if __name__ == '__main__':
    main()
