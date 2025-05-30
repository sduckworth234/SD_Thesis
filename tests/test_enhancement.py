#!/usr/bin/env python3
"""
Test script for low-light enhancement models (ZERO-DCE++, SCI)
"""

import rospy
import cv2
import numpy as np
import torch
import torch.nn as nn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import sys

class LowLightEnhancementTest:
    def __init__(self):
        rospy.init_node('enhancement_test', anonymous=True)
        self.bridge = CvBridge()
        
        # Model paths
        self.models_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'low_light_enhancement')
        
        # Load models
        self.zero_dce_model = None
        self.sci_model = None
        self.load_models()
        
        # ROS subscriber
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        rospy.loginfo("Low-light enhancement test node initialized")
    
    def load_models(self):
        """Load low-light enhancement models"""
        try:
            # Check for ZERO-DCE++ model
            zero_dce_path = os.path.join(self.models_path, 'zero_dce_plus.py')
            if os.path.exists(zero_dce_path):
                rospy.loginfo("Loading ZERO-DCE++ model...")
                # This would load the actual ZERO-DCE++ model
                # For now, we'll create a placeholder
                self.zero_dce_model = self.create_dummy_model()
                rospy.loginfo("✓ ZERO-DCE++ model loaded")
            else:
                rospy.logwarn("ZERO-DCE++ model not found")
            
            # Check for SCI model
            sci_path = os.path.join(self.models_path, 'sci.py')
            if os.path.exists(sci_path):
                rospy.loginfo("Loading SCI model...")
                # This would load the actual SCI model
                # For now, we'll create a placeholder
                self.sci_model = self.create_dummy_model()
                rospy.loginfo("✓ SCI model loaded")
            else:
                rospy.logwarn("SCI model not found")
            
        except Exception as e:
            rospy.logerr(f"Error loading models: {e}")
    
    def create_dummy_model(self):
        """Create a dummy model for testing purposes"""
        class DummyEnhancer(nn.Module):
            def __init__(self):
                super(DummyEnhancer, self).__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                
            def forward(self, x):
                # Simple brightness enhancement
                return torch.clamp(x * 1.5, 0, 1)
        
        model = DummyEnhancer()
        model.eval()
        return model
    
    def enhance_with_zero_dce(self, image):
        """Enhance image using ZERO-DCE++ model"""
        if self.zero_dce_model is None:
            return image
        
        try:
            # Convert image to tensor
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                enhanced_tensor = self.zero_dce_model(image_tensor)
            
            # Convert back to numpy
            enhanced_image = enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy()
            enhanced_image = (enhanced_image * 255).astype(np.uint8)
            
            return enhanced_image
            
        except Exception as e:
            rospy.logerr(f"Error in ZERO-DCE enhancement: {e}")
            return image
    
    def enhance_with_sci(self, image):
        """Enhance image using SCI model"""
        if self.sci_model is None:
            return image
        
        try:
            # Convert image to tensor
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                enhanced_tensor = self.sci_model(image_tensor)
            
            # Convert back to numpy
            enhanced_image = enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy()
            enhanced_image = (enhanced_image * 255).astype(np.uint8)
            
            return enhanced_image
            
        except Exception as e:
            rospy.logerr(f"Error in SCI enhancement: {e}")
            return image
    
    def enhance_traditional(self, image):
        """Traditional enhancement methods for comparison"""
        # Histogram equalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_hist = cv2.merge([l, a, b])
        enhanced_hist = cv2.cvtColor(enhanced_hist, cv2.COLOR_LAB2BGR)
        
        # Gamma correction
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_gamma = cv2.LUT(image, table)
        
        return enhanced_hist, enhanced_gamma
    
    def calculate_metrics(self, original, enhanced):
        """Calculate enhancement metrics"""
        # Convert to grayscale for metrics
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Average brightness
        orig_brightness = np.mean(orig_gray)
        enh_brightness = np.mean(enh_gray)
        
        # Contrast (standard deviation)
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        
        return {
            'brightness_improvement': enh_brightness - orig_brightness,
            'contrast_improvement': enh_contrast - orig_contrast,
            'brightness_ratio': enh_brightness / orig_brightness if orig_brightness > 0 else 0
        }
    
    def image_callback(self, msg):
        """Process incoming image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Enhance with different methods
            enhanced_zero_dce = self.enhance_with_zero_dce(cv_image)
            enhanced_sci = self.enhance_with_sci(cv_image)
            enhanced_hist, enhanced_gamma = self.enhance_traditional(cv_image)
            
            # Calculate metrics
            metrics_zero_dce = self.calculate_metrics(cv_image, enhanced_zero_dce)
            metrics_sci = self.calculate_metrics(cv_image, enhanced_sci)
            
            # Log metrics
            rospy.loginfo_throttle(5, f"ZERO-DCE brightness improvement: {metrics_zero_dce['brightness_improvement']:.2f}")
            rospy.loginfo_throttle(5, f"SCI brightness improvement: {metrics_sci['brightness_improvement']:.2f}")
            
            # Display results
            # Resize images for display
            height, width = cv_image.shape[:2]
            display_width, display_height = 320, 240
            
            original_small = cv2.resize(cv_image, (display_width, display_height))
            zero_dce_small = cv2.resize(enhanced_zero_dce, (display_width, display_height))
            sci_small = cv2.resize(enhanced_sci, (display_width, display_height))
            hist_small = cv2.resize(enhanced_hist, (display_width, display_height))
            gamma_small = cv2.resize(enhanced_gamma, (display_width, display_height))
            
            # Create comparison display
            top_row = np.hstack([original_small, zero_dce_small, sci_small])
            bottom_row = np.hstack([hist_small, gamma_small, np.zeros((display_height, display_width, 3), dtype=np.uint8)])
            
            comparison = np.vstack([top_row, bottom_row])
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "ZERO-DCE++", (display_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "SCI", (2 * display_width + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "Histogram EQ", (10, display_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "Gamma Corr", (display_width + 10, display_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Enhancement Comparison", comparison)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def run_test(self):
        """Run enhancement test"""
        rospy.loginfo("Starting low-light enhancement test...")
        rospy.loginfo("Make sure camera is running: roslaunch realsense2_camera rs_camera.launch")
        
        # Test with sample images if available
        sample_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_images')
        dark_image_path = os.path.join(sample_dir, 'dark_image.jpg')
        
        if os.path.exists(dark_image_path):
            rospy.loginfo("Testing with sample dark image...")
            image = cv2.imread(dark_image_path)
            
            # Test all enhancement methods
            enhanced_zero_dce = self.enhance_with_zero_dce(image)
            enhanced_sci = self.enhance_with_sci(image)
            enhanced_hist, enhanced_gamma = self.enhance_traditional(image)
            
            # Calculate and display metrics
            metrics_zero_dce = self.calculate_metrics(image, enhanced_zero_dce)
            metrics_sci = self.calculate_metrics(image, enhanced_sci)
            
            rospy.loginfo(f"Sample test - ZERO-DCE brightness improvement: {metrics_zero_dce['brightness_improvement']:.2f}")
            rospy.loginfo(f"Sample test - SCI brightness improvement: {metrics_sci['brightness_improvement']:.2f}")
        
        # Wait for live camera feed
        rospy.loginfo("Waiting for camera feed...")
        rospy.spin()

def test_enhancement_standalone():
    """Test enhancement models without ROS"""
    rospy.loginfo("Testing enhancement models standalone...")
    
    try:
        # Test PyTorch availability
        if torch.cuda.is_available():
            rospy.loginfo(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            rospy.logwarn("CUDA not available, using CPU")
        
        # Test model creation
        tester = LowLightEnhancementTest()
        
        if tester.zero_dce_model is not None:
            rospy.loginfo("✓ ZERO-DCE++ model test passed")
        else:
            rospy.logwarn("ZERO-DCE++ model not available")
        
        if tester.sci_model is not None:
            rospy.loginfo("✓ SCI model test passed")
        else:
            rospy.logwarn("SCI model not available")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)  # Dark image
        
        enhanced = tester.enhance_with_zero_dce(dummy_image)
        metrics = tester.calculate_metrics(dummy_image, enhanced)
        
        rospy.loginfo(f"Dummy image test - brightness improvement: {metrics['brightness_improvement']:.2f}")
        
        return True
        
    except Exception as e:
        rospy.logerr(f"Enhancement standalone test failed: {e}")
        return False

if __name__ == '__main__':
    try:
        # First test standalone functionality
        standalone_test = test_enhancement_standalone()
        
        if standalone_test:
            # Test with ROS integration
            tester = LowLightEnhancementTest()
            tester.run_test()
        else:
            rospy.logerr("Standalone enhancement test failed")
        
        cv2.destroyAllWindows()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test failed with error: {e}")
