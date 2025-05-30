# SD_Thesis API Documentation

This document provides comprehensive API documentation for the SD_Thesis low-light vision system components.

## Table of Contents
1. [Core Modules](#core-modules)
2. [ROS Nodes](#ros-nodes)
3. [Enhancement Models](#enhancement-models)
4. [Detection Systems](#detection-systems)
5. [SLAM Integration](#slam-integration)
6. [Utilities](#utilities)
7. [Configuration](#configuration)

## Core Modules

### CameraNode
Camera interface for Intel RealSense D435i.

```python
class CameraNode:
    """
    ROS node for camera capture and streaming.
    
    Publishes:
        /camera/color/image_raw (sensor_msgs/Image): Color camera stream
        /camera/depth/image_raw (sensor_msgs/Image): Depth camera stream
        /camera/camera_info (sensor_msgs/CameraInfo): Camera calibration
    """
    
    def __init__(self, width=640, height=480, fps=30, enable_depth=True):
        """
        Initialize camera node.
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels
            fps (int): Frames per second
            enable_depth (bool): Enable depth stream
        """
        
    def configure_camera(self, device_id=""):
        """
        Configure camera pipeline.
        
        Args:
            device_id (str): Specific device serial number (optional)
            
        Returns:
            bool: True if configuration successful
        """
        
    def start_streaming(self):
        """Start camera streaming."""
        
    def stop_streaming(self):
        """Stop camera streaming."""
        
    def get_frame(self):
        """
        Get current frame.
        
        Returns:
            tuple: (color_image, depth_image, timestamp)
        """
```

### EnhancementNode
Low-light image enhancement using Zero-DCE++ or SCI models.

```python
class EnhancementNode:
    """
    ROS node for low-light image enhancement.
    
    Subscribes:
        /camera/color/image_raw (sensor_msgs/Image): Input images
        
    Publishes:
        /enhanced/image (sensor_msgs/Image): Enhanced images
        /enhancement/metadata (std_msgs/String): Enhancement metadata
    """
    
    def __init__(self, model_type="zero_dce_plus", model_path="", device="cuda"):
        """
        Initialize enhancement node.
        
        Args:
            model_type (str): "zero_dce_plus" or "sci"
            model_path (str): Path to trained model weights
            device (str): "cuda" or "cpu"
        """
        
    def load_model(self, model_path):
        """
        Load enhancement model.
        
        Args:
            model_path (str): Path to model file
            
        Returns:
            bool: True if loading successful
        """
        
    def enhance_image(self, image):
        """
        Enhance input image.
        
        Args:
            image (np.ndarray): Input image [H, W, 3]
            
        Returns:
            np.ndarray: Enhanced image [H, W, 3]
        """
        
    def set_enhancement_parameters(self, brightness=1.0, contrast=1.0):
        """
        Set enhancement parameters.
        
        Args:
            brightness (float): Brightness adjustment factor
            contrast (float): Contrast adjustment factor
        """
```

### DetectionNode
Person detection using YOLO v4.

```python
class DetectionNode:
    """
    ROS node for person detection.
    
    Subscribes:
        /enhanced/image (sensor_msgs/Image): Enhanced images
        
    Publishes:
        /detections (vision_msgs/Detection2DArray): Detected persons
        /detection/debug_image (sensor_msgs/Image): Debug visualization
    """
    
    def __init__(self, model_path="", confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize detection node.
        
        Args:
            model_path (str): Path to YOLO weights
            confidence_threshold (float): Minimum detection confidence
            nms_threshold (float): Non-maximum suppression threshold
        """
        
    def load_yolo_model(self, model_path, config_path="", weights_path=""):
        """
        Load YOLO model.
        
        Args:
            model_path (str): Path to model file
            config_path (str): Path to config file
            weights_path (str): Path to weights file
            
        Returns:
            bool: True if loading successful
        """
        
    def detect_persons(self, image):
        """
        Detect persons in image.
        
        Args:
            image (np.ndarray): Input image [H, W, 3]
            
        Returns:
            list: List of detection dictionaries with keys:
                - bbox: [x, y, w, h]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        
    def set_detection_parameters(self, conf_thresh=0.5, nms_thresh=0.4, input_size=416):
        """
        Set detection parameters.
        
        Args:
            conf_thresh (float): Confidence threshold
            nms_thresh (float): NMS threshold
            input_size (int): Input image size for model
        """
```

### SLAMNode
Visual SLAM using ORB-SLAM2.

```python
class SLAMNode:
    """
    ROS node for visual SLAM.
    
    Subscribes:
        /enhanced/image (sensor_msgs/Image): Enhanced images
        /camera/depth/image_raw (sensor_msgs/Image): Depth images
        
    Publishes:
        /slam/pose (geometry_msgs/PoseStamped): Current camera pose
        /slam/map_points (sensor_msgs/PointCloud2): Map points
        /slam/keyframes (visualization_msgs/MarkerArray): Keyframe poses
    """
    
    def __init__(self, vocab_path="", config_path="", slam_mode="RGBD"):
        """
        Initialize SLAM node.
        
        Args:
            vocab_path (str): Path to ORB vocabulary
            config_path (str): Path to camera configuration
            slam_mode (str): "RGBD", "Stereo", or "Monocular"
        """
        
    def initialize_slam(self, vocab_path, config_path):
        """
        Initialize SLAM system.
        
        Args:
            vocab_path (str): Path to vocabulary file
            config_path (str): Path to configuration file
            
        Returns:
            bool: True if initialization successful
        """
        
    def track_frame(self, image, depth_image=None, timestamp=None):
        """
        Track camera pose for current frame.
        
        Args:
            image (np.ndarray): Current RGB image
            depth_image (np.ndarray): Current depth image (for RGBD mode)
            timestamp (float): Frame timestamp
            
        Returns:
            dict: Tracking result with keys:
                - pose: 4x4 transformation matrix
                - status: tracking status
                - num_features: number of tracked features
        """
        
    def get_map_points(self):
        """
        Get current map points.
        
        Returns:
            np.ndarray: Map points [N, 3]
        """
        
    def save_trajectory(self, filename):
        """
        Save camera trajectory to file.
        
        Args:
            filename (str): Output file path
        """
```

## Enhancement Models

### Zero-DCE++ Model

```python
class ZeroDCEPlusModel(nn.Module):
    """
    Zero-DCE++ enhancement model.
    
    A lightweight deep learning model for low-light image enhancement
    that learns image-specific curves for pixel-wise enhancement.
    """
    
    def __init__(self, num_iterations=8):
        """
        Initialize Zero-DCE++ model.
        
        Args:
            num_iterations (int): Number of enhancement iterations
        """
        
    def forward(self, input_image):
        """
        Forward pass through the model.
        
        Args:
            input_image (torch.Tensor): Input image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Enhanced image [B, 3, H, W]
        """
        
    def enhance_batch(self, images):
        """
        Enhance a batch of images.
        
        Args:
            images (torch.Tensor): Batch of images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Enhanced images [B, 3, H, W]
        """

class ZeroDCEPlusLoss(nn.Module):
    """Loss functions for Zero-DCE++ training."""
    
    def __init__(self, spa_weight=1.0, exp_weight=10.0, col_weight=5.0, ill_weight=1.0):
        """
        Initialize loss function.
        
        Args:
            spa_weight (float): Spatial consistency loss weight
            exp_weight (float): Exposure loss weight
            col_weight (float): Color constancy loss weight
            ill_weight (float): Illumination smoothness loss weight
        """
        
    def forward(self, enhanced_image, original_image, curve_params):
        """
        Compute total loss.
        
        Args:
            enhanced_image (torch.Tensor): Enhanced output
            original_image (torch.Tensor): Original input
            curve_params (torch.Tensor): Learned curve parameters
            
        Returns:
            dict: Loss components
        """
```

### SCI Model

```python
class SCIModel(nn.Module):
    """
    Self-Calibrated Illumination (SCI) enhancement model.
    
    Uses self-calibrated modules with attention mechanisms
    for adaptive low-light enhancement.
    """
    
    def __init__(self, in_channels=3, out_channels=3, num_blocks=4):
        """
        Initialize SCI model.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            num_blocks (int): Number of residual blocks
        """
        
    def forward(self, input_image):
        """
        Forward pass through SCI model.
        
        Args:
            input_image (torch.Tensor): Input image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Enhanced image [B, 3, H, W]
        """

class AttentionBlock(nn.Module):
    """Attention mechanism for SCI model."""
    
    def __init__(self, channels, reduction=16):
        """
        Initialize attention block.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Channel reduction factor
        """
```

## Utilities

### PerformanceMonitor

```python
class PerformanceMonitor:
    """
    System performance monitoring utility.
    
    Monitors CPU, GPU, memory usage and pipeline performance.
    """
    
    def __init__(self, log_file="/tmp/performance.log", monitor_frequency=1.0):
        """
        Initialize performance monitor.
        
        Args:
            log_file (str): Path to log file
            monitor_frequency (float): Monitoring frequency in Hz
        """
        
    def start_monitoring(self):
        """Start performance monitoring."""
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        
    def get_system_stats(self):
        """
        Get current system statistics.
        
        Returns:
            dict: System statistics including:
                - cpu_usage: CPU usage percentage
                - memory_usage: Memory usage percentage
                - gpu_usage: GPU usage percentage
                - temperature: System temperature
        """
        
    def get_pipeline_stats(self):
        """
        Get pipeline performance statistics.
        
        Returns:
            dict: Pipeline statistics including:
                - fps: Frames per second
                - latency: Processing latency
                - dropped_frames: Number of dropped frames
        """
        
    def log_performance(self, stats):
        """
        Log performance statistics.
        
        Args:
            stats (dict): Performance statistics
        """
```

### BenchmarkSystem

```python
class BenchmarkSystem:
    """
    Comprehensive benchmarking system for performance evaluation.
    """
    
    def __init__(self, data_path="", output_path="results/"):
        """
        Initialize benchmark system.
        
        Args:
            data_path (str): Path to benchmark datasets
            output_path (str): Output directory for results
        """
        
    def run_enhancement_benchmark(self, model_path, test_images):
        """
        Benchmark enhancement model performance.
        
        Args:
            model_path (str): Path to enhancement model
            test_images (list): List of test image paths
            
        Returns:
            dict: Benchmark results including:
                - psnr: Peak Signal-to-Noise Ratio
                - ssim: Structural Similarity Index
                - processing_time: Average processing time
        """
        
    def run_detection_benchmark(self, model_path, test_images, annotations):
        """
        Benchmark detection model performance.
        
        Args:
            model_path (str): Path to detection model
            test_images (list): List of test image paths
            annotations (list): Ground truth annotations
            
        Returns:
            dict: Benchmark results including:
                - map: Mean Average Precision
                - precision: Precision scores
                - recall: Recall scores
                - fps: Frames per second
        """
        
    def run_slam_benchmark(self, config_path, sequence_path):
        """
        Benchmark SLAM performance.
        
        Args:
            config_path (str): SLAM configuration path
            sequence_path (str): Test sequence path
            
        Returns:
            dict: SLAM benchmark results including:
                - ate: Absolute Trajectory Error
                - rpe: Relative Pose Error
                - tracking_success: Tracking success rate
        """
        
    def generate_report(self, results, output_file="benchmark_report.html"):
        """
        Generate comprehensive benchmark report.
        
        Args:
            results (dict): Benchmark results
            output_file (str): Output report file path
        """
```

## Configuration

### Launch File Parameters

#### complete_pipeline.launch
```xml
<!-- Camera Configuration -->
<param name="camera/width" value="640" />
<param name="camera/height" value="480" />
<param name="camera/fps" value="30" />
<param name="camera/enable_depth" value="true" />

<!-- Enhancement Configuration -->
<param name="enhancement/model_type" value="zero_dce_plus" />
<param name="enhancement/model_path" value="$(find sd_thesis)/models/zero_dce_plus.pth" />
<param name="enhancement/device" value="cuda" />

<!-- Detection Configuration -->
<param name="detection/model_path" value="$(find sd_thesis)/models/yolov4.weights" />
<param name="detection/config_path" value="$(find sd_thesis)/config/yolov4.cfg" />
<param name="detection/confidence_threshold" value="0.5" />
<param name="detection/nms_threshold" value="0.4" />

<!-- SLAM Configuration -->
<param name="slam/vocab_path" value="$(find sd_thesis)/config/ORBvoc.txt" />
<param name="slam/config_path" value="$(find sd_thesis)/config/camera.yaml" />
<param name="slam/mode" value="RGBD" />
```

#### jetson_pipeline.launch (Optimized for Jetson)
```xml
<!-- Reduced resolution for performance -->
<param name="camera/width" value="320" />
<param name="camera/height" value="240" />
<param name="camera/fps" value="15" />

<!-- Optimized models -->
<param name="enhancement/model_path" value="$(find sd_thesis)/models/zero_dce_plus_optimized.pth" />
<param name="detection/model_path" value="$(find sd_thesis)/models/yolov4_optimized.pt" />

<!-- Reduced feature count for SLAM -->
<param name="slam/num_features" value="500" />
```

### Model Configuration Files

#### camera.yaml
```yaml
# Camera calibration parameters
Camera.fx: 615.0
Camera.fy: 615.0
Camera.cx: 320.0
Camera.cy: 240.0

# Distortion parameters
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

# Camera resolution
Camera.width: 640
Camera.height: 480
Camera.fps: 30.0

# ORB extractor parameters
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
```

#### training_config.yaml
```yaml
# Training configuration for enhancement models
training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 100
  
  # Zero-DCE+ specific
  zero_dce:
    num_iterations: 8
    loss_weights:
      spatial: 1.0
      exposure: 10.0
      color: 5.0
      illumination: 1.0
  
  # SCI specific
  sci:
    num_blocks: 4
    attention: true
    loss_weights:
      reconstruction: 1.0
      perceptual: 0.1
      adversarial: 0.01
```

## Error Codes

| Code | Component | Description | Solution |
|------|-----------|-------------|----------|
| E001 | Camera | Device not found | Check USB connection |
| E002 | Enhancement | Model loading failed | Verify model file path |
| E003 | Detection | CUDA OOM | Reduce batch size |
| E004 | SLAM | Tracking lost | Improve lighting/movement |
| E005 | System | Performance degraded | Check system resources |

## Usage Examples

### Basic Pipeline Usage
```python
import rospy
from sd_thesis import CameraNode, EnhancementNode, DetectionNode

# Initialize nodes
rospy.init_node('sd_thesis_pipeline')

camera = CameraNode(width=640, height=480, fps=30)
enhancer = EnhancementNode(model_type="zero_dce_plus")
detector = DetectionNode(confidence_threshold=0.5)

# Start pipeline
camera.start_streaming()
enhancer.start()
detector.start()

rospy.spin()
```

### Training Enhancement Model
```python
from sd_thesis.training import train_zero_dce

# Train Zero-DCE+ model
train_zero_dce(
    train_data_path="data/train/",
    val_data_path="data/val/",
    model_save_path="models/zero_dce_plus.pth",
    num_epochs=100,
    batch_size=16,
    learning_rate=0.0001
)
```

### Running Benchmarks
```python
from sd_thesis.benchmark import BenchmarkSystem

benchmark = BenchmarkSystem(
    data_path="data/benchmark/",
    output_path="results/"
)

# Run complete benchmark
results = benchmark.run_complete_benchmark()
benchmark.generate_report(results)
```

This API documentation provides comprehensive coverage of all system components, their interfaces, and usage patterns for the SD_Thesis low-light vision system.
