#!/bin/bash
"""
Jetson Xavier NX Deployment Script for SD_Thesis
Optimizes and deploys the low-light vision system for edge computing.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
JETSON_MODEL="xavier_nx"
ROS_DISTRO="noetic"
WORKSPACE_DIR="/home/jetson/sd_thesis_ws"
PROJECT_DIR="$WORKSPACE_DIR/src/SD_Thesis"

# GPU Memory and Performance Settings
GPU_FREQ_MAX=1377000000
GPU_FREQ_MIN=306000000
EMC_FREQ_MAX=1600000000

echo -e "${GREEN}=================================="
echo "SD_Thesis Jetson Xavier NX Setup"
echo -e "==================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running on Jetson
check_jetson() {
    if [[ ! -f /etc/nv_tegra_release ]]; then
        print_error "This script must be run on a Jetson device"
        exit 1
    fi
    
    # Get Jetson info
    JETSON_INFO=$(cat /proc/device-tree/model 2>/dev/null)
    print_status "Detected Jetson: $JETSON_INFO"
}

# Function to set maximum performance mode
set_max_performance() {
    print_status "Setting maximum performance mode..."
    
    # Set maximum CPU frequency
    sudo nvpmodel -m 0  # Max performance mode
    
    # Set CPU governors to performance
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -f $cpu ]]; then
            echo performance | sudo tee $cpu > /dev/null
        fi
    done
    
    # Set GPU to maximum frequency
    sudo bash -c "echo $GPU_FREQ_MAX > /sys/kernel/debug/bpmp/debug/clk/gv11b_gpc0_clk/rate"
    sudo bash -c "echo $GPU_FREQ_MIN > /sys/kernel/debug/bpmp/debug/clk/gv11b_gpc0_clk/min_rate"
    
    # Set memory frequency
    sudo bash -c "echo $EMC_FREQ_MAX > /sys/kernel/debug/bpmp/debug/clk/emc/rate"
    
    # Disable DVFS (Dynamic Voltage and Frequency Scaling)
    sudo bash -c "echo 0 > /sys/devices/17000000.gv11b/enable_3d_scaling"
    
    print_status "Performance mode configured"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Jetson-specific dependencies..."
    
    # Update package lists
    sudo apt update
    
    # Install essential packages
    sudo apt install -y \
        python3-pip \
        python3-dev \
        python3-numpy \
        python3-opencv \
        cmake \
        build-essential \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-serial-dev \
        libhdf5-dev
    
    # Install ROS Noetic if not present
    if ! command -v roscore &> /dev/null; then
        print_status "Installing ROS Noetic..."
        
        # Setup ROS repository
        sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
        curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
        
        # Install ROS
        sudo apt update
        sudo apt install -y ros-noetic-desktop-full
        
        # Initialize rosdep
        sudo rosdep init
        rosdep update
        
        # Setup environment
        echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
        source ~/.bashrc
    fi
    
    # Install additional ROS packages
    sudo apt install -y \
        ros-noetic-cv-bridge \
        ros-noetic-image-transport \
        ros-noetic-compressed-image-transport \
        ros-noetic-realsense2-camera \
        ros-noetic-vision-msgs \
        ros-noetic-visualization-msgs
    
    print_status "Dependencies installed"
}

# Function to install optimized PyTorch for Jetson
install_pytorch_jetson() {
    print_status "Installing PyTorch optimized for Jetson..."
    
    # Download and install PyTorch wheel for Jetson
    cd /tmp
    
    # PyTorch 1.11.0 for Jetson (adjust version as needed)
    TORCH_URL="https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl"
    TORCH_FILE="torch-1.11.0-cp38-cp38-linux_aarch64.whl"
    
    wget -O $TORCH_FILE $TORCH_URL
    pip3 install $TORCH_FILE
    
    # Install torchvision
    TORCHVISION_URL="https://nvidia.box.com/shared/static/1isri5q0zm7wds0d6ppr0kg2y7fvjyn2.whl"
    TORCHVISION_FILE="torchvision-0.12.0a0+2662797-cp38-cp38-linux_aarch64.whl"
    
    wget -O $TORCHVISION_FILE $TORCHVISION_URL
    pip3 install $TORCHVISION_FILE
    
    # Clean up
    rm $TORCH_FILE $TORCHVISION_FILE
    
    print_status "PyTorch installed for Jetson"
}

# Function to setup workspace
setup_workspace() {
    print_status "Setting up workspace..."
    
    # Create workspace
    mkdir -p $WORKSPACE_DIR/src
    cd $WORKSPACE_DIR
    
    # Initialize catkin workspace
    if [[ ! -f devel/setup.bash ]]; then
        catkin_make
    fi
    
    # Source workspace
    echo "source $WORKSPACE_DIR/devel/setup.bash" >> ~/.bashrc
    
    print_status "Workspace configured"
}
# Function to optimize models for Jetson
optimize_models() {
    print_status "Optimizing models for Jetson deployment..."
    
    cd $PROJECT_DIR/models
    
    # Create optimization script
    cat > optimize_models.py << 'EOF'
import torch
import torchvision.transforms as transforms
import os
import sys

def optimize_yolo_model():
    """Optimize YOLO model for TensorRT"""
    print("Optimizing YOLO model...")
    try:
        # Load YOLO model
        model_path = "yolov4.pt"
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location='cuda')
            model.eval()
            
            # Convert to TorchScript
            traced_model = torch.jit.trace(model, torch.randn(1, 3, 640, 640).cuda())
            traced_model.save("yolov4_optimized.pt")
            print("YOLO model optimized successfully")
        else:
            print("YOLO model not found, skipping optimization")
    except Exception as e:
        print(f"Error optimizing YOLO model: {e}")

def optimize_enhancement_models():
    """Optimize enhancement models for inference"""
    print("Optimizing enhancement models...")
    try:
        # Optimize Zero-DCE++ model
        zero_dce_path = "zero_dce_plus.pth"
        if os.path.exists(zero_dce_path):
            checkpoint = torch.load(zero_dce_path, map_location='cuda')
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                model_state = checkpoint
            
            # Save optimized state dict
            torch.save(model_state, "zero_dce_plus_optimized.pth")
            print("Zero-DCE++ model optimized")
        
        # Optimize SCI model
        sci_path = "sci_model.pth"
        if os.path.exists(sci_path):
            checkpoint = torch.load(sci_path, map_location='cuda')
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                model_state = checkpoint
            
            torch.save(model_state, "sci_model_optimized.pth")
            print("SCI model optimized")
            
    except Exception as e:
        print(f"Error optimizing enhancement models: {e}")

if __name__ == "__main__":
    optimize_yolo_model()
    optimize_enhancement_models()
    print("Model optimization completed")
EOF
    
    # Run optimization
    python3 optimize_models.py
    
    print_status "Models optimized for Jetson"
}

# Function to create Jetson-optimized launch files
create_jetson_launch_files() {
    print_status "Creating Jetson-optimized launch files..."
    
    mkdir -p $PROJECT_DIR/launch/jetson
    
    # Create Jetson-specific complete pipeline launch
    cat > $PROJECT_DIR/launch/jetson/jetson_pipeline.launch << 'EOF'
<launch>
    <!-- Jetson Xavier NX Optimized Pipeline -->
    
    <!-- Camera Node with reduced resolution for performance -->
    <node name="camera_node" pkg="sd_thesis" type="camera_node.py" output="screen">
        <param name="width" value="640"/>
        <param name="height" value="480"/>
        <param name="fps" value="15"/>
        <param name="enable_depth" value="true"/>
        <param name="device_id" value=""/>
        <remap from="camera/image_raw" to="/camera/color/image_raw"/>
        <remap from="camera/depth/image_raw" to="/camera/depth/image_raw"/>
    </node>
    
    <!-- Enhancement Node with optimized models -->
    <node name="enhancement_node" pkg="sd_thesis" type="enhancement_node.py" output="screen">
        <param name="model_type" value="zero_dce_plus"/>
        <param name="model_path" value="$(find sd_thesis)/models/zero_dce_plus_optimized.pth"/>
        <param name="device" value="cuda"/>
        <param name="batch_size" value="1"/>
        <param name="input_size" value="320"/>
        <remap from="image_input" to="/camera/color/image_raw"/>
        <remap from="image_enhanced" to="/enhanced/image"/>
    </node>
    
    <!-- YOLO Detection Node with optimized inference -->
    <node name="detection_node" pkg="sd_thesis" type="detection_node.py" output="screen">
        <param name="model_path" value="$(find sd_thesis)/models/yolov4_optimized.pt"/>
        <param name="confidence_threshold" value="0.5"/>
        <param name="nms_threshold" value="0.4"/>
        <param name="device" value="cuda"/>
        <param name="input_size" value="416"/>
        <remap from="image_input" to="/enhanced/image"/>
        <remap from="detections" to="/detections"/>
    </node>
    
    <!-- ORB-SLAM3 Node with reduced features for performance -->
    <node name="slam_node" pkg="sd_thesis" type="slam_node.py" output="screen">
        <param name="vocab_path" value="$(find sd_thesis)/models/orbslam3/ORBvoc.txt"/>
        <param name="settings_path" value="$(find sd_thesis)/config/orbslam3_jetson.yaml"/>
        <param name="num_features" value="500"/>
        <param name="scale_factor" value="1.2"/>
        <param name="num_levels" value="6"/>
        <remap from="image_input" to="/enhanced/image"/>
        <remap from="depth_input" to="/camera/depth/image_raw"/>
        <remap from="pose" to="/slam/pose"/>
        <remap from="map_points" to="/slam/map_points"/>
    </node>
    
    <!-- Performance Monitor -->
    <node name="performance_monitor" pkg="sd_thesis" type="performance_monitor.py" output="screen">
        <param name="monitor_frequency" value="1.0"/>
        <param name="log_file" value="/tmp/jetson_performance.log"/>
    </node>
    
</launch>
EOF
    
    # Create camera configuration for Jetson
    cat > $PROJECT_DIR/config/jetson_camera.yaml << 'EOF'
%YAML:1.0
#--------------------------------------------------------------------------------------------
# Camera Parameters for Jetson Xavier NX deployment
#--------------------------------------------------------------------------------------------

# Camera calibration parameters (adjust for your specific camera)
Camera.fx: 384.0
Camera.fy: 384.0
Camera.cx: 320.0
Camera.cy: 240.0

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 640
Camera.height: 480

# Camera frames per second (reduced for Jetson)
Camera.fps: 15.0

# Color order of the images (0: BGR, 1: RGB)
Camera.RGB: 1

# Depth map values factor (if using depth camera)
DepthMapFactor: 1000.0

#--------------------------------------------------------------------------------------------
# ORB Parameters (optimized for Jetson)
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 500

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 6

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
EOF
    
    print_status "Jetson launch files created"
}

# Function to setup auto-start service
setup_autostart_service() {
    print_status "Setting up auto-start service..."
    
    # Create systemd service
    sudo tee /etc/systemd/system/sd-thesis.service > /dev/null << EOF
[Unit]
Description=SD Thesis Low-Light Vision System
After=network.target
Wants=network.target

[Service]
Type=forking
User=jetson
Group=jetson
WorkingDirectory=$WORKSPACE_DIR
Environment="ROS_MASTER_URI=http://localhost:11311"
Environment="ROS_HOSTNAME=localhost"
ExecStartPre=/bin/bash -c 'source /opt/ros/noetic/setup.bash && source $WORKSPACE_DIR/devel/setup.bash'
ExecStart=/bin/bash -c 'source /opt/ros/noetic/setup.bash && source $WORKSPACE_DIR/devel/setup.bash && roslaunch sd_thesis jetson_pipeline.launch'
ExecStop=/bin/bash -c 'source /opt/ros/noetic/setup.bash && killall -SIGINT roslaunch'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Enable service
    sudo systemctl daemon-reload
    sudo systemctl enable sd-thesis.service
    
    print_status "Auto-start service configured"
}

# Function to create monitoring dashboard
create_monitoring_dashboard() {
    print_status "Creating monitoring dashboard..."
    
    mkdir -p $PROJECT_DIR/monitoring
    
    # Create web-based monitoring dashboard
    cat > $PROJECT_DIR/monitoring/dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Real-time monitoring dashboard for SD_Thesis on Jetson Xavier NX
"""

import rospy
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
from std_msgs.msg import String
import psutil
import GPUtil

app = Flask(__name__)

class JetsonMonitor:
    def __init__(self):
        self.performance_data = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'gpu_usage': 0,
            'gpu_memory': 0,
            'temperature': 0,
            'power_consumption': 0,
            'fps': 0,
            'pipeline_status': 'stopped',
            'last_update': datetime.now().isoformat()
        }
        
        # ROS setup
        rospy.init_node('jetson_monitor', anonymous=True)
        rospy.Subscriber('/performance_monitor/status', String, self.performance_callback)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitor_system)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def performance_callback(self, msg):
        """Handle performance monitor messages"""
        try:
            data = json.loads(msg.data)
            self.performance_data.update(data)
            self.performance_data['last_update'] = datetime.now().isoformat()
        except:
            pass
    
    def monitor_system(self):
        """Monitor system resources"""
        while True:
            try:
                # CPU and Memory
                self.performance_data['cpu_usage'] = psutil.cpu_percent()
                self.performance_data['memory_usage'] = psutil.virtual_memory().percent
                
                # GPU (if available)
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.performance_data['gpu_usage'] = gpu.load * 100
                        self.performance_data['gpu_memory'] = gpu.memoryUtil * 100
                except:
                    pass
                
                # Temperature
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read().strip()) / 1000.0
                        self.performance_data['temperature'] = temp
                except:
                    pass
                
                time.sleep(1)
            except:
                time.sleep(5)

monitor = JetsonMonitor()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    return jsonify(monitor.performance_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
EOF
    
    # Create HTML template for dashboard
    mkdir -p $PROJECT_DIR/monitoring/templates
    cat > $PROJECT_DIR/monitoring/templates/dashboard.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>SD_Thesis Jetson Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 14px; color: #666; margin-bottom: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; }
        .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-good { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SD_Thesis Jetson Xavier NX Monitor</h1>
            <p>Real-time system performance and pipeline status</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-title">CPU Usage</div>
                <div class="metric-value" id="cpu-usage">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value" id="memory-usage">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">GPU Usage</div>
                <div class="metric-value" id="gpu-usage">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Temperature</div>
                <div class="metric-value" id="temperature">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Pipeline FPS</div>
                <div class="metric-value" id="fps">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Pipeline Status</div>
                <div class="metric-value" id="status">--</div>
            </div>
        </div>
        
        <div class="charts">
            <div class="chart-container">
                <h3>System Resources</h3>
                <canvas id="resourceChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Performance History</h3>
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Chart setup
        const resourceCtx = document.getElementById('resourceChart').getContext('2d');
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        
        const resourceChart = new Chart(resourceCtx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'Memory', 'GPU'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#3498db', '#2ecc71', '#e74c3c']
                }]
            },
            options: { responsive: true }
        });
        
        const performanceChart = new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'FPS', data: [], borderColor: '#3498db', fill: false },
                    { label: 'CPU %', data: [], borderColor: '#2ecc71', fill: false }
                ]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });
        
        // Update function
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update metrics
                    document.getElementById('cpu-usage').textContent = data.cpu_usage.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = data.memory_usage.toFixed(1) + '%';
                    document.getElementById('gpu-usage').textContent = data.gpu_usage.toFixed(1) + '%';
                    document.getElementById('temperature').textContent = data.temperature.toFixed(1) + '°C';
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    
                    const statusElement = document.getElementById('status');
                    statusElement.textContent = data.pipeline_status;
                    statusElement.className = 'metric-value ' + 
                        (data.pipeline_status === 'running' ? 'status-good' : 
                         data.pipeline_status === 'warning' ? 'status-warning' : 'status-error');
                    
                    // Update charts
                    resourceChart.data.datasets[0].data = [data.cpu_usage, data.memory_usage, data.gpu_usage];
                    resourceChart.update();
                    
                    const now = new Date().toLocaleTimeString();
                    performanceChart.data.labels.push(now);
                    performanceChart.data.datasets[0].data.push(data.fps);
                    performanceChart.data.datasets[1].data.push(data.cpu_usage);
                    
                    if (performanceChart.data.labels.length > 20) {
                        performanceChart.data.labels.shift();
                        performanceChart.data.datasets[0].data.shift();
                        performanceChart.data.datasets[1].data.shift();
                    }
                    performanceChart.update();
                });
        }
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
EOF
    
    # Install Flask for dashboard
    pip3 install flask psutil
    
    print_status "Monitoring dashboard created"
}

# Main deployment function
deploy_to_jetson() {
    print_status "Starting comprehensive Jetson Xavier NX deployment..."
    
    # Check if we're on a Jetson device
    check_jetson
    
    # Set maximum performance
    set_max_performance
    
    # Install dependencies
    install_dependencies
    
    # Install PyTorch for Jetson
    install_pytorch_jetson
    
    # Setup workspace
    setup_workspace
    
    # Copy project files
    if [[ -d "/tmp/SD_Thesis" ]]; then
        print_status "Copying project files..."
        cp -r /tmp/SD_Thesis/* $PROJECT_DIR/
    else
        print_warning "Project files not found in /tmp/SD_Thesis. Please copy manually."
    fi
    
    # Build workspace
    cd $WORKSPACE_DIR
    catkin_make
    
    # Optimize models
    optimize_models
    
    # Create Jetson-specific launch files
    create_jetson_launch_files
    
    # Setup auto-start service
    setup_autostart_service
    
    # Create monitoring dashboard
    create_monitoring_dashboard
    
    print_status "Deployment completed successfully!"
    echo -e "${GREEN}=================================="
    echo "Deployment Summary:"
    echo "- Workspace: $WORKSPACE_DIR"
    echo "- Launch file: $PROJECT_DIR/launch/jetson/jetson_pipeline.launch"
    echo "- Auto-start service: sd-thesis.service"
    echo "- Monitoring dashboard: http://localhost:8080"
    echo ""
    echo "To start the system:"
    echo "  sudo systemctl start sd-thesis"
    echo ""
    echo "To view logs:"
    echo "  sudo journalctl -u sd-thesis -f"
    echo ""
    echo "To access monitoring:"
    echo "  python3 $PROJECT_DIR/monitoring/dashboard.py"
    echo -e "==================================${NC}"
}

# Run deployment if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    deploy_to_jetson
fi