# SD_Thesis Testing Guide

## 📋 **Testing Sequence**

This document outlines the systematic testing approach for the SD_Thesis pipeline.

### 🎯 **Testing Order**
1. **SLAM Module** (ORB-SLAM3) ✅ 
2. **Camera Module** (RealSense D435i)
3. **Detection Module** (YOLO v4)
4. **Enhancement Module** (SCI/Zero-DCE)
5. **Integrated Pipeline**

---

## 🔧 **Pre-Testing Setup**

### Environment Setup
```bash
cd /home/duck/Desktop/SD_Thesis
source ~/.bashrc
export ORB_SLAM3_ROOT_PATH=/opt/ORB_SLAM3
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/opt/ORB_SLAM3/Examples/ROS
```

### Verify Installation
```bash
python3 scripts/verify_orbslam3.py
python3 scripts/test_orbslam3_integration.py
```

---

## 1️⃣ **SLAM Module Testing**

### Test 1: ORB-SLAM3 Standalone
```bash
# Test SLAM node independently
roscore &
sleep 2
rosrun sd_thesis slam_node.py
```

### Test 2: SLAM with Dummy Data
```bash
roslaunch sd_thesis test_slam.launch
```

### Test 3: SLAM Visualization
```bash
# In separate terminals:
roscore &
rviz -d config/slam_visualization.rviz &
rosrun sd_thesis slam_node.py
```

---

## 2️⃣ **Camera Module Testing**

### Test 1: Camera Hardware Detection
```bash
# Check if RealSense is detected
lsusb | grep -i intel
rs-enumerate-devices
```

### Test 2: Camera Node
```bash
roslaunch sd_thesis test_camera.launch
```

### Test 3: Camera Visualization
```bash
roslaunch sd_thesis test_camera.launch rviz:=true
```

---

## 3️⃣ **Detection Module Testing**

### Test 1: YOLO Detection
```bash
python3 src/detection/yolo_v4/person_detector.py
```

### Test 2: Detection Node
```bash
rosrun sd_thesis detection_node.py
```

---

## 4️⃣ **Enhancement Module Testing**

### Test 1: SCI Enhancement
```bash
python3 src/enhancement/sci/enhance.py
```

### Test 2: Zero-DCE Enhancement
```bash
python3 src/enhancement/zero_dce/enhance.py
```

---

## 5️⃣ **Integrated Pipeline Testing**

### Test 1: Complete Pipeline
```bash
roslaunch sd_thesis complete_pipeline.launch
```

### Test 2: Real Hardware Test
```bash
roslaunch sd_thesis real_hardware.launch
```

---

## 📊 **Expected Outputs**

### SLAM Module
- ✅ ORB-SLAM3 process starts
- ✅ Pose estimation topics published
- ✅ Map visualization in RViz
- ✅ No errors in logs

### Camera Module
- ✅ Color and depth streams
- ✅ Camera info published
- ✅ Images visible in RViz/image_view
- ✅ Proper frame rates (30 FPS)

### Detection Module
- ✅ Person bounding boxes
- ✅ Detection confidence scores
- ✅ Real-time performance

### Enhancement Module
- ✅ Improved image quality
- ✅ Low-light enhancement
- ✅ Noise reduction

---

## 🔍 **Troubleshooting**

### Common Issues
1. **ORB-SLAM3 not starting**: Check vocabulary and config paths
2. **Camera not detected**: Verify USB connection and permissions
3. **ROS node failures**: Check Python paths and dependencies
4. **Performance issues**: Monitor CPU/GPU usage

### Debug Commands
```bash
# Check ROS topics
rostopic list

# Monitor topic data
rostopic echo /camera/color/image_raw

# Check node status
rosnode list
rosnode info /slam_node

# View logs
rosrun rqt_console rqt_console
```

---

## 📝 **Test Results Log**

### Date: 2025-06-01

#### SLAM Module Tests
- [ ] ORB-SLAM3 Standalone
- [ ] SLAM with Dummy Data  
- [ ] SLAM Visualization

#### Camera Module Tests
- [ ] Hardware Detection
- [ ] Camera Node
- [ ] Camera Visualization

#### Detection Module Tests
- [ ] YOLO Detection
- [ ] Detection Node

#### Enhancement Module Tests
- [ ] SCI Enhancement
- [ ] Zero-DCE Enhancement

#### Integration Tests
- [ ] Complete Pipeline
- [ ] Real Hardware Test

---

## 🎯 **Success Criteria**

### SLAM
- ✅ ORB-SLAM3 initializes successfully
- ✅ Pose estimation accuracy within acceptable range
- ✅ Map building functionality working
- ✅ Real-time performance (>20 FPS processing)

### Camera
- ✅ Stable 30 FPS color stream
- ✅ Synchronized depth data
- ✅ Correct camera calibration
- ✅ Low latency (<100ms)

### Detection
- ✅ Person detection accuracy >80%
- ✅ Real-time performance (>15 FPS)
- ✅ Reliable bounding box tracking

### Enhancement
- ✅ Improved image quality metrics
- ✅ Real-time processing capability
- ✅ Consistent enhancement across frames

---

*Last Updated: 2025-06-01*
*ORB-SLAM3 Installation Completed Successfully*
