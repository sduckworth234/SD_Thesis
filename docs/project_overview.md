# Project Overview: Low-Light Vision for Search and Rescue UAV Systems

## 🎯 Project Vision
Develop an autonomous UAV system equipped with computer vision capabilities for search and rescue operations in low-light, GPS-denied environments. The system will integrate advanced image enhancement, person detection, and visual SLAM technologies optimized for edge deployment.

## 🚁 System Components

### Hardware Stack
- **Primary Platform**: Intel RealSense D435i RGB-D camera
- **Development Environment**: Ubuntu 20.04, AMD CPU, NVIDIA RTX 4070
- **Target Deployment**: NVIDIA Jetson Xavier NX (Ubuntu 18.04, aarch64)
- **UAV Platform**: To be integrated (supporting Intel RealSense D435i)

### Software Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    ROS Noetic Framework                      │
├─────────────────────────────────────────────────────────────┤
│  Camera Node     │  Enhancement     │  Detection      │  SLAM │
│  (RealSense)     │  (ZERO-DCE++/SCI)│  (YOLO v4)     │(ORB-2)│
├─────────────────────────────────────────────────────────────┤
│               Mission Planning & Control                     │
├─────────────────────────────────────────────────────────────┤
│          Gazebo Simulation & RViz Visualization             │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Research Focus Areas

### 1. Low-Light Image Enhancement
**Objective**: Improve visibility in dark environments for better computer vision performance

**Models Under Investigation**:
- ZERO-DCE++ (Zero-Reference Deep Curve Estimation)
- SCI (Self-Calibrating Illumination)
- RetinexNet
- EnlightenGAN

**Key Metrics**:
- PSNR, SSIM, LPIPS for image quality
- Processing time for real-time constraints
- Memory usage for edge deployment

### 2. Person Detection in Low-Light
**Objective**: Reliable human detection in enhanced low-light imagery

**Detection Framework**: YOLO v4
**Enhancement Pipeline**: Raw Image → Enhancement → Detection → Tracking

**Evaluation Metrics**:
- mAP (mean Average Precision)
- Precision, Recall, F1-Score
- Detection confidence analysis
- Real-time performance (FPS)

### 3. Visual SLAM in Challenging Conditions
**Objective**: Robust localization and mapping without GPS

**SLAM System**: ORB-SLAM2
**Input**: Enhanced RGB-D streams from RealSense D435i

**Performance Metrics**:
- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- Map quality and density
- Loop closure detection rate

## 📊 Experimental Methodology

### Phase 1: Individual Component Benchmarking
1. **Enhancement Models**: Quality vs. speed trade-offs
2. **Detection Performance**: Enhanced vs. raw image comparison
3. **SLAM Robustness**: Tracking success in various lighting conditions

### Phase 2: Integrated System Testing
1. **End-to-End Pipeline**: Complete processing chain evaluation
2. **Real-Time Performance**: Latency and throughput analysis
3. **Resource Utilization**: CPU, GPU, memory profiling

### Phase 3: Mission Validation
1. **Simulation Testing**: Gazebo-based disaster scenarios
2. **Hardware Validation**: Controlled environment testing
3. **Performance Benchmarking**: Quantitative mission success metrics

## 🎛️ Development Workflow

### Repository Structure
```
SD_Thesis/
├── src/                    # Source code modules
│   ├── camera/            # RealSense interface
│   ├── enhancement/       # Low-light enhancement models
│   ├── detection/         # YOLO v4 person detection
│   ├── slam/             # ORB-SLAM2 integration
│   └── ros_nodes/        # ROS node implementations
├── config/               # Configuration files
├── launch/              # ROS launch files
├── tests/               # Unit and integration tests
├── scripts/             # Setup and deployment scripts
├── docs/                # Documentation and methodology
├── data/                # Datasets and results
└── logs/                # Experiment logs
```

### Development Environment
- **Version Control**: Git with detailed commit history
- **Documentation**: Markdown with methodology and setup guides
- **Testing**: Automated unit and integration tests
- **Deployment**: Containerized setup for reproducibility

## 🎯 Key Research Questions

1. **Primary**: How do different low-light enhancement algorithms affect person detection and visual SLAM performance in dark environments?

2. **Secondary**: What is the optimal balance between computational efficiency and accuracy for real-time deployment on edge devices?

3. **Tertiary**: How does enhanced imagery improve overall mission success rates in GPS-denied, low-light search and rescue scenarios?

## 📈 Expected Outcomes

### Technical Contributions
- Comprehensive benchmark of low-light enhancement models for UAV applications
- Optimized computer vision pipeline for edge deployment
- Validated system performance in realistic search and rescue scenarios

### Performance Targets
- **Real-time Processing**: >10 FPS on Jetson Xavier NX
- **Detection Accuracy**: >80% mAP in low-light conditions
- **SLAM Robustness**: <30cm ATE in controlled environments
- **Mission Success**: >80% success rate in simulated scenarios

## 🚀 Deployment Strategy

### Development to Production Pipeline
1. **Algorithm Development**: Ubuntu 20.04 + RTX 4070
2. **Performance Optimization**: Model compression and optimization
3. **Edge Deployment**: Jetson Xavier NX adaptation
4. **Field Testing**: Real-world validation
5. **Mission Deployment**: UAV integration

### Cross-Platform Compatibility
- **Architecture Support**: x86_64 → aarch64 migration
- **Dependency Management**: Containerized environments
- **Performance Scaling**: Adaptive model selection

## 📚 Documentation Strategy

### Academic Documentation
- **Methodology**: Detailed experimental design
- **Results**: Quantitative analysis and benchmarks
- **Discussion**: Performance trade-offs and limitations
- **Future Work**: Extension opportunities

### Technical Documentation
- **Setup Guides**: Environment configuration
- **API Reference**: Code documentation
- **Deployment Instructions**: Jetson Xavier NX setup
- **Troubleshooting**: Common issues and solutions

## 🔄 Continuous Integration

### Development Practices
- **Code Quality**: Automated linting and testing
- **Performance Monitoring**: Benchmark tracking
- **Experiment Logging**: Detailed parameter and result logging
- **Reproducibility**: Environment and data versioning

This project represents a comprehensive approach to developing practical computer vision solutions for emergency response applications, bridging the gap between academic research and real-world deployment challenges.
