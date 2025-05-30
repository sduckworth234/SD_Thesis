# Methodology for Thesis Work

## Introduction
The objective of this thesis is to develop a comprehensive system that utilizes the Intel RealSense D435i camera for low-light image enhancement and person detection, ultimately aimed at search and rescue operations in disaster-stricken environments.

## System Overview
The proposed system integrates several key components:
1. **Camera Module**: Captures images and video streams using the RealSense D435i.
2. **Detection Module**: Implements YOLO v4 for real-time person detection.
3. **Enhancement Module**: Utilizes ZERO-DCE++ and SCI models for enhancing low-light images.
4. **SLAM Module**: Employs ORB-SLAM2 for mapping and localization.
5. **ROS Integration**: Facilitates communication between different modules and manages the overall system.

## Methodology Steps

### 1. Literature Review
Conduct a thorough review of existing literature on low-light image enhancement, object detection, and SLAM techniques to identify gaps and opportunities for improvement.

### 2. System Design
Design the architecture of the system, detailing the interactions between the camera, detection, enhancement, and SLAM modules. Create flowcharts and diagrams to visualize the data flow.

### 3. Implementation
- **Camera Setup**: Implement the camera capture functionality and calibrate the camera to ensure accurate image processing.
- **Detection Implementation**: Integrate YOLO v4 for detecting persons in real-time from the camera feed.
- **Image Enhancement**: Implement the ZERO-DCE++ and SCI models to enhance images captured in low-light conditions.
- **SLAM Integration**: Wrap the ORB-SLAM2 functionality to work seamlessly with the camera and detection outputs.

### 4. Testing and Validation
- **Unit Testing**: Develop unit tests for each module to ensure individual components function correctly.
- **Integration Testing**: Test the entire pipeline to validate the interactions between modules and overall system performance.
- **Benchmarking**: Establish a set of benchmarks to evaluate the effectiveness of the detection and enhancement models in low-light conditions.

### 5. Deployment
Prepare the system for deployment on the Jetson Xavier NX, ensuring compatibility with the aarch64 architecture. Optimize the code for performance on the target hardware.

### 6. Documentation
Maintain comprehensive documentation throughout the project, including setup instructions, methodology, and results. This will facilitate reproducibility and provide a reference for future work.

## Conclusion
This methodology outlines a structured approach to developing a robust system for low-light image enhancement and person detection using advanced technologies. The ultimate goal is to create a reliable solution for search and rescue operations in challenging environments.