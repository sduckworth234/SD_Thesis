# Benchmarking Plan for SD_Thesis

## Introduction
This document outlines the benchmarking plan for the thesis project, which aims to develop a drone-based system utilizing the Intel RealSense D435i camera for search and rescue operations in low-light environments. The benchmarks will evaluate the performance of various components, including detection algorithms, image enhancement models, and SLAM functionalities.

## Objectives
- To assess the effectiveness of YOLO v4 for person detection in low-light conditions.
- To evaluate the performance of low-light image enhancement models, specifically ZERO-DCE++ and SCI.
- To measure the accuracy and efficiency of the ORB-SLAM2 system when integrated with the camera and detection modules.

## Methodology
1. **Dataset Preparation**
   - Collect a diverse set of images and videos in varying low-light conditions.
   - Annotate the dataset for person detection tasks.

2. **Benchmarking Metrics**
   - **Detection Accuracy**: Measure precision, recall, and F1-score for YOLO v4.
   - **Processing Time**: Record the time taken for detection and enhancement processes.
   - **Image Quality**: Use metrics such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) to evaluate image enhancement results.
   - **SLAM Performance**: Assess the mapping accuracy and processing speed of the ORB-SLAM2 system.

3. **Experimental Setup**
   - Configure the Intel RealSense D435i camera for optimal performance.
   - Set up the necessary software environment, including ROS and relevant dependencies.
   - Implement a controlled testing environment to minimize external variables.

4. **Testing Procedure**
   - Run detection and enhancement algorithms on the prepared dataset.
   - Collect and log results for each test case.
   - Repeat tests under different lighting conditions to ensure robustness.

5. **Data Analysis**
   - Analyze the collected data to identify trends and performance bottlenecks.
   - Compare results against established benchmarks in the literature.

## Conclusion
The benchmarking plan aims to provide a comprehensive evaluation of the developed system's capabilities in low-light search and rescue scenarios. The results will contribute to the overall thesis and help refine the methodologies used in the project.