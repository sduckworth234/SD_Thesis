# Jetson Deployment Instructions

This document outlines the steps required to deploy the SD_Thesis project on the Jetson Xavier NX.

## Prerequisites

1. **Jetson Xavier NX Setup**: Ensure that your Jetson Xavier NX is set up with Ubuntu 18.04 and has internet access.
2. **CUDA and cuDNN**: Install the appropriate versions of CUDA and cuDNN compatible with your Jetson Xavier NX.
3. **ROS Noetic**: Install ROS Noetic on your Jetson device. Follow the official ROS installation guide for Jetson devices.

## Environment Setup

1. **Clone the Repository**: Clone the SD_Thesis repository to your Jetson Xavier NX.
   ```bash
   git clone <repository_url>
   cd SD_Thesis
   ```

2. **Run Setup Script**: Execute the setup script to configure the environment.
   ```bash
   chmod +x scripts/setup_environment.sh
   ./scripts/setup_environment.sh
   ```

3. **Install Dependencies**: Install the required dependencies using the provided script.
   ```bash
   chmod +x scripts/install_dependencies.sh
   ./scripts/install_dependencies.sh
   ```

## Deployment Steps

1. **Modify Configuration Files**: Ensure that the configuration files in the `config` directory are set up correctly for the Jetson Xavier NX environment. Pay special attention to camera parameters and model paths.

2. **Run Jetson Deployment Script**: Execute the deployment script to set up the project on the Jetson device.
   ```bash
   chmod +x scripts/jetson_deployment.sh
   ./scripts/jetson_deployment.sh
   ```

3. **Launch ROS Nodes**: Start the necessary ROS nodes for camera, detection, and SLAM functionalities.
   ```bash
   roslaunch launch/real_hardware.launch
   ```

## Testing

1. **Run Unit Tests**: Before deploying the full system, run unit tests to ensure all components are functioning correctly.
   ```bash
   python3 -m unittest discover -s tests/unit_tests
   ```

2. **Integration Tests**: After unit tests, run integration tests to validate the entire processing pipeline.
   ```bash
   python3 -m unittest discover -s tests/integration_tests
   ```

## Troubleshooting

- If you encounter compatibility issues, ensure that all dependencies are correctly installed and compatible with the Jetson Xavier NX architecture.
- Check the logs in the `logs/experiment_logs` directory for any errors or warnings during execution.

## Conclusion

Following these steps will help you successfully deploy the SD_Thesis project on the Jetson Xavier NX. Ensure to document any changes made during the deployment process for future reference.