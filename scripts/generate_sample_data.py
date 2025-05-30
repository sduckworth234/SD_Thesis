#!/usr/bin/env python3
"""
Sample Data Generation Script for SD_Thesis Project
Creates synthetic low-light images and test datasets for benchmarking.
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
import argparse

class SampleDataGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories for sample data."""
        dirs = [
            'low_light_samples',
            'normal_light_samples', 
            'person_detection_samples',
            'slam_test_sequences',
            'benchmark_sets'
        ]
        
        for dir_name in dirs:
            dir_path = os.path.join(self.output_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def generate_low_light_images(self, count=50):
        """Generate synthetic low-light images from normal images."""
        print(f"Generating {count} low-light sample images...")
        
        # Create synthetic base images
        base_images = self._create_base_images()
        
        for i in range(count):
            # Select random base image
            base_img = base_images[i % len(base_images)]
            
            # Apply low-light simulation
            low_light_img = self._simulate_low_light(base_img, 
                                                   brightness_factor=np.random.uniform(0.1, 0.4),
                                                   noise_level=np.random.uniform(0.01, 0.05))
            
            # Save images
            normal_path = os.path.join(self.output_dir, 'normal_light_samples', f'normal_{i:03d}.jpg')
            low_light_path = os.path.join(self.output_dir, 'low_light_samples', f'low_light_{i:03d}.jpg')
            
            cv2.imwrite(normal_path, base_img)
            cv2.imwrite(low_light_path, low_light_img)
            
            # Create metadata
            metadata = {
                'image_id': i,
                'normal_path': normal_path,
                'low_light_path': low_light_path,
                'brightness_factor': float(np.mean(low_light_img) / np.mean(base_img)),
                'noise_estimate': self._estimate_noise(low_light_img),
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.output_dir, 'low_light_samples', f'metadata_{i:03d}.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Generated {count} low-light image pairs")
    
    def generate_person_detection_samples(self, count=30):
        """Generate images with simulated person silhouettes for detection testing."""
        print(f"Generating {count} person detection samples...")
        
        for i in range(count):
            # Create background
            img = self._create_scene_background()
            
            # Add person silhouettes
            num_persons = np.random.randint(1, 4)
            bboxes = []
            
            for j in range(num_persons):
                bbox = self._add_person_silhouette(img)
                if bbox is not None:
                    bboxes.append(bbox)
            
            # Apply low-light conditions
            if np.random.random() > 0.5:
                img = self._simulate_low_light(img, 
                                             brightness_factor=np.random.uniform(0.2, 0.6),
                                             noise_level=np.random.uniform(0.01, 0.03))
                sample_type = 'low_light'
            else:
                sample_type = 'normal'
            
            # Save image and annotations
            img_path = os.path.join(self.output_dir, 'person_detection_samples', 
                                  f'person_sample_{sample_type}_{i:03d}.jpg')
            cv2.imwrite(img_path, img)
            
            # YOLO format annotations
            annotation = {
                'image_path': img_path,
                'image_size': [img.shape[1], img.shape[0]],  # width, height
                'bboxes': bboxes,
                'num_persons': len(bboxes),
                'lighting_condition': sample_type,
                'created_at': datetime.now().isoformat()
            }
            
            annotation_path = os.path.join(self.output_dir, 'person_detection_samples',
                                         f'annotations_{sample_type}_{i:03d}.json')
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)
        
        print(f"Generated {count} person detection samples")
    
    def generate_slam_test_sequence(self, sequence_length=100):
        """Generate a sequence of images for SLAM testing."""
        print(f"Generating SLAM test sequence with {sequence_length} frames...")
        
        slam_dir = os.path.join(self.output_dir, 'slam_test_sequences', 'indoor_sequence_01')
        os.makedirs(slam_dir, exist_ok=True)
        
        # Camera trajectory parameters
        center = np.array([320, 240])
        radius = 100
        height_variation = 20
        
        trajectory_data = []
        
        for i in range(sequence_length):
            # Generate camera pose
            angle = 2 * np.pi * i / sequence_length
            camera_pos = np.array([
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                height_variation * np.sin(2 * angle)
            ])
            
            # Generate synthetic indoor scene
            img = self._generate_indoor_scene(camera_pos, angle)
            
            # Apply lighting variations
            if i % 10 < 3:  # 30% low light
                img = self._simulate_low_light(img, 
                                             brightness_factor=np.random.uniform(0.3, 0.7),
                                             noise_level=np.random.uniform(0.01, 0.02))
            
            # Save frame
            frame_path = os.path.join(slam_dir, f'frame_{i:06d}.jpg')
            cv2.imwrite(frame_path, img)
            
            # Store pose ground truth
            pose_data = {
                'frame_id': i,
                'timestamp': i * 0.033,  # 30 FPS
                'camera_position': camera_pos.tolist(),
                'camera_orientation': [0, 0, angle],  # Roll, pitch, yaw
                'frame_path': frame_path
            }
            trajectory_data.append(pose_data)
        
        # Save trajectory ground truth
        trajectory_path = os.path.join(slam_dir, 'groundtruth_trajectory.json')
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"Generated SLAM sequence in {slam_dir}")
    
    def create_benchmark_sets(self):
        """Create organized benchmark sets for different evaluation scenarios."""
        print("Creating benchmark test sets...")
        
        benchmark_dir = os.path.join(self.output_dir, 'benchmark_sets')
        
        # Enhancement benchmark
        enhancement_dir = os.path.join(benchmark_dir, 'enhancement_evaluation')
        os.makedirs(enhancement_dir, exist_ok=True)
        
        # Detection benchmark  
        detection_dir = os.path.join(benchmark_dir, 'detection_evaluation')
        os.makedirs(detection_dir, exist_ok=True)
        
        # SLAM benchmark
        slam_dir = os.path.join(benchmark_dir, 'slam_evaluation')
        os.makedirs(slam_dir, exist_ok=True)
        
        # Create test configuration
        config = {
            'enhancement_tests': {
                'low_light_samples': os.path.join(self.output_dir, 'low_light_samples'),
                'expected_improvements': ['brightness', 'contrast', 'noise_reduction'],
                'metrics': ['PSNR', 'SSIM', 'brightness_improvement']
            },
            'detection_tests': {
                'person_samples': os.path.join(self.output_dir, 'person_detection_samples'),
                'expected_detections': 'load_from_annotations',
                'metrics': ['mAP', 'precision', 'recall', 'inference_time']
            },
            'slam_tests': {
                'test_sequences': os.path.join(self.output_dir, 'slam_test_sequences'),
                'ground_truth': 'groundtruth_trajectory.json',
                'metrics': ['trajectory_error', 'map_accuracy', 'loop_closure_accuracy']
            },
            'created_at': datetime.now().isoformat()
        }
        
        config_path = os.path.join(benchmark_dir, 'benchmark_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created benchmark configuration in {config_path}")
    
    def _create_base_images(self):
        """Create synthetic base images for processing."""
        images = []
        
        # Urban scene
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img1, (50, 100), (200, 400), (100, 100, 100), -1)  # Building
        cv2.rectangle(img1, (250, 150), (400, 450), (120, 120, 120), -1)  # Building
        cv2.rectangle(img1, (450, 200), (600, 400), (90, 90, 90), -1)   # Building
        cv2.circle(img1, (320, 100), 30, (200, 200, 0), -1)            # Sun/light
        images.append(img1)
        
        # Indoor scene
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2[:] = (50, 50, 40)  # Dark indoor background
        cv2.rectangle(img2, (100, 200), (300, 400), (80, 60, 40), -1)   # Furniture
        cv2.rectangle(img2, (400, 150), (550, 350), (70, 50, 30), -1)   # Furniture
        cv2.circle(img2, (200, 100), 15, (255, 255, 200), -1)          # Light source
        images.append(img2)
        
        # Outdoor scene
        img3 = np.zeros((480, 640, 3), dtype=np.uint8)
        img3[:240] = (135, 206, 235)  # Sky
        img3[240:] = (34, 139, 34)    # Ground
        cv2.ellipse(img3, (320, 480), (200, 100), 0, 180, 360, (139, 69, 19), -1)  # Hill
        images.append(img3)
        
        return images
    
    def _simulate_low_light(self, img, brightness_factor=0.3, noise_level=0.02):
        """Simulate low-light conditions on an image."""
        # Reduce brightness
        low_light = (img.astype(np.float32) * brightness_factor).astype(np.uint8)
        
        # Add noise
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.float32)
        low_light = np.clip(low_light.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return low_light
    
    def _estimate_noise(self, img):
        """Estimate noise level in an image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.std(cv2.Laplacian(gray, cv2.CV_64F))
    
    def _create_scene_background(self):
        """Create a scene background for person detection samples."""
        img = np.random.randint(20, 80, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure
        cv2.rectangle(img, (0, 350), (640, 480), (40, 30, 20), -1)      # Floor
        cv2.rectangle(img, (100, 100), (300, 350), (60, 50, 40), -1)    # Wall
        cv2.rectangle(img, (400, 150), (600, 350), (50, 60, 40), -1)    # Wall
        
        return img
    
    def _add_person_silhouette(self, img):
        """Add a person silhouette to the image and return bounding box."""
        h, w = img.shape[:2]
        
        # Random person size and position
        person_width = np.random.randint(30, 80)
        person_height = np.random.randint(80, 150)
        
        x = np.random.randint(0, w - person_width)
        y = np.random.randint(h//2, h - person_height)
        
        # Simple person silhouette (rectangle with head circle)
        body_color = (np.random.randint(30, 120), np.random.randint(30, 120), np.random.randint(30, 120))
        
        # Body
        cv2.rectangle(img, (x, y + person_height//4), 
                     (x + person_width, y + person_height), body_color, -1)
        
        # Head
        head_radius = person_width // 3
        cv2.circle(img, (x + person_width//2, y + head_radius), 
                  head_radius, body_color, -1)
        
        # Return bounding box in YOLO format (normalized)
        bbox = {
            'class': 'person',
            'x_center': (x + person_width/2) / w,
            'y_center': (y + person_height/2) / h,
            'width': person_width / w,
            'height': person_height / h,
            'confidence': 1.0
        }
        
        return bbox
    
    def _generate_indoor_scene(self, camera_pos, angle):
        """Generate an indoor scene for SLAM testing."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (40, 40, 35)  # Dark indoor background
        
        # Add walls and features based on camera position
        # Wall features
        wall_x = int(320 + 100 * np.cos(angle + np.pi/2))
        cv2.line(img, (wall_x, 0), (wall_x, 480), (80, 70, 60), 5)
        
        # Corner features
        corner_x = int(320 + 150 * np.cos(angle))
        corner_y = int(240 + 50 * np.sin(angle))
        cv2.circle(img, (corner_x, corner_y), 5, (120, 100, 80), -1)
        
        # Floor pattern
        for i in range(0, 640, 40):
            cv2.line(img, (i, 400), (i, 480), (60, 50, 40), 1)
        
        return img

def main():
    parser = argparse.ArgumentParser(description='Generate sample data for SD_Thesis project')
    parser.add_argument('--output-dir', default='../data/datasets', 
                       help='Output directory for generated samples')
    parser.add_argument('--low-light-count', type=int, default=50,
                       help='Number of low-light sample pairs to generate')
    parser.add_argument('--detection-count', type=int, default=30,
                       help='Number of person detection samples to generate')
    parser.add_argument('--slam-length', type=int, default=100,
                       help='Length of SLAM test sequence')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sample data in: {output_dir}")
    
    # Initialize generator
    generator = SampleDataGenerator(output_dir)
    
    # Generate all sample types
    generator.generate_low_light_images(args.low_light_count)
    generator.generate_person_detection_samples(args.detection_count)
    generator.generate_slam_test_sequence(args.slam_length)
    generator.create_benchmark_sets()
    
    print("\nSample data generation complete!")
    print(f"Data saved to: {output_dir}")
    print("\nGenerated datasets:")
    print("- Low-light enhancement samples")
    print("- Person detection samples") 
    print("- SLAM test sequences")
    print("- Benchmark test sets")

if __name__ == "__main__":
    main()
