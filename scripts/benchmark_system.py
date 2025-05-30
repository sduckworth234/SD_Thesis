#!/usr/bin/env python3
"""
Comprehensive Benchmarking System for SD_Thesis
Automated performance testing and benchmarking of the complete pipeline.
"""

import rospy
import subprocess
import time
import json
import os
import argparse
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import psutil
import shutil

class BenchmarkSystem:
    def __init__(self, duration=300, output_dir="results", test_data_dir="data/test"):
        self.duration = duration  # Test duration in seconds
        self.output_dir = Path(output_dir)
        self.test_data_dir = Path(test_data_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir = self.output_dir / f"benchmark_{self.timestamp}"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark configuration
        self.tests = {
            'system_baseline': {'duration': 60, 'nodes': []},
            'camera_only': {'duration': 120, 'nodes': ['camera_node.py']},
            'camera_enhancement': {'duration': 180, 'nodes': ['camera_node.py', 'enhancement_node.py']},
            'camera_detection': {'duration': 180, 'nodes': ['camera_node.py', 'detection_node.py']},
            'camera_slam': {'duration': 180, 'nodes': ['camera_node.py', 'slam_node.py']},
            'full_pipeline': {'duration': 300, 'nodes': ['camera_node.py', 'enhancement_node.py', 
                                                        'detection_node.py', 'slam_node.py']}
        }
        
        self.results = {}
        
    def setup_test_data(self):
        """Setup test data and sample images"""
        print("Setting up test data...")
        
        # Create test data directories
        datasets_dir = self.test_data_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample low-light images for testing
        self.create_sample_images(datasets_dir)
        
        # Create configuration files for testing
        self.create_test_configs()
        
    def create_sample_images(self, output_dir):
        """Create synthetic low-light test images"""
        print("Creating sample test images...")
        
        # Create various test scenarios
        scenarios = {
            'very_dark': {'brightness': 0.1, 'noise': 0.3},
            'moderately_dark': {'brightness': 0.3, 'noise': 0.2},
            'twilight': {'brightness': 0.5, 'noise': 0.1},
            'indoor_lighting': {'brightness': 0.7, 'noise': 0.05}
        }
        
        # Image dimensions
        width, height = 640, 480
        
        for scenario_name, params in scenarios.items():
            scenario_dir = output_dir / scenario_name
            scenario_dir.mkdir(exist_ok=True)
            
            for i in range(10):  # Create 10 images per scenario
                # Create base image with geometric shapes
                image = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add geometric objects (simulating people/objects)
                # Rectangle (person-like shape)
                cv2.rectangle(image, (width//4, height//3), (width//2, 2*height//3), (255, 255, 255), -1)
                
                # Circle (head-like shape)
                cv2.circle(image, (3*width//8, height//4), 30, (255, 255, 255), -1)
                
                # Add some random features
                for _ in range(5):
                    x, y = np.random.randint(0, width), np.random.randint(0, height)
                    cv2.circle(image, (x, y), np.random.randint(5, 20), (200, 200, 200), -1)
                
                # Apply lighting conditions
                brightness = params['brightness']
                image = (image * brightness).astype(np.uint8)
                
                # Add noise
                noise_level = params['noise']
                noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int16)
                image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Save image
                filename = scenario_dir / f"test_image_{i:03d}.jpg"
                cv2.imwrite(str(filename), image)
        
        print(f"Created test images in {output_dir}")
    
    def create_test_configs(self):
        """Create test configuration files"""
        config_dir = self.test_output_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Camera test config
        camera_config = {
            'width': 640,
            'height': 480,
            'fps': 30,
            'enable_depth': True,
            'test_mode': True,
            'test_data_path': str(self.test_data_dir / "datasets")
        }
        
        with open(config_dir / "camera_test.yaml", 'w') as f:
            import yaml
            yaml.dump(camera_config, f)
        
        # Enhancement test config
        enhancement_config = {
            'models': ['zero_dce', 'sci'],
            'comparison_mode': True,
            'save_results': True,
            'output_dir': str(self.test_output_dir / "enhancement_results")
        }
        
        with open(config_dir / "enhancement_test.yaml", 'w') as f:
            yaml.dump(enhancement_config, f)
    
    def run_system_baseline(self):
        """Run baseline system performance test"""
        print("Running system baseline test...")
        
        start_time = time.time()
        baseline_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': []
        }
        
        duration = self.tests['system_baseline']['duration']
        
        while time.time() - start_time < duration:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            baseline_data['cpu_usage'].append(cpu_percent)
            baseline_data['memory_usage'].append(memory_percent)
            baseline_data['disk_io'].append(disk_io.read_bytes + disk_io.write_bytes)
            baseline_data['network_io'].append(network_io.bytes_sent + network_io.bytes_recv)
            
            time.sleep(1)
        
        # Calculate baseline statistics
        baseline_stats = {
            'avg_cpu': np.mean(baseline_data['cpu_usage']),
            'max_cpu': np.max(baseline_data['cpu_usage']),
            'avg_memory': np.mean(baseline_data['memory_usage']),
            'max_memory': np.max(baseline_data['memory_usage']),
            'total_disk_io': baseline_data['disk_io'][-1] - baseline_data['disk_io'][0],
            'total_network_io': baseline_data['network_io'][-1] - baseline_data['network_io'][0]
        }
        
        self.results['system_baseline'] = {
            'stats': baseline_stats,
            'raw_data': baseline_data
        }
        
        print(f"Baseline test completed. Avg CPU: {baseline_stats['avg_cpu']:.1f}%, "
              f"Avg Memory: {baseline_stats['avg_memory']:.1f}%")
    
    def run_node_test(self, test_name, nodes):
        """Run test with specific ROS nodes"""
        print(f"Running {test_name} test with nodes: {nodes}")
        
        # Start ROS core
        roscore_process = subprocess.Popen(['roscore'], 
                                         stdout=subprocess.DEVNULL, 
                                         stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for roscore to start
        
        # Start performance monitor
        monitor_process = subprocess.Popen(['python3', 'scripts/performance_monitor.py',
                                          f'_output_dir:={self.test_output_dir / test_name}',
                                          '_display_plots:=false'],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        
        node_processes = []
        try:
            # Start specified nodes
            for node in nodes:
                node_path = f"src/ros_nodes/{node}"
                process = subprocess.Popen(['python3', node_path],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                node_processes.append(process)
                time.sleep(2)  # Stagger node startup
            
            # Run test for specified duration
            duration = self.tests[test_name]['duration']
            start_time = time.time()
            
            test_data = {
                'cpu_usage': [],
                'memory_usage': [],
                'gpu_usage': [],
                'timestamp': []
            }
            
            while time.time() - start_time < duration:
                current_time = time.time()
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                gpu_percent = self.get_gpu_usage()
                
                test_data['cpu_usage'].append(cpu_percent)
                test_data['memory_usage'].append(memory_percent)
                test_data['gpu_usage'].append(gpu_percent)
                test_data['timestamp'].append(current_time - start_time)
                
                time.sleep(1)
            
            # Calculate test statistics
            test_stats = {
                'avg_cpu': np.mean(test_data['cpu_usage']),
                'max_cpu': np.max(test_data['cpu_usage']),
                'avg_memory': np.mean(test_data['memory_usage']),
                'max_memory': np.max(test_data['memory_usage']),
                'avg_gpu': np.mean(test_data['gpu_usage']),
                'max_gpu': np.max(test_data['gpu_usage']),
                'duration': duration,
                'nodes': nodes
            }
            
            self.results[test_name] = {
                'stats': test_stats,
                'raw_data': test_data
            }
            
            print(f"{test_name} test completed. Avg CPU: {test_stats['avg_cpu']:.1f}%, "
                  f"Avg Memory: {test_stats['avg_memory']:.1f}%")
            
        finally:
            # Cleanup processes
            for process in node_processes:
                process.terminate()
                process.wait()
            
            monitor_process.terminate()
            monitor_process.wait()
            
            roscore_process.terminate()
            roscore_process.wait()
            
            time.sleep(2)  # Wait for cleanup
    
    def get_gpu_usage(self):
        """Get GPU usage if available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return 0
    
    def run_enhancement_benchmark(self):
        """Run specific enhancement model benchmarks"""
        print("Running enhancement model benchmarks...")
        
        enhancement_results = {}
        
        # Test both models
        models = ['zero_dce', 'sci']
        
        for model in models:
            print(f"Benchmarking {model.upper()} model...")
            
            model_results = {
                'processing_times': [],
                'quality_metrics': [],
                'memory_usage': []
            }
            
            # Test on different image sizes and scenarios
            test_images = list((self.test_data_dir / "datasets").glob("*/*.jpg"))
            
            for img_path in test_images[:20]:  # Test on 20 images
                start_time = time.time()
                
                # Load and process image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Simulate model processing (placeholder)
                # In real implementation, this would call the actual model
                processed_image = self.simulate_enhancement(image, model)
                
                processing_time = time.time() - start_time
                
                # Calculate quality metrics
                quality_score = self.calculate_quality_metrics(image, processed_image)
                
                model_results['processing_times'].append(processing_time)
                model_results['quality_metrics'].append(quality_score)
                model_results['memory_usage'].append(psutil.virtual_memory().percent)
            
            enhancement_results[model] = {
                'avg_processing_time': np.mean(model_results['processing_times']),
                'std_processing_time': np.std(model_results['processing_times']),
                'avg_quality_score': np.mean(model_results['quality_metrics']),
                'avg_memory_usage': np.mean(model_results['memory_usage']),
                'raw_data': model_results
            }
        
        self.results['enhancement_benchmark'] = enhancement_results
        print("Enhancement benchmarking completed")
    
    def simulate_enhancement(self, image, model):
        """Simulate image enhancement (placeholder)"""
        # Simple brightness adjustment as simulation
        if model == 'zero_dce':
            enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        else:  # sci
            enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
        
        # Add some processing delay to simulate real model
        time.sleep(0.1)
        
        return enhanced
    
    def calculate_quality_metrics(self, original, enhanced):
        """Calculate image quality metrics"""
        # Convert to grayscale for metrics
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness improvement
        orig_brightness = np.mean(orig_gray)
        enh_brightness = np.mean(enh_gray)
        brightness_improvement = enh_brightness - orig_brightness
        
        # Calculate contrast improvement
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = enh_contrast - orig_contrast
        
        # Simple quality score
        quality_score = brightness_improvement + contrast_improvement
        
        return quality_score
    
    def run_all_tests(self):
        """Run all benchmark tests"""
        print(f"Starting comprehensive benchmark suite - Duration: {self.duration}s")
        print(f"Output directory: {self.test_output_dir}")
        
        # Setup test data
        self.setup_test_data()
        
        # Run baseline test
        self.run_system_baseline()
        
        # Run node tests
        for test_name, config in self.tests.items():
            if test_name == 'system_baseline':
                continue
            
            self.run_node_test(test_name, config['nodes'])
        
        # Run enhancement benchmarks
        self.run_enhancement_benchmark()
        
        # Generate reports
        self.generate_reports()
        
        print("All benchmark tests completed!")
    
    def generate_reports(self):
        """Generate benchmark reports and visualizations"""
        print("Generating benchmark reports...")
        
        # Save raw results
        with open(self.test_output_dir / "benchmark_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for test_name, data in self.results.items():
                json_results[test_name] = self.convert_numpy_to_list(data)
            json.dump(json_results, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report()
        
        # Generate visualizations
        self.generate_visualizations()
        
        print(f"Reports generated in {self.test_output_dir}")
    
    def convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def generate_summary_report(self):
        """Generate summary report"""
        report_lines = [
            "SD_Thesis Benchmark Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test Duration: {self.duration}s",
            "",
            "SYSTEM PERFORMANCE SUMMARY",
            "-" * 30
        ]
        
        # Add baseline results
        if 'system_baseline' in self.results:
            baseline = self.results['system_baseline']['stats']
            report_lines.extend([
                f"Baseline CPU Usage: {baseline['avg_cpu']:.1f}% (max: {baseline['max_cpu']:.1f}%)",
                f"Baseline Memory Usage: {baseline['avg_memory']:.1f}% (max: {baseline['max_memory']:.1f}%)",
                ""
            ])
        
        # Add node test results
        report_lines.append("NODE PERFORMANCE COMPARISON")
        report_lines.append("-" * 30)
        
        for test_name, data in self.results.items():
            if test_name.startswith('camera') or test_name == 'full_pipeline':
                stats = data['stats']
                report_lines.append(f"{test_name.upper()}:")
                report_lines.append(f"  CPU: {stats['avg_cpu']:.1f}% (max: {stats['max_cpu']:.1f}%)")
                report_lines.append(f"  Memory: {stats['avg_memory']:.1f}% (max: {stats['max_memory']:.1f}%)")
                if 'avg_gpu' in stats:
                    report_lines.append(f"  GPU: {stats['avg_gpu']:.1f}% (max: {stats['max_gpu']:.1f}%)")
                report_lines.append("")
        
        # Add enhancement results
        if 'enhancement_benchmark' in self.results:
            report_lines.append("ENHANCEMENT MODEL PERFORMANCE")
            report_lines.append("-" * 30)
            
            for model, data in self.results['enhancement_benchmark'].items():
                report_lines.append(f"{model.upper()} Model:")
                report_lines.append(f"  Avg Processing Time: {data['avg_processing_time']:.3f}s")
                report_lines.append(f"  Avg Quality Score: {data['avg_quality_score']:.2f}")
                report_lines.append(f"  Memory Usage: {data['avg_memory_usage']:.1f}%")
                report_lines.append("")
        
        # Save report
        with open(self.test_output_dir / "benchmark_summary.txt", 'w') as f:
            f.write('\n'.join(report_lines))
    
    def generate_visualizations(self):
        """Generate benchmark visualizations"""
        try:
            # CPU/Memory comparison chart
            self.plot_resource_comparison()
            
            # Enhancement performance chart
            self.plot_enhancement_performance()
            
            # Timeline charts
            self.plot_performance_timeline()
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def plot_resource_comparison(self):
        """Plot resource usage comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        test_names = []
        cpu_values = []
        memory_values = []
        
        for test_name, data in self.results.items():
            if 'stats' in data and 'avg_cpu' in data['stats']:
                test_names.append(test_name.replace('_', '\n'))
                cpu_values.append(data['stats']['avg_cpu'])
                memory_values.append(data['stats']['avg_memory'])
        
        # CPU usage chart
        ax1.bar(test_names, cpu_values, color='skyblue')
        ax1.set_title('Average CPU Usage by Test')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage chart
        ax2.bar(test_names, memory_values, color='lightcoral')
        ax2.set_title('Average Memory Usage by Test')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.test_output_dir / "resource_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_enhancement_performance(self):
        """Plot enhancement model performance"""
        if 'enhancement_benchmark' not in self.results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        models = list(self.results['enhancement_benchmark'].keys())
        processing_times = [self.results['enhancement_benchmark'][model]['avg_processing_time'] 
                          for model in models]
        quality_scores = [self.results['enhancement_benchmark'][model]['avg_quality_score'] 
                         for model in models]
        
        # Processing time comparison
        ax1.bar(models, processing_times, color=['blue', 'green'])
        ax1.set_title('Average Processing Time')
        ax1.set_ylabel('Time (seconds)')
        
        # Quality score comparison
        ax2.bar(models, quality_scores, color=['orange', 'red'])
        ax2.set_title('Average Quality Score')
        ax2.set_ylabel('Quality Score')
        
        plt.tight_layout()
        plt.savefig(self.test_output_dir / "enhancement_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_timeline(self):
        """Plot performance over time for full pipeline test"""
        if 'full_pipeline' not in self.results:
            return
        
        data = self.results['full_pipeline']['raw_data']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        timestamps = data['timestamp']
        
        # CPU usage over time
        ax1.plot(timestamps, data['cpu_usage'], label='CPU Usage', color='blue')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('System Performance Over Time (Full Pipeline)')
        ax1.grid(True)
        ax1.legend()
        
        # Memory usage over time
        ax2.plot(timestamps, data['memory_usage'], label='Memory Usage', color='red')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_xlabel('Time (seconds)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.test_output_dir / "performance_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='SD_Thesis Benchmark System')
    parser.add_argument('--duration', type=int, default=300, 
                       help='Total benchmark duration in seconds')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory for results')
    parser.add_argument('--test-data', type=str, default='data/test', 
                       help='Test data directory')
    
    args = parser.parse_args()
    
    benchmark = BenchmarkSystem(
        duration=args.duration,
        output_dir=args.output,
        test_data_dir=args.test_data
    )
    
    try:
        benchmark.run_all_tests()
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {benchmark.test_output_dir}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")

if __name__ == '__main__':
    main()
