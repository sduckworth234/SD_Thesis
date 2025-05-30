#!/usr/bin/env python3
"""
Real-time Performance Monitor for SD_Thesis System
Monitors system performance, ROS node statistics, and resource usage.
"""

import rospy
import psutil
import json
import time
import threading
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import sys
import os

class PerformanceMonitor:
    def __init__(self):
        rospy.init_node('performance_monitor', anonymous=True)
        
        # Performance data storage
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.camera_fps = deque(maxlen=100)
        self.detection_fps = deque(maxlen=100)
        self.slam_fps = deque(maxlen=100)
        self.enhancement_fps = deque(maxlen=100)
        
        self.timestamps = deque(maxlen=100)
        
        # Node statistics
        self.node_stats = {
            'camera': {'frames': 0, 'last_time': time.time()},
            'detection': {'frames': 0, 'last_time': time.time()},
            'slam': {'frames': 0, 'last_time': time.time()},
            'enhancement': {'frames': 0, 'last_time': time.time()}
        }
        
        # Parameters
        self.monitor_rate = rospy.get_param('~monitor_rate', 1.0)  # Hz
        self.save_data = rospy.get_param('~save_data', True)
        self.output_dir = rospy.get_param('~output_dir', 'logs/performance')
        self.display_plots = rospy.get_param('~display_plots', True)
        
        # Create output directory
        if self.save_data:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # ROS subscribers
        self.setup_subscribers()
        
        # Performance data file
        if self.save_data:
            self.performance_file = open(f"{self.output_dir}/performance_data.json", 'w')
        
        # Matplotlib setup for real-time plotting
        if self.display_plots:
            self.setup_plots()
        
        rospy.loginfo("Performance Monitor initialized")
    
    def setup_subscribers(self):
        """Setup ROS subscribers for monitoring"""
        # Camera monitoring
        rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)
        
        # Detection monitoring
        rospy.Subscriber('/detection/annotated_image', Image, self.detection_callback)
        
        # SLAM monitoring
        rospy.Subscriber('/slam/pose', PoseStamped, self.slam_callback)
        
        # Enhancement monitoring
        rospy.Subscriber('/enhancement/zero_dce', Image, self.enhancement_callback)
        
        # Node statistics
        rospy.Subscriber('/detection/statistics', String, self.detection_stats_callback)
        rospy.Subscriber('/slam/status', String, self.slam_stats_callback)
    
    def setup_plots(self):
        """Setup matplotlib plots for real-time visualization"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('SD_Thesis System Performance Monitor')
        
        # CPU and Memory plot
        self.ax_system = self.axes[0, 0]
        self.ax_system.set_title('System Resources')
        self.ax_system.set_ylabel('Usage (%)')
        self.ax_system.set_ylim(0, 100)
        
        # FPS plot
        self.ax_fps = self.axes[0, 1]
        self.ax_fps.set_title('Node FPS')
        self.ax_fps.set_ylabel('FPS')
        self.ax_fps.set_ylim(0, 40)
        
        # Processing Times
        self.ax_timing = self.axes[1, 0]
        self.ax_timing.set_title('Processing Times')
        self.ax_timing.set_ylabel('Time (ms)')
        
        # Network Usage
        self.ax_network = self.axes[1, 1]
        self.ax_network.set_title('ROS Network Usage')
        self.ax_network.set_ylabel('Messages/sec')
        
        plt.tight_layout()
        
        # Start animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                         interval=1000, blit=False)
    
    def camera_callback(self, msg):
        """Monitor camera node performance"""
        self.node_stats['camera']['frames'] += 1
    
    def detection_callback(self, msg):
        """Monitor detection node performance"""
        self.node_stats['detection']['frames'] += 1
    
    def slam_callback(self, msg):
        """Monitor SLAM node performance"""
        self.node_stats['slam']['frames'] += 1
    
    def enhancement_callback(self, msg):
        """Monitor enhancement node performance"""
        self.node_stats['enhancement']['frames'] += 1
    
    def detection_stats_callback(self, msg):
        """Process detection statistics"""
        try:
            stats = json.loads(msg.data)
            # Update detection-specific metrics
            if 'average_fps' in stats:
                self.detection_fps.append(stats['average_fps'])
        except json.JSONDecodeError:
            pass
    
    def slam_stats_callback(self, msg):
        """Process SLAM statistics"""
        try:
            stats = json.loads(msg.data)
            # Update SLAM-specific metrics
            if 'average_fps' in stats:
                self.slam_fps.append(stats['average_fps'])
        except json.JSONDecodeError:
            pass
    
    def collect_system_metrics(self):
        """Collect system-wide performance metrics"""
        current_time = time.time()
        
        # System resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)
        self.timestamps.append(current_time)
        
        # Calculate FPS for each node
        for node_name, stats in self.node_stats.items():
            time_diff = current_time - stats['last_time']
            if time_diff > 0:
                fps = stats['frames'] / time_diff
                
                if node_name == 'camera':
                    self.camera_fps.append(fps)
                elif node_name == 'detection' and len(self.detection_fps) == 0:
                    self.detection_fps.append(fps)
                elif node_name == 'slam' and len(self.slam_fps) == 0:
                    self.slam_fps.append(fps)
                elif node_name == 'enhancement':
                    self.enhancement_fps.append(fps)
                
                # Reset counters
                stats['frames'] = 0
                stats['last_time'] = current_time
        
        # GPU usage (if available)
        gpu_usage = self.get_gpu_usage()
        
        # Network statistics
        network_stats = self.get_network_stats()
        
        # Create performance record
        performance_data = {
            'timestamp': current_time,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'gpu_usage': gpu_usage,
            'network_stats': network_stats,
            'node_fps': {
                'camera': self.camera_fps[-1] if self.camera_fps else 0,
                'detection': self.detection_fps[-1] if self.detection_fps else 0,
                'slam': self.slam_fps[-1] if self.slam_fps else 0,
                'enhancement': self.enhancement_fps[-1] if self.enhancement_fps else 0
            }
        }
        
        # Save to file
        if self.save_data:
            self.performance_file.write(json.dumps(performance_data) + '\n')
            self.performance_file.flush()
        
        # Log summary
        rospy.loginfo(f"Performance - CPU: {cpu_percent:.1f}%, RAM: {memory_percent:.1f}%, "
                     f"Camera FPS: {self.camera_fps[-1] if self.camera_fps else 0:.1f}")
        
        return performance_data
    
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
    
    def get_network_stats(self):
        """Get network statistics"""
        try:
            network = psutil.net_io_counters()
            return {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        except:
            return {}
    
    def update_plots(self, frame):
        """Update real-time plots"""
        if not self.timestamps:
            return
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot system resources
        if self.cpu_usage and self.memory_usage:
            times = list(range(len(self.cpu_usage)))
            self.axes[0, 0].plot(times, list(self.cpu_usage), label='CPU %', color='red')
            self.axes[0, 0].plot(times, list(self.memory_usage), label='Memory %', color='blue')
            self.axes[0, 0].set_title('System Resources')
            self.axes[0, 0].set_ylabel('Usage (%)')
            self.axes[0, 0].set_ylim(0, 100)
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True)
        
        # Plot FPS
        if any([self.camera_fps, self.detection_fps, self.slam_fps, self.enhancement_fps]):
            max_len = max(len(self.camera_fps), len(self.detection_fps), 
                         len(self.slam_fps), len(self.enhancement_fps))
            times = list(range(max_len))
            
            if self.camera_fps:
                self.axes[0, 1].plot(times[-len(self.camera_fps):], 
                                   list(self.camera_fps), label='Camera', color='green')
            if self.detection_fps:
                self.axes[0, 1].plot(times[-len(self.detection_fps):], 
                                   list(self.detection_fps), label='Detection', color='orange')
            if self.slam_fps:
                self.axes[0, 1].plot(times[-len(self.slam_fps):], 
                                   list(self.slam_fps), label='SLAM', color='purple')
            if self.enhancement_fps:
                self.axes[0, 1].plot(times[-len(self.enhancement_fps):], 
                                   list(self.enhancement_fps), label='Enhancement', color='cyan')
            
            self.axes[0, 1].set_title('Node FPS')
            self.axes[0, 1].set_ylabel('FPS')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True)
        
        # Plot processing times (placeholder)
        self.axes[1, 0].set_title('Processing Times')
        self.axes[1, 0].set_ylabel('Time (ms)')
        self.axes[1, 0].grid(True)
        
        # Plot network usage (placeholder)
        self.axes[1, 1].set_title('ROS Network Usage')
        self.axes[1, 1].set_ylabel('Messages/sec')
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
    
    def run(self):
        """Main monitoring loop"""
        rate = rospy.Rate(self.monitor_rate)
        
        while not rospy.is_shutdown():
            try:
                self.collect_system_metrics()
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Performance monitoring error: {e}")
                rate.sleep()
    
    def shutdown(self):
        """Clean shutdown"""
        if self.save_data and hasattr(self, 'performance_file'):
            self.performance_file.close()
        
        if self.display_plots:
            plt.close('all')
        
        rospy.loginfo("Performance monitor shutdown complete")

def main():
    try:
        monitor = PerformanceMonitor()
        rospy.on_shutdown(monitor.shutdown)
        
        if monitor.display_plots:
            # Run monitoring in separate thread
            monitor_thread = threading.Thread(target=monitor.run)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Show plots
            plt.show()
        else:
            monitor.run()
            
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nShutting down performance monitor...")

if __name__ == '__main__':
    main()
