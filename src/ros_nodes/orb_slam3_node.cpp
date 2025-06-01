#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/String.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// ORB-SLAM3 headers
#include "/opt/ORB_SLAM3/include/System.h"
#include <sophus/se3.hpp>

class ORBSlam3ROS {
public:
    ORBSlam3ROS(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
    ~ORBSlam3ROS();
    
    void run();

private:
    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    // Subscribers
    ros::Subscriber rgb_sub_;
    ros::Subscriber depth_sub_;
    
    // Publishers
    ros::Publisher pose_pub_;
    ros::Publisher path_pub_;
    ros::Publisher map_points_pub_;
    ros::Publisher keyframes_pub_;
    ros::Publisher status_pub_;
    
    // TF
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    
    // ORB-SLAM3
    ORB_SLAM3::System* SLAM_;
    
    // Parameters
    std::string vocabulary_file_;
    std::string settings_file_;
    std::string orb_slam3_path_;
    
    // Current data
    cv::Mat current_rgb_;
    cv::Mat current_depth_;
    bool new_rgb_;
    bool new_depth_;
    std::mutex data_mutex_;
    
    // Trajectory
    nav_msgs::Path trajectory_msg_;
    
    // Callbacks
    void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);
    
    // Processing
    void processFrames();
    void publishPose(const cv::Mat& Tcw, const ros::Time& timestamp);
    void publishTrajectory(const ros::Time& timestamp);
    void publishMapPoints(const std::vector<cv::Point3f>& map_points, const ros::Time& timestamp);
    void publishKeyframes(const std::vector<cv::Mat>& keyframes, const ros::Time& timestamp);
    void publishStatus(const std::string& status);
    
    // Utilities
    geometry_msgs::PoseStamped matToPoseStamped(const cv::Mat& Tcw, const ros::Time& timestamp);
    sensor_msgs::PointCloud2 mapPointsToPointCloud(const std::vector<cv::Point3f>& map_points, const ros::Time& timestamp);
};

ORBSlam3ROS::ORBSlam3ROS(ros::NodeHandle &nh, ros::NodeHandle &nh_private) 
    : nh_(nh), nh_private_(nh_private), new_rgb_(false), new_depth_(false) {
    
    // Get parameters
    nh_private_.param<std::string>("vocabulary_file", vocabulary_file_, 
                                   "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt");
    nh_private_.param<std::string>("settings_file", settings_file_, 
                                   "/home/duck/Desktop/SD_Thesis/config/orbslam3_realsense.yaml");
    nh_private_.param<std::string>("orb_slam3_path", orb_slam3_path_, 
                                   "/opt/ORB_SLAM3");
    
    // Initialize ORB-SLAM3
    ROS_INFO("Initializing ORB-SLAM3...");
    ROS_INFO("Vocabulary: %s", vocabulary_file_.c_str());
    ROS_INFO("Settings: %s", settings_file_.c_str());
    
    try {
        // Initialize ORB-SLAM3 without viewer to avoid segfault in headless environment
        SLAM_ = new ORB_SLAM3::System(vocabulary_file_, settings_file_, ORB_SLAM3::System::RGBD, false);
        ROS_INFO("ORB-SLAM3 system initialized successfully (viewer disabled)");
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize ORB-SLAM3: %s", e.what());
        SLAM_ = nullptr;
    }
    
    // Subscribers
    rgb_sub_ = nh_.subscribe("/camera/color/image_raw", 1, &ORBSlam3ROS::rgbCallback, this);
    depth_sub_ = nh_.subscribe("/camera/depth/image_rect_raw", 1, &ORBSlam3ROS::depthCallback, this);
    
    // Publishers
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/orb_slam3/camera_pose", 1);
    path_pub_ = nh_.advertise<nav_msgs::Path>("/orb_slam3/trajectory", 1);
    map_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/orb_slam3/map_points", 1);
    keyframes_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/orb_slam3/keyframes", 1);
    status_pub_ = nh_.advertise<std_msgs::String>("/orb_slam3/status", 1);
    
    // Initialize trajectory
    trajectory_msg_.header.frame_id = "map";
    
    ROS_INFO("ORB-SLAM3 ROS wrapper initialized");
}

ORBSlam3ROS::~ORBSlam3ROS() {
    if (SLAM_) {
        SLAM_->Shutdown();
        delete SLAM_;
    }
}

void ORBSlam3ROS::rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        if (!msg || msg->data.empty()) {
            ROS_WARN("Received empty RGB image message");
            return;
        }
        
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        if (!cv_ptr || cv_ptr->image.empty()) {
            ROS_WARN("Failed to convert RGB image or image is empty");
            return;
        }
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_rgb_ = cv_ptr->image.clone();
        new_rgb_ = true;
        
        // Reset error counter on successful frame
        static int error_count = 0;
        error_count = 0;
        
    } catch (cv_bridge::Exception& e) {
        static int error_count = 0;
        error_count++;
        if (error_count % 10 == 1) { // Log every 10th error to avoid spam
            ROS_ERROR("cv_bridge RGB exception (count: %d): %s", error_count, e.what());
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Unexpected RGB callback exception: %s", e.what());
    }
}

void ORBSlam3ROS::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        if (!msg || msg->data.empty()) {
            ROS_WARN("Received empty depth image message");
            return;
        }
        
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "16UC1");
        if (!cv_ptr || cv_ptr->image.empty()) {
            ROS_WARN("Failed to convert depth image or image is empty");
            return;
        }
        
        std::lock_guard<std::mutex> lock(data_mutex_);
        current_depth_ = cv_ptr->image.clone();
        new_depth_ = true;
        
        // Reset error counter on successful frame
        static int error_count = 0;
        error_count = 0;
        
    } catch (cv_bridge::Exception& e) {
        static int error_count = 0;
        error_count++;
        if (error_count % 10 == 1) { // Log every 10th error to avoid spam
            ROS_ERROR("cv_bridge depth exception (count: %d): %s", error_count, e.what());
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Unexpected depth callback exception: %s", e.what());
    }
}

void ORBSlam3ROS::processFrames() {
    cv::Mat rgb, depth;
    bool has_new_data = false;
    
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!new_rgb_ || !new_depth_) {
            return;
        }
        
        if (current_rgb_.empty() || current_depth_.empty()) {
            ROS_WARN_THROTTLE(5.0, "Skipping frame processing - empty images");
            return;
        }
        
        rgb = current_rgb_.clone();
        depth = current_depth_.clone();
        new_rgb_ = false;
        new_depth_ = false;
        has_new_data = true;
    }
    
    if (!has_new_data || rgb.empty() || depth.empty()) {
        return;
    }
    
    ros::Time timestamp = ros::Time::now();
    
    // Process with ORB-SLAM3
    if (SLAM_) {
        try {
            Sophus::SE3f pose = SLAM_->TrackRGBD(rgb, depth, timestamp.toSec());
            
            // Convert Sophus SE3 to cv::Mat for compatibility with existing code
            Eigen::Matrix4f pose_matrix = pose.matrix();
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    Tcw.at<float>(i, j) = pose_matrix(i, j);
                }
            }
            
            if (!Tcw.empty()) {
                // Publish results
                publishPose(Tcw, timestamp);
                publishTrajectory(timestamp);
                publishStatus("TRACKING");
                
                // Get map points from ORB-SLAM3
                std::vector<ORB_SLAM3::MapPoint*> map_points = SLAM_->GetTrackedMapPoints();
                std::vector<cv::Point3f> points_3d;
                
                for (auto mp : map_points) {
                    if (mp && !mp->isBad()) {
                        try {
                            Eigen::Vector3f pos = mp->GetWorldPos();
                            points_3d.push_back(cv::Point3f(pos(0), pos(1), pos(2)));
                        } catch (const std::exception& e) {
                            // Skip invalid map points
                            continue;
                        }
                    }
                }
                
                if (!points_3d.empty()) {
                    publishMapPoints(points_3d, timestamp);
                }
            } else {
                publishStatus("LOST");
            }
        } catch (const std::exception& e) {
            static int error_count = 0;
            error_count++;
            if (error_count % 30 == 1) { // Log every 30th error (1 second at 30Hz)
                ROS_ERROR("ORB-SLAM3 tracking exception (count: %d): %s", error_count, e.what());
            }
            publishStatus("ERROR");
        }
    } else {
        // Fallback to mock pose if SLAM not initialized
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        static float t = 0;
        t += 0.033f; // 30 FPS
        
        // Create circular motion
        float radius = 2.0f;
        Tcw.at<float>(0, 3) = radius * cos(t * 0.1f);
        Tcw.at<float>(1, 3) = radius * sin(t * 0.1f);
        Tcw.at<float>(2, 3) = 1.0f + 0.5f * sin(t * 0.2f);
        
        // Publish results
        publishPose(Tcw, timestamp);
        publishTrajectory(timestamp);
        publishStatus("MOCK_TRACKING");
        
        // Mock map points
        static int frame_count = 0;
        frame_count++;
        if (frame_count % 10 == 0) {
            std::vector<cv::Point3f> mock_points;
            for (int i = 0; i < 100; ++i) {
                mock_points.push_back(cv::Point3f(
                    (rand() % 1000 - 500) / 100.0f,
                    (rand() % 1000 - 500) / 100.0f,
                    (rand() % 300) / 100.0f
                ));
            }
            publishMapPoints(mock_points, timestamp);
        }
    }
}

geometry_msgs::PoseStamped ORBSlam3ROS::matToPoseStamped(const cv::Mat& Tcw, const ros::Time& timestamp) {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = timestamp;
    pose.header.frame_id = "map";
    
    if (Tcw.empty()) {
        return pose;
    }
    
    // Extract translation
    pose.pose.position.x = Tcw.at<float>(0, 3);
    pose.pose.position.y = Tcw.at<float>(1, 3);
    pose.pose.position.z = Tcw.at<float>(2, 3);
    
    // Extract rotation and convert to quaternion
    cv::Mat R = Tcw.rowRange(0, 3).colRange(0, 3);
    
    // Convert rotation matrix to quaternion (simplified)
    tf2::Matrix3x3 tf_R(
        R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
        R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
        R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2)
    );
    
    tf2::Quaternion q;
    tf_R.getRotation(q);
    
    pose.pose.orientation = tf2::toMsg(q);
    
    return pose;
}

void ORBSlam3ROS::publishPose(const cv::Mat& Tcw, const ros::Time& timestamp) {
    geometry_msgs::PoseStamped pose = matToPoseStamped(Tcw, timestamp);
    pose_pub_.publish(pose);
    
    // Also publish TF
    geometry_msgs::TransformStamped transform;
    transform.header = pose.header;
    transform.child_frame_id = "camera_link";
    transform.transform.translation.x = pose.pose.position.x;
    transform.transform.translation.y = pose.pose.position.y;
    transform.transform.translation.z = pose.pose.position.z;
    transform.transform.rotation = pose.pose.orientation;
    
    tf_broadcaster_.sendTransform(transform);
}

void ORBSlam3ROS::publishTrajectory(const ros::Time& timestamp) {
    trajectory_msg_.header.stamp = timestamp;
    path_pub_.publish(trajectory_msg_);
}

sensor_msgs::PointCloud2 ORBSlam3ROS::mapPointsToPointCloud(const std::vector<cv::Point3f>& map_points, const ros::Time& timestamp) {
    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp = timestamp;
    cloud.header.frame_id = "map";
    
    cloud.width = map_points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    
    // Point cloud fields
    sensor_msgs::PointField field_x, field_y, field_z;
    field_x.name = "x"; field_x.offset = 0; field_x.datatype = sensor_msgs::PointField::FLOAT32; field_x.count = 1;
    field_y.name = "y"; field_y.offset = 4; field_y.datatype = sensor_msgs::PointField::FLOAT32; field_y.count = 1;
    field_z.name = "z"; field_z.offset = 8; field_z.datatype = sensor_msgs::PointField::FLOAT32; field_z.count = 1;
    
    cloud.fields = {field_x, field_y, field_z};
    cloud.point_step = 12;
    cloud.row_step = cloud.point_step * cloud.width;
    
    cloud.data.resize(cloud.row_step);
    float* data_ptr = reinterpret_cast<float*>(cloud.data.data());
    
    for (size_t i = 0; i < map_points.size(); ++i) {
        data_ptr[i * 3 + 0] = map_points[i].x;
        data_ptr[i * 3 + 1] = map_points[i].y;
        data_ptr[i * 3 + 2] = map_points[i].z;
    }
    
    return cloud;
}

void ORBSlam3ROS::publishMapPoints(const std::vector<cv::Point3f>& map_points, const ros::Time& timestamp) {
    sensor_msgs::PointCloud2 cloud = mapPointsToPointCloud(map_points, timestamp);
    map_points_pub_.publish(cloud);
}

void ORBSlam3ROS::publishStatus(const std::string& status) {
    std_msgs::String status_msg;
    status_msg.data = status;
    status_pub_.publish(status_msg);
}

void ORBSlam3ROS::run() {
    ros::Rate rate(30); // 30 Hz
    
    while (ros::ok()) {
        processFrames();
        ros::spinOnce();
        rate.sleep();
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "orb_slam3_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    try {
        ORBSlam3ROS orb_slam3(nh, nh_private);
        orb_slam3.run();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in ORB-SLAM3 node: %s", e.what());
        return 1;
    }
    
    return 0;
}
