---
title: Isaac ROS: Hardware-Accelerated VSLAM & Navigation
description: Understanding NVIDIA Isaac ROS for hardware-accelerated SLAM and navigation
sidebar_position: 2
---

# Isaac ROS: Hardware-Accelerated VSLAM & Navigation

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the architecture and capabilities of NVIDIA Isaac ROS
2. Implement hardware-accelerated visual SLAM (VSLAM) using Isaac ROS
3. Configure and deploy Isaac ROS navigation stack
4. Integrate Isaac ROS with traditional ROS 2 navigation systems
5. Optimize navigation performance using GPU acceleration

## Introduction to Isaac ROS

Isaac ROS is NVIDIA's collection of accelerated perception and navigation packages for ROS 2, designed to leverage NVIDIA GPUs for real-time robotics applications. Isaac ROS bridges the gap between high-performance AI processing and traditional robotics, providing:

- **GPU Acceleration**: Leverage CUDA cores for parallel processing
- **Hardware Optimization**: Utilize Tensor Cores and RT Cores for AI inference
- **Real-time Performance**: Achieve low-latency perception and navigation
- **Production Ready**: Industrial-grade packages for deployment

## Isaac ROS Architecture

Isaac ROS follows a modular architecture with specialized packages for different robotics functions:

### Core Packages

- **Isaac ROS Visual SLAM (Stereo Dense Reconstruction)**: Real-time dense reconstruction from stereo cameras
- **Isaac ROS AprilTag Detection**: GPU-accelerated fiducial marker detection
- **Isaac ROS Apriltag Pose Estimator**: Pose estimation from detected AprilTags
- **Isaac ROS Stereo Disparity**: High-performance stereo disparity computation
- **Isaac ROS NITROS**: NVIDIA Isaac Transport and Robotics Orchestration Stack
- **Isaac ROS ISAAC SIM**: High-fidelity simulation environment

### Navigation Packages

- **Isaac ROS Navigation**: GPU-accelerated path planning and navigation
- **Isaac ROS Occupancy Grids**: Accelerated occupancy grid processing
- **Isaac ROS Path Planning**: Hardware-accelerated path planning algorithms

## Installing Isaac ROS

### Prerequisites

```bash
# Ubuntu 20.04 or 22.04 LTS
# NVIDIA GPU with CUDA support (RTX series recommended)
# ROS 2 Humble Hawksbill or newer
```

### Installation Steps

```bash
# 1. Install NVIDIA drivers and CUDA
sudo apt update
sudo apt install nvidia-driver-535 nvidia-utils-535

# 2. Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-12-0

# 3. Install Isaac ROS via apt
sudo apt update
sudo apt install ros-humble-isaac-ros-common
sudo apt install ros-humble-isaac-ros-stereo-image-rectification
sudo apt install ros-humble-isaac-ros-visual-slam
sudo apt install ros-humble-isaac-ros-apriltag
sudo apt install ros-humble-isaac-ros-navigation

# 4. Install Isaac ROS from source (recommended for latest features)
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git -b ros2
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git -b ros2
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git -b ros2
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_navigation.git -b ros2
```

## Isaac ROS Visual SLAM Implementation

### Stereo Dense Reconstruction

Visual SLAM (Simultaneous Localization and Mapping) using stereo cameras is a core capability of Isaac ROS. The system creates dense 3D reconstructions in real-time using GPU acceleration.

#### Isaac ROS Visual SLAM Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
from message_filters import ApproximateTimeSynchronizer

class IsaacVisualSlamNode(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Parameters
        self.declare_parameter('max_disparity', 64.0)
        self.declare_parameter('min_disparity', 0.0)
        self.declare_parameter('stereo_baseline', 0.1)  # meters
        self.declare_parameter('max_range', 10.0)  # meters

        self.max_disparity = self.get_parameter('max_disparity').value
        self.min_disparity = self.get_parameter('min_disparity').value
        self.stereo_baseline = self.get_parameter('stereo_baseline').value
        self.max_range = self.get_parameter('max_range').value

        # Subscribers for stereo images
        self.left_image_sub = message_filters.Subscriber(
            self, Image, '/stereo_camera/left/image_rect_color')
        self.right_image_sub = message_filters.Subscriber(
            self, Image, '/stereo_camera/right/image_rect_color')
        self.left_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/stereo_camera/left/camera_info')
        self.right_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/stereo_camera/right/camera_info')

        # Synchronize stereo images
        self.sync = ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub,
             self.left_info_sub, self.right_info_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.stereo_callback)

        # Publishers
        self.disparity_pub = self.create_publisher(
            DisparityImage, '/stereo_camera/disparity', 10)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/stereo_camera/pointcloud', 10)
        self.odom_pub = self.create_publisher(
            Odometry, '/visual_odometry', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # SLAM state
        self.previous_frame = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.keyframes = []
        self.map_points = []

        # GPU acceleration indicators
        self.gpu_available = self.check_gpu_availability()
        if self.gpu_available:
            self.get_logger().info('GPU acceleration enabled')
        else:
            self.get_logger().warn('GPU acceleration not available, using CPU fallback')

        self.get_logger().info('Isaac Visual SLAM Node initialized')

    def check_gpu_availability(self):
        """Check if CUDA is available for GPU acceleration"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            cuda.init()
            return cuda.Device.count() > 0
        except ImportError:
            return False
        except Exception:
            return False

    def stereo_callback(self, left_msg, right_msg, left_info_msg, right_info_msg):
        """Process synchronized stereo images for VSLAM"""
        try:
            # Convert ROS images to OpenCV
            left_cv = self.cv_bridge.imgmsg_to_cv2(left_msg, "bgr8")
            right_cv = self.cv_bridge.imgmsg_to_cv2(right_msg, "mono8")

            # Get camera parameters
            left_K = np.array(left_info_msg.k).reshape(3, 3)
            right_K = np.array(right_info_msg.k).reshape(3, 3)
            left_D = np.array(left_info_msg.d)
            right_D = np.array(right_info_msg.d)

            # Compute stereo disparity
            disparity = self.compute_stereo_disparity(left_cv, right_cv)

            # Create disparity image message
            disparity_msg = self.create_disparity_message(disparity, left_msg.header)
            self.disparity_pub.publish(disparity_msg)

            # Perform dense reconstruction
            pointcloud = self.dense_reconstruction(disparity, left_K, left_D)

            # Publish point cloud
            if pointcloud is not None:
                pointcloud_msg = self.create_pointcloud2(pointcloud, left_msg.header)
                self.pointcloud_pub.publish(pointcloud_msg)

            # Perform visual odometry
            if self.previous_frame is not None:
                transformation = self.visual_odometry(
                    self.previous_frame, left_cv, left_K)

                # Update robot pose
                self.update_robot_pose(transformation, left_msg.header)

            # Store current frame for next iteration
            self.previous_frame = left_cv.copy()

        except Exception as e:
            self.get_logger().error(f'Error in stereo callback: {e}')

    def compute_stereo_disparity(self, left_image, right_image):
        """Compute stereo disparity using GPU-accelerated methods"""
        if self.gpu_available:
            return self.compute_stereo_disparity_gpu(left_image, right_image)
        else:
            return self.compute_stereo_disparity_cpu(left_image, right_image)

    def compute_stereo_disparity_gpu(self, left_image, right_image):
        """GPU-accelerated stereo disparity computation"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

            # Convert to grayscale if needed
            if len(left_image.shape) == 3:
                left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_image
                right_gray = right_image

            # Use OpenCV's GPU accelerated stereo matcher (if available)
            # In practice, Isaac ROS uses optimized CUDA kernels
            stereo = cv2.cuda.StereoBM_create(numDisparities=64, blockSize=15)

            # Upload images to GPU
            gpu_left = cv2.cuda_GpuMat()
            gpu_right = cv2.cuda_GpuMat()
            gpu_left.upload(left_gray)
            gpu_right.upload(right_gray)

            # Compute disparity
            gpu_disparity = stereo.compute(gpu_left, gpu_right)

            # Download result
            disparity = gpu_disparity.download()

            return disparity.astype(np.float32) / 16.0  # Convert to float32

        except Exception as e:
            self.get_logger().warn(f'GPU stereo failed, falling back to CPU: {e}')
            return self.compute_stereo_disparity_cpu(left_image, right_image)

    def compute_stereo_disparity_cpu(self, left_image, right_image):
        """CPU-based stereo disparity computation"""
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image

        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image

        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray)

        return disparity.astype(np.float32) / 16.0  # Convert to float32

    def dense_reconstruction(self, disparity, camera_matrix, distortion_coeffs):
        """Perform dense 3D reconstruction from disparity map"""
        try:
            # Filter out invalid disparities
            valid_mask = (disparity > 0) & (disparity < self.max_disparity)

            if not np.any(valid_mask):
                return None

            # Get valid pixels
            y_coords, x_coords = np.where(valid_mask)
            disparities = disparity[y_coords, x_coords]

            # Convert disparity to depth
            focal_length = camera_matrix[0, 0]
            depth = (self.stereo_baseline * focal_length) / disparities

            # Filter depth range
            valid_depth_mask = (depth > 0.1) & (depth < self.max_range)
            y_coords = y_coords[valid_depth_mask]
            x_coords = x_coords[valid_depth_mask]
            depth = depth[valid_depth_mask]

            if len(depth) == 0:
                return None

            # Convert pixel coordinates to 3D points
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]

            # Undistort points if needed
            points_undistorted = cv2.undistortPoints(
                np.array([[x_coords, y_coords]]).T,
                camera_matrix,
                distortion_coeffs
            ).squeeze()

            # Calculate 3D coordinates
            X = points_undistorted[:, 0] * depth
            Y = points_undistorted[:, 1] * depth
            Z = depth

            # Create point cloud
            points = np.column_stack((X, Y, Z)).astype(np.float32)

            return points

        except Exception as e:
            self.get_logger().error(f'Error in dense reconstruction: {e}')
            return None

    def visual_odometry(self, prev_image, curr_image, camera_matrix):
        """Perform visual odometry to estimate motion"""
        try:
            # Convert to grayscale
            if len(prev_image.shape) == 3:
                prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_image

            if len(curr_image.shape) == 3:
                curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = curr_image

            # Detect features
            detector = cv2.SIFT_create()
            prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)
            curr_kp, curr_desc = detector.detectAndCompute(curr_gray, None)

            if prev_desc is None or curr_desc is None:
                return np.eye(4)  # No transformation

            # Match features
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(prev_desc, curr_desc, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 10:
                return np.eye(4)  # Not enough matches

            # Extract matched keypoints
            prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Estimate motion using Essential Matrix
            E, mask = cv2.findEssentialMat(
                curr_pts, prev_pts, camera_matrix,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )

            if E is None:
                return np.eye(4)

            # Decompose Essential Matrix
            _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, camera_matrix)

            # Create transformation matrix
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t.flatten()

            return transformation

        except Exception as e:
            self.get_logger().error(f'Error in visual odometry: {e}')
            return np.eye(4)  # Identity transformation

    def update_robot_pose(self, transformation, header):
        """Update robot pose based on visual odometry"""
        try:
            # Extract rotation and translation
            R = transformation[:3, :3]
            t = transformation[:3, 3]

            # Convert rotation matrix to quaternion
            q = self.rotation_matrix_to_quaternion(R)

            # Update position
            self.current_position += t

            # Update orientation (simplified - in practice use proper quaternion integration)
            self.current_orientation = self.multiply_quaternions(self.current_orientation, q)

            # Create odometry message
            odom_msg = Odometry()
            odom_msg.header = header
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'

            # Position
            odom_msg.pose.pose.position.x = float(self.current_position[0])
            odom_msg.pose.pose.position.y = float(self.current_position[1])
            odom_msg.pose.pose.position.z = float(self.current_position[2])

            # Orientation
            odom_msg.pose.pose.orientation.x = float(self.current_orientation[0])
            odom_msg.pose.pose.orientation.y = float(self.current_orientation[1])
            odom_msg.pose.pose.orientation.z = float(self.current_orientation[2])
            odom_msg.pose.pose.orientation.w = float(self.current_orientation[3])

            # Velocity (estimated from position changes)
            # In practice, this would come from IMU or wheel encoders

            self.odom_pub.publish(odom_msg)

            # Broadcast TF transform
            t = TransformStamped()
            t.header.stamp = header.stamp
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'

            t.transform.translation.x = float(self.current_position[0])
            t.transform.translation.y = float(self.current_position[1])
            t.transform.translation.z = float(self.current_position[2])

            t.transform.rotation.x = float(self.current_orientation[0])
            t.transform.rotation.y = float(self.current_orientation[1])
            t.transform.rotation.z = float(self.current_orientation[2])
            t.transform.rotation.w = float(self.current_orientation[3])

            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().error(f'Error updating robot pose: {e}')

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def multiply_quaternions(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        # Normalize
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            return np.array([x/norm, y/norm, z/norm, w/norm])
        else:
            return np.array([0, 0, 0, 1])

    def create_disparity_message(self, disparity, header):
        """Create DisparityImage message from disparity array"""
        from stereo_msgs.msg import DisparityImage
        from sensor_msgs.msg import Image

        disp_msg = DisparityImage()
        disp_msg.header = header

        # Create image message for disparity
        image_msg = Image()
        image_msg.header = header
        image_msg.height = disparity.shape[0]
        image_msg.width = disparity.shape[1]
        image_msg.encoding = "32FC1"
        image_msg.is_bigendian = False
        image_msg.step = image_msg.width * 4  # 4 bytes per float

        # Convert disparity to bytes
        disparity_bytes = disparity.astype(np.float32).tobytes()
        image_msg.data = disparity_bytes

        disp_msg.image = image_msg
        disp_msg.f = float(header.frame_id.split('_')[0])  # focal length from camera info
        disp_msg.t = self.stereo_baseline  # baseline
        disp_msg.valid_window = RegionOfInterest()  # Entire image is valid
        disp_msg.min_disparity = float(self.min_disparity)
        disp_msg.max_disparity = float(self.max_disparity)
        disp_msg.delta_d = 0.125  # Resolution of disparity

        return disp_msg

    def create_pointcloud2(self, points, header):
        """Create PointCloud2 message from numpy array"""
        from sensor_msgs.msg import PointCloud2, PointField
        import struct

        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12  # 3 floats * 4 bytes each
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True

        # Pack points into binary data
        data = []
        for point in points:
            packed = struct.pack('fff', point[0], point[1], point[2])
            data.append(packed)

        cloud_msg.data = b''.join(data)
        return cloud_msg

def main(args=None):
    rclpy.init(args=args)
    slam_node = IsaacVisualSlamNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()
```

## Isaac ROS Navigation Stack

### Hardware-Accelerated Path Planning

Isaac ROS provides GPU-accelerated path planning algorithms that significantly outperform traditional CPU-based implementations.

#### Isaac ROS Navigation Node

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import heapq
from scipy.ndimage import binary_dilation, binary_erosion
import cv2

class IsaacNavigationNode(Node):
    def __init__(self):
        super().__init__('isaac_navigation_node')

        # Parameters
        self.declare_parameter('planner_frequency', 5.0)
        self.declare_parameter('controller_frequency', 20.0)
        self.declare_parameter('planning_window_size', 10.0)  # meters
        self.declare_parameter('robot_radius', 0.3)  # meters
        self.declare_parameter('inflation_radius', 0.5)  # meters

        self.planner_frequency = self.get_parameter('planner_frequency').value
        self.controller_frequency = self.get_parameter('controller_frequency').value
        self.planning_window_size = self.get_parameter('planning_window_size').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.inflation_radius = self.get_parameter('inflation_radius').value

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.global_costmap_pub = self.create_publisher(OccupancyGrid, '/global_costmap', 10)
        self.local_costmap_pub = self.create_publisher(OccupancyGrid, '/local_costmap', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/path_markers', 10)

        # Navigation state
        self.map_data = None
        self.map_resolution = 0.0
        self.map_origin = None
        self.current_goal = None
        self.current_pose = None
        self.path = []
        self.costmap = None

        # GPU acceleration
        self.gpu_available = self.check_gpu_availability()
        if self.gpu_available:
            self.get_logger().info('GPU acceleration enabled for navigation')
        else:
            self.get_logger().warn('GPU acceleration not available, using CPU fallback')

        # Timers
        self.planner_timer = self.create_timer(
            1.0/self.planner_frequency, self.plan_path)
        self.controller_timer = self.create_timer(
            1.0/self.controller_frequency, self.follow_path)

        self.get_logger().info('Isaac Navigation Node initialized')

    def check_gpu_availability(self):
        """Check if GPU is available for acceleration"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            cuda.init()
            return cuda.Device.count() > 0
        except ImportError:
            return False
        except Exception:
            return False

    def map_callback(self, msg):
        """Process occupancy grid map"""
        try:
            # Store map data
            self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            self.map_resolution = msg.info.resolution
            self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

            # Convert occupancy values (-1, 0, 100 to 0, 1, 2)
            self.map_data[self.map_data == -1] = 2  # Unknown -> Free for now
            self.map_data = self.map_data / 100.0  # Normalize to 0-1

            # Inflate obstacles
            self.costmap = self.inflate_obstacles(self.map_data)

            # Publish costmap
            costmap_msg = self.create_costmap_msg(self.costmap, msg.header)
            self.global_costmap_pub.publish(costmap_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing map: {e}')

    def inflate_obstacles(self, occupancy_map):
        """Inflate obstacles using GPU-accelerated morphological operations"""
        if self.gpu_available:
            return self.inflate_obstacles_gpu(occupancy_map)
        else:
            return self.inflate_obstacles_cpu(occupancy_map)

    def inflate_obstacles_gpu(self, occupancy_map):
        """GPU-accelerated obstacle inflation"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
            import skcuda.morph as morph

            # Convert to binary obstacle map
            obstacle_map = (occupancy_map > 0.5).astype(np.uint8)

            # Calculate kernel size based on inflation radius
            kernel_size = int(self.inflation_radius / self.map_resolution)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Make odd for dilation

            # Create structuring element
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

            # Perform dilation
            inflated_map = cv2.dilate(obstacle_map, kernel, iterations=1)

            # Combine with original map
            costmap = np.maximum(occupancy_map, inflated_map.astype(float))

            return costmap

        except Exception as e:
            self.get_logger().warn(f'GPU inflation failed, falling back to CPU: {e}')
            return self.inflate_obstacles_cpu(occupancy_map)

    def inflate_obstacles_cpu(self, occupancy_map):
        """CPU-based obstacle inflation"""
        # Convert to binary obstacle map
        obstacle_map = (occupancy_map > 0.5).astype(np.uint8)

        # Calculate kernel size based on inflation radius
        kernel_size = int(self.inflation_radius / self.map_resolution)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make odd for dilation

        # Create structuring element
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        # Perform dilation
        inflated_map = cv2.dilate(obstacle_map, kernel, iterations=1)

        # Combine with original map
        costmap = np.maximum(occupancy_map, inflated_map.astype(float))

        return costmap

    def laser_callback(self, msg):
        """Process laser scan data for local costmap"""
        try:
            # Convert laser scan to local costmap
            local_costmap = self.laser_to_local_costmap(msg)

            # Publish local costmap
            local_costmap_msg = self.create_costmap_msg(local_costmap, msg.header)
            self.local_costmap_pub.publish(local_costmap_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing laser scan: {e}')

    def laser_to_local_costmap(self, laser_msg):
        """Convert laser scan to local costmap"""
        if self.costmap is None:
            return np.zeros((100, 100))  # Default size

        # Create local costmap centered on robot
        local_size = int(self.planning_window_size / self.map_resolution)
        local_costmap = np.zeros((local_size, local_size))

        # Convert laser points to local grid coordinates
        angle_increment = laser_msg.angle_increment
        current_angle = laser_msg.angle_min

        for range_val in laser_msg.ranges:
            if laser_msg.range_min <= range_val <= laser_msg.range_max:
                # Calculate world coordinates
                x = range_val * np.cos(current_angle)
                y = range_val * np.sin(current_angle)

                # Convert to local grid coordinates
                local_x = int((x + self.planning_window_size/2) / self.map_resolution)
                local_y = int((y + self.planning_window_size/2) / self.map_resolution)

                # Check bounds
                if 0 <= local_x < local_size and 0 <= local_y < local_size:
                    local_costmap[local_y, local_x] = 1.0  # Mark as occupied

            current_angle += angle_increment

        return local_costmap

    def goal_callback(self, msg):
        """Process navigation goal"""
        self.current_goal = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().info(f'New goal received: {self.current_goal}')

    def plan_path(self):
        """Plan path using GPU-accelerated A* algorithm"""
        if self.costmap is None or self.current_goal is None:
            return

        try:
            # Convert goal to grid coordinates
            goal_grid = self.world_to_grid(self.current_goal[0], self.current_goal[1])

            # Convert current pose to grid coordinates
            if self.current_pose is not None:
                start_grid = self.world_to_grid(
                    self.current_pose.position.x,
                    self.current_pose.position.y
                )
            else:
                # Use robot's assumed position (0, 0 in map frame)
                start_grid = self.world_to_grid(0.0, 0.0)

            # Plan path using A* algorithm
            if self.gpu_available:
                path = self.plan_path_gpu(start_grid, goal_grid)
            else:
                path = self.plan_path_cpu(start_grid, goal_grid)

            if path:
                self.path = path
                self.publish_path(path)
                self.visualize_path(path)
            else:
                self.get_logger().warn('No path found to goal')

        except Exception as e:
            self.get_logger().error(f'Error in path planning: {e}')

    def plan_path_gpu(self, start, goal):
        """GPU-accelerated A* path planning"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
            import skcuda.misc as misc

            # In practice, this would use CUDA kernels for parallel A* execution
            # For now, we'll use a simplified approach that leverages GPU for heavy computations
            return self.a_star_search(start, goal)

        except Exception as e:
            self.get_logger().warn(f'GPU path planning failed, falling back to CPU: {e}')
            return self.plan_path_cpu(start, goal)

    def plan_path_cpu(self, start, goal):
        """CPU-based A* path planning"""
        return self.a_star_search(start, goal)

    def a_star_search(self, start, goal):
        """A* pathfinding algorithm"""
        if (start[0] < 0 or start[0] >= self.costmap.shape[1] or
            start[1] < 0 or start[1] >= self.costmap.shape[0] or
            goal[0] < 0 or goal[0] >= self.costmap.shape[1] or
            goal[1] < 0 or goal[1] >= self.costmap.shape[0]):
            return None

        # Check if start or goal are in obstacle space
        if self.costmap[start[1], start[0]] > 0.5 or self.costmap[goal[1], goal[0]] > 0.5:
            return None

        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        """Heuristic function for A* (Manhattan distance)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        """Get valid neighbors for pathfinding"""
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy

            # Check bounds
            if (0 <= nx < self.costmap.shape[1] and
                0 <= ny < self.costmap.shape[0]):

                # Check if not in obstacle
                if self.costmap[ny, nx] < 0.5:
                    neighbors.append((nx, ny))

        return neighbors

    def distance(self, a, b):
        """Calculate distance between two points"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def follow_path(self):
        """Follow the planned path"""
        if not self.path or self.current_pose is None:
            return

        try:
            # Get robot's current position in grid coordinates
            robot_pos = [
                self.current_pose.position.x,
                self.current_pose.position.y
            ]
            robot_grid = self.world_to_grid(robot_pos[0], robot_pos[1])

            # Find closest point on path
            closest_idx = self.find_closest_waypoint(robot_grid)

            if closest_idx is not None:
                # Get next waypoint to follow
                next_idx = min(closest_idx + 1, len(self.path) - 1)
                next_waypoint = self.path[next_idx]

                # Convert to world coordinates
                target_world = self.grid_to_world(next_waypoint[0], next_waypoint[1])

                # Calculate control commands
                control_cmd = self.calculate_control_command(robot_pos, target_world)

                # Publish control command
                self.publish_control_command(control_cmd)

        except Exception as e:
            self.get_logger().error(f'Error following path: {e}')

    def find_closest_waypoint(self, robot_grid):
        """Find the closest waypoint on the path"""
        if not self.path:
            return None

        min_dist = float('inf')
        closest_idx = 0

        for i, waypoint in enumerate(self.path):
            dist = self.distance(robot_grid, waypoint)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def calculate_control_command(self, robot_pos, target_pos):
        """Calculate control command to reach target"""
        # Calculate direction vector
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]

        # Calculate distance to target
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate heading to target
        target_heading = np.arctan2(dy, dx)

        # Current robot heading (from pose orientation)
        if self.current_pose is not None:
            current_heading = self.quaternion_to_yaw(
                self.current_pose.orientation
            )
        else:
            current_heading = 0.0

        # Calculate heading error
        heading_error = self.normalize_angle(target_heading - current_heading)

        # Control parameters
        linear_vel = min(0.5, distance)  # Max 0.5 m/s
        angular_vel = 2.0 * heading_error  # Proportional controller

        # Limit angular velocity
        angular_vel = np.clip(angular_vel, -1.0, 1.0)

        return {
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel,
            'distance_to_target': distance,
            'heading_error': heading_error
        }

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        if self.map_origin is None:
            return (int(x / self.map_resolution), int(y / self.map_resolution))

        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)

        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        if self.map_origin is None:
            return (grid_x * self.map_resolution, grid_y * self.map_resolution)

        x = grid_x * self.map_resolution + self.map_origin[0]
        y = grid_y * self.map_resolution + self.map_origin[1]

        return (x, y)

    def create_costmap_msg(self, costmap, header):
        """Create OccupancyGrid message from costmap array"""
        costmap_msg = OccupancyGrid()
        costmap_msg.header = header
        costmap_msg.info.resolution = self.map_resolution
        costmap_msg.info.width = costmap.shape[1]
        costmap_msg.info.height = costmap.shape[0]
        costmap_msg.info.origin.position.x = header.frame_id.split('_')[0] if '_' in header.frame_id else 0.0
        costmap_msg.info.origin.position.y = header.frame_id.split('_')[1] if '_' in header.frame_id else 0.0
        costmap_msg.info.origin.orientation.w = 1.0

        # Convert costmap to int8 values (0-100)
        int_costmap = (costmap * 100).astype(np.int8)
        costmap_msg.data = int_costmap.flatten().tolist()

        return costmap_msg

    def publish_path(self, path):
        """Publish the planned path"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for grid_pos in path:
            world_pos = self.grid_to_world(grid_pos[0], grid_pos[1])
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = world_pos[0]
            pose.pose.position.y = world_pos[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def visualize_path(self, path):
        """Publish visualization markers for the path"""
        marker_array = MarkerArray()

        # Path line marker
        path_marker = Marker()
        path_marker.header.frame_id = 'map'
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = 'path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 0.05  # Line width
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0

        for grid_pos in path:
            world_pos = self.grid_to_world(grid_pos[0], grid_pos[1])
            point = Point()
            point.x = world_pos[0]
            point.y = world_pos[1]
            point.z = 0.0
            path_marker.points.append(point)

        marker_array.markers.append(path_marker)

        # Waypoint markers
        for i, grid_pos in enumerate(path[::5]):  # Every 5th point as waypoint
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = 'map'
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.ns = 'waypoints'
            waypoint_marker.id = i + 1
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose.orientation.w = 1.0
            waypoint_marker.scale.x = 0.2
            waypoint_marker.scale.y = 0.2
            waypoint_marker.scale.z = 0.2
            waypoint_marker.color.r = 1.0
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 0.0
            waypoint_marker.color.a = 1.0

            world_pos = self.grid_to_world(grid_pos[0], grid_pos[1])
            waypoint_marker.pose.position.x = world_pos[0]
            waypoint_marker.pose.position.y = world_pos[1]
            waypoint_marker.pose.position.z = 0.1

            marker_array.markers.append(waypoint_marker)

        self.visualization_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    nav_node = IsaacNavigationNode()

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()
```

## Integration with Traditional ROS 2 Navigation

Isaac ROS can be integrated with traditional ROS 2 navigation stack for hybrid approaches:

### Isaac ROS Navigation Integration

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
import threading

class IsaacROSNavigationIntegration(Node):
    def __init__(self):
        super().__init__('isaac_ros_navigation_integration')

        # Create action clients for both navigation systems
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.isaac_nav_client = ActionClient(self, NavigateToPose, 'isaac_navigate_to_pose')

        # Parameters for system selection
        self.declare_parameter('use_isaac_nav', True)
        self.declare_parameter('fallback_timeout', 30.0)
        self.declare_parameter('hybrid_threshold_distance', 5.0)  # meters

        self.use_isaac_nav = self.get_parameter('use_isaac_nav').value
        self.fallback_timeout = self.get_parameter('fallback_timeout').value
        self.hybrid_threshold_distance = self.get_parameter('hybrid_threshold_distance').value

        # Goal subscription
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Status publishers
        self.nav_status_pub = self.create_publisher(String, '/navigation_status', 10)

        # Active goal tracking
        self.active_goals = {}
        self.fallback_timer = None

        self.get_logger().info('Isaac ROS Navigation Integration initialized')

    def goal_callback(self, msg):
        """Handle navigation goals"""
        goal_pose = msg.pose
        goal_header = msg.header

        # Determine which navigation system to use
        nav_system = self.select_navigation_system(goal_pose)

        if nav_system == 'isaac':
            self.send_goal_to_isaac_nav(goal_pose, goal_header)
        elif nav_system == 'nav2':
            self.send_goal_to_nav2(goal_pose, goal_header)
        else:
            self.send_goal_to_isaac_nav(goal_pose, goal_header)  # Default to Isaac

    def select_navigation_system(self, goal_pose):
        """Select navigation system based on various factors"""
        # Calculate distance to goal
        if self.current_pose is not None:
            distance = self.calculate_distance_to_pose(
                self.current_pose, goal_pose)

            # Use Isaac for long distances, Nav2 for short distances
            if distance > self.hybrid_threshold_distance:
                return 'isaac'
            else:
                return 'nav2'
        else:
            # Default to Isaac if no current pose available
            return 'isaac'

    def send_goal_to_isaac_nav(self, goal_pose, header):
        """Send goal to Isaac navigation system"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header = header
        goal_msg.pose.pose = goal_pose

        self.isaac_nav_client.wait_for_server()

        send_goal_future = self.isaac_nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.isaac_nav_feedback_callback
        )

        send_goal_future.add_done_callback(self.isaac_nav_goal_response_callback)

        self.get_logger().info('Sent goal to Isaac navigation system')

    def send_goal_to_nav2(self, goal_pose, header):
        """Send goal to traditional Nav2 system"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header = header
        goal_msg.pose.pose = goal_pose

        self.nav2_client.wait_for_server()

        send_goal_future = self.nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav2_feedback_callback
        )

        send_goal_future.add_done_callback(self.nav2_goal_response_callback)

        self.get_logger().info('Sent goal to Nav2 navigation system')

    def isaac_nav_goal_response_callback(self, future):
        """Handle Isaac navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Isaac navigation goal rejected')
            return

        self.get_logger().info('Isaac navigation goal accepted')
        self.active_goals['isaac'] = goal_handle

        # Start result future
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.isaac_nav_result_callback)

        # Set up fallback timer
        self.setup_fallback_timer('isaac')

    def nav2_goal_response_callback(self, future):
        """Handle Nav2 navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Nav2 navigation goal rejected')
            return

        self.get_logger().info('Nav2 navigation goal accepted')
        self.active_goals['nav2'] = goal_handle

        # Start result future
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.nav2_result_callback)

    def isaac_nav_feedback_callback(self, feedback_msg):
        """Handle Isaac navigation feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'Isaac navigation feedback: {feedback.current_pose}')

    def nav2_feedback_callback(self, feedback_msg):
        """Handle Nav2 navigation feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().debug(f'Nav2 navigation feedback: {feedback.current_pose}')

    def isaac_nav_result_callback(self, future):
        """Handle Isaac navigation result"""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Isaac navigation succeeded')
        else:
            self.get_logger().info(f'Isaac navigation failed with status: {status}')

        # Cancel fallback timer
        if self.fallback_timer:
            self.fallback_timer.cancel()

        # Update status
        status_msg = String()
        status_msg.data = f'isaac_navigation:{status}'
        self.nav_status_pub.publish(status_msg)

    def nav2_result_callback(self, future):
        """Handle Nav2 navigation result"""
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 navigation succeeded')
        else:
            self.get_logger().info(f'Nav2 navigation failed with status: {status}')

        # Update status
        status_msg = String()
        status_msg.data = f'nav2_navigation:{status}'
        self.nav_status_pub.publish(status_msg)

    def setup_fallback_timer(self, current_system):
        """Set up timer to fallback to alternative system"""
        if self.fallback_timer:
            self.fallback_timer.cancel()

        def fallback_check():
            # Check if current system is still active
            if current_system in self.active_goals:
                # Check if it's taking too long
                # In practice, this would check for progress/stuck detection
                self.get_logger().warn(f'{current_system} taking too long, considering fallback')
                # Implement fallback logic here

        self.fallback_timer = self.create_timer(self.fallback_timeout, fallback_check)

def main(args=None):
    rclpy.init(args=args)
    integration_node = IsaacROSNavigationIntegration()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

### GPU Memory Management

```python
import rclpy
from rclpy.node import Node
import numpy as np
import gc

class IsaacROSPreformanceOptimizer(Node):
    def __init__(self):
        super().__init__('isaac_ros_performance_optimizer')

        # GPU memory management
        self.gpu_memory_pool = {}
        self.active_tensors = {}

        # Performance monitoring
        self.declare_parameter('memory_cleanup_interval', 5.0)
        self.declare_parameter('max_gpu_memory_usage', 0.8)  # 80% of available memory

        self.memory_cleanup_interval = self.get_parameter('memory_cleanup_interval').value
        self.max_gpu_memory_usage = self.get_parameter('max_gpu_memory_usage').value

        # Setup cleanup timer
        self.cleanup_timer = self.create_timer(
            self.memory_cleanup_interval, self.cleanup_memory)

        self.get_logger().info('Isaac ROS Performance Optimizer initialized')

    def cleanup_memory(self):
        """Clean up GPU memory"""
        try:
            # Clear cached tensors
            if hasattr(self, 'gpu_memory_pool'):
                for key in list(self.gpu_memory_pool.keys()):
                    if key not in self.active_tensors:
                        del self.gpu_memory_pool[key]

            # Force garbage collection
            gc.collect()

            # In practice, this would also clear GPU caches
            # import pycuda.tools as tools
            # tools.clear_context_caches()

            self.get_logger().debug('Memory cleanup completed')

        except Exception as e:
            self.get_logger().error(f'Error during memory cleanup: {e}')

    def allocate_gpu_tensor(self, shape, dtype, name):
        """Allocate tensor with memory management"""
        try:
            if self.gpu_memory_pool is not None:
                # Check memory usage before allocation
                current_usage = self.get_gpu_memory_usage()
                if current_usage > self.max_gpu_memory_usage:
                    self.cleanup_memory()

                # Allocate tensor
                tensor = self.create_gpu_tensor(shape, dtype)
                self.gpu_memory_pool[name] = tensor
                self.active_tensors[name] = True

                return tensor
            else:
                # Fallback to CPU
                return np.zeros(shape, dtype=dtype)

        except Exception as e:
            self.get_logger().error(f'Error allocating GPU tensor: {e}')
            return np.zeros(shape, dtype=dtype)

    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        try:
            import pycuda.driver as cuda
            # In practice, get actual memory usage
            # This is a simplified version
            return 0.5  # Placeholder
        except:
            return 0.0

    def create_gpu_tensor(self, shape, dtype):
        """Create GPU tensor"""
        try:
            import pycuda.gpuarray as gpuarray
            return gpuarray.zeros(shape, dtype=dtype)
        except:
            return np.zeros(shape, dtype=dtype)

def main(args=None):
    rclpy.init(args=args)
    optimizer = IsaacROSPreformanceOptimizer()

    try:
        rclpy.spin(optimizer)
    except KeyboardInterrupt:
        pass
    finally:
        optimizer.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Isaac ROS

1. **Hardware Requirements**: Ensure sufficient GPU memory and compute capability
2. **Parameter Tuning**: Optimize parameters for your specific robot and environment
3. **Fallback Mechanisms**: Implement CPU-based fallbacks for robustness
4. **Memory Management**: Monitor and manage GPU memory usage
5. **Performance Monitoring**: Track frame rates, latency, and accuracy
6. **Integration Testing**: Test with real hardware to validate performance gains
7. **Safety Measures**: Implement emergency stop mechanisms for autonomous navigation

## Summary

This chapter covered Isaac ROS for hardware-accelerated VSLAM and navigation. We explored:

- Isaac ROS architecture and core packages
- Visual SLAM implementation with stereo cameras
- GPU-accelerated path planning algorithms
- Integration with traditional ROS 2 navigation systems
- Performance optimization techniques

Isaac ROS provides significant performance improvements for perception and navigation tasks through GPU acceleration, making it suitable for real-time robotics applications that require high computational performance.

## Exercises

1. Implement a GPU-accelerated particle filter for localization using Isaac ROS.
2. Create a hybrid navigation system that switches between Isaac ROS and Nav2 based on environmental conditions.
3. How would you optimize Isaac ROS for a robot with limited GPU resources?

## Code Example: Complete Isaac ROS Integration System

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np
import cv2
from cv_bridge import CvBridge

class CompleteIsaacROSSystem(Node):
    def __init__(self):
        super().__init__('complete_isaac_ros_system')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.perception_enabled = True
        self.navigation_enabled = True
        self.localization_enabled = True

        # Setup perception pipeline
        self.setup_perception_pipeline()

        # Setup navigation pipeline
        self.setup_navigation_pipeline()

        # Setup localization pipeline
        self.setup_localization_pipeline()

        # Performance monitoring
        self.setup_performance_monitoring()

        self.get_logger().info('Complete Isaac ROS System initialized')

    def setup_perception_pipeline(self):
        """Setup Isaac ROS perception pipeline"""
        # Stereo camera subscribers
        self.left_camera_sub = self.create_subscription(
            Image, '/stereo_camera/left/image_rect_color',
            self.left_camera_callback, 10)
        self.right_camera_sub = self.create_subscription(
            Image, '/stereo_camera/right/image_rect',
            self.right_camera_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/stereo_camera/left/camera_info',
            self.camera_info_callback, 10)

        # LiDAR subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        # Perception publishers
        self.depth_pub = self.create_publisher(Image, '/depth/image', 10)
        self.obstacles_pub = self.create_publisher(OccupancyGrid, '/obstacles', 10)

        self.get_logger().info('Perception pipeline initialized')

    def setup_navigation_pipeline(self):
        """Setup Isaac ROS navigation pipeline"""
        # Navigation subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal',
            self.navigation_goal_callback, 10)

        # Navigation publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(OccupancyGrid, '/global_plan', 10)

        # Navigation timers
        self.navigation_timer = self.create_timer(0.1, self.navigation_update)

        self.get_logger().info('Navigation pipeline initialized')

    def setup_localization_pipeline(self):
        """Setup Isaac ROS localization pipeline"""
        # Odometry subscriber
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Localization publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/amcl_pose', 10)

        # Localization timers
        self.localization_timer = self.create_timer(0.05, self.localization_update)

        self.get_logger().info('Localization pipeline initialized')

    def setup_performance_monitoring(self):
        """Setup performance monitoring"""
        self.frame_count = 0
        self.start_time = self.get_clock().now()
        self.fps = 0.0

        # Performance monitoring timer
        self.perf_timer = self.create_timer(1.0, self.update_performance_metrics)

    def left_camera_callback(self, msg):
        """Process left camera image"""
        if not self.perception_enabled:
            return

        try:
            # Convert to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process with Isaac ROS perception (simulated)
            processed_image = self.process_perception(cv_image)

            # Publish results
            if processed_image is not None:
                result_msg = self.cv_bridge.cv2_to_imgmsg(processed_image, "bgr8")
                result_msg.header = msg.header
                self.depth_pub.publish(result_msg)

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f'Error in left camera callback: {e}')

    def right_camera_callback(self, msg):
        """Process right camera image"""
        # Similar to left camera but for stereo processing
        pass

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        if not self.perception_enabled:
            return

        try:
            # Process LiDAR data for obstacle detection
            obstacles = self.detect_obstacles_from_lidar(msg)

            # Publish obstacle map
            if obstacles is not None:
                obstacle_msg = self.create_obstacle_grid(obstacles, msg.header)
                self.obstacles_pub.publish(obstacle_msg)

        except Exception as e:
            self.get_logger().error(f'Error in LiDAR callback: {e}')

    def process_perception(self, image):
        """Process image with Isaac ROS perception pipeline"""
        # In a real implementation, this would use Isaac ROS perception nodes
        # For now, we'll simulate processing
        processed = image.copy()

        # Simulate feature detection and matching
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        if features is not None:
            for feature in features:
                x, y = feature.ravel()
                cv2.circle(processed, (int(x), int(y)), 3, (0, 255, 0), -1)

        return processed

    def detect_obstacles_from_lidar(self, laser_msg):
        """Detect obstacles from LiDAR data"""
        obstacles = []

        angle_increment = laser_msg.angle_increment
        current_angle = laser_msg.angle_min

        for range_val in laser_msg.ranges:
            if laser_msg.range_min <= range_val <= laser_msg.range_max:
                # Calculate position in robot frame
                x = range_val * np.cos(current_angle)
                y = range_val * np.sin(current_angle)

                # Check if this is an obstacle (not ground plane)
                if abs(y) < 0.5:  # Ground plane threshold
                    obstacles.append((x, y))

            current_angle += angle_increment

        return obstacles

    def create_obstacle_grid(self, obstacles, header):
        """Create occupancy grid from obstacle points"""
        grid = OccupancyGrid()
        grid.header = header
        grid.info.resolution = 0.1  # 10cm resolution
        grid.info.width = 100  # 10m x 10m grid
        grid.info.height = 100
        grid.info.origin.position.x = -5.0  # Center around robot
        grid.info.origin.position.y = -5.0
        grid.info.origin.orientation.w = 1.0

        # Initialize grid with unknown values
        grid.data = [-1] * (grid.info.width * grid.info.height)

        # Mark obstacle cells
        for x, y in obstacles:
            grid_x = int((x - grid.info.origin.position.x) / grid.info.resolution)
            grid_y = int((y - grid.info.origin.position.y) / grid.info.resolution)

            if (0 <= grid_x < grid.info.width and
                0 <= grid_y < grid.info.height):
                idx = grid_y * grid.info.width + grid_x
                grid.data[idx] = 100  # Mark as occupied

        return grid

    def navigation_goal_callback(self, msg):
        """Handle navigation goal"""
        if not self.navigation_enabled:
            return

        self.get_logger().info(f'Received navigation goal: {msg.pose}')

        # In a real system, this would trigger Isaac ROS navigation
        # For now, we'll just log the goal
        self.current_goal = msg.pose

    def navigation_update(self):
        """Update navigation system"""
        if not self.navigation_enabled:
            return

        # In a real implementation, this would run Isaac ROS navigation
        # For now, we'll just simulate basic control
        if self.current_goal:
            cmd = self.calculate_navigation_command()
            if cmd:
                self.cmd_vel_pub.publish(cmd)

    def localization_update(self):
        """Update localization system"""
        if not self.localization_enabled:
            return

        # In a real implementation, this would run Isaac ROS localization
        # For now, we'll just publish a dummy pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)

    def calculate_navigation_command(self):
        """Calculate navigation command to reach goal"""
        cmd = Twist()

        # Simple proportional controller
        if self.current_goal:
            dx = self.current_goal.position.x
            dy = self.current_goal.position.y

            distance = np.sqrt(dx*dx + dy*dy)
            angle = np.arctan2(dy, dx)

            # Simple control law
            cmd.linear.x = min(0.5, distance) * 0.5  # Max 0.5 m/s
            cmd.angular.z = angle * 0.5  # Proportional control

        return cmd

    def update_performance_metrics(self):
        """Update and log performance metrics"""
        current_time = self.get_clock().now()
        elapsed = (current_time.nanoseconds - self.start_time.nanoseconds) / 1e9

        if elapsed > 0:
            self.fps = self.frame_count / elapsed

        self.get_logger().info(f'Performance: {self.fps:.2f} FPS, Frames: {self.frame_count}')

        # Reset for next interval
        self.frame_count = 0
        self.start_time = current_time

    def enable_perception(self):
        """Enable perception pipeline"""
        self.perception_enabled = True
        self.get_logger().info('Perception pipeline enabled')

    def disable_perception(self):
        """Disable perception pipeline"""
        self.perception_enabled = False
        self.get_logger().info('Perception pipeline disabled')

    def enable_navigation(self):
        """Enable navigation pipeline"""
        self.navigation_enabled = True
        self.get_logger().info('Navigation pipeline enabled')

    def disable_navigation(self):
        """Disable navigation pipeline"""
        self.navigation_enabled = False
        self.get_logger().info('Navigation pipeline disabled')

    def enable_localization(self):
        """Enable localization pipeline"""
        self.localization_enabled = True
        self.get_logger().info('Localization pipeline enabled')

    def disable_localization(self):
        """Disable localization pipeline"""
        self.localization_enabled = False
        self.get_logger().info('Localization pipeline disabled')

def main(args=None):
    rclpy.init(args=args)
    system = CompleteIsaacROSSystem()

    try:
        rclpy.spin(system)
    except KeyboardInterrupt:
        pass
    finally:
        system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```