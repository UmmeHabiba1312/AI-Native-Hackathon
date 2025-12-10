---
title: Sensor Simulation: LiDAR, IMU, Depth Cameras
description: Understanding sensor simulation in Gazebo and Unity for digital twins
sidebar_position: 2
---

# Sensor Simulation: LiDAR, IMU, Depth Cameras

## Learning Objectives

By the end of this chapter, you will be able to:
1. Implement realistic sensor simulation for LiDAR, IMU, and depth cameras
2. Configure sensor parameters for different applications
3. Integrate simulated sensors with ROS 2
4. Generate synthetic sensor data for training and testing

## Introduction to Sensor Simulation

Sensor simulation is a critical component of digital twin technology, enabling the creation of realistic synthetic data for training, testing, and validation of robotics systems. Accurate sensor simulation allows developers to:

- Test algorithms in controlled environments
- Generate large amounts of training data
- Validate perception systems before deployment
- Reduce costs and risks associated with real-world testing

## LiDAR Simulation

LiDAR (Light Detection and Ranging) sensors are crucial for robotics applications, providing accurate 3D spatial information about the environment.

### Gazebo LiDAR Implementation

In Gazebo, LiDAR sensors can be implemented using the libgazebo_ros_ray.so plugin:

```xml
<sensor name="lidar_sensor" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -PI -->
        <max_angle>3.14159</max_angle>   <!-- PI -->
      </horizontal>
      <vertical>
        <samples>1</samples>
        <resolution>1</resolution>
        <min_angle>0</min_angle>
        <max_angle>0</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_plugin" filename="libgazebo_ros_ray.so">
    <ros>
      <namespace>/lidar</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

### Advanced LiDAR Configuration

For more complex LiDAR systems like multi-beam sensors:

```xml
<sensor name="velodyne_sensor" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>1800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.2</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="velodyne_plugin" filename="libgazebo_ros_velodyne_gpu.so">
    <ros>
      <namespace>/velodyne</namespace>
      <remapping>~/out:=points</remapping>
    </ros>
    <min_intensity>0.1</min_intensity>
    <gpu>True</gpu>
  </plugin>
</sensor>
```

### LiDAR Data Processing in ROS 2

Processing LiDAR data in ROS 2:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header
import numpy as np
from scipy.spatial import KDTree

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribe to LiDAR data
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/lidar/scan',
            self.lidar_callback,
            10)

        # Publisher for processed data
        self.obstacle_publisher = self.create_publisher(
            PointCloud2,
            '/lidar/obstacles',
            10)

        # Parameters
        self.min_obstacle_distance = 0.5  # meters
        self.obstacle_threshold = 0.8     # confidence threshold

        self.get_logger().info('LiDAR Processor initialized')

    def lidar_callback(self, msg):
        """Process incoming LiDAR scan data"""
        try:
            # Convert polar coordinates to Cartesian
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            ranges = np.array(msg.ranges)

            # Filter out invalid ranges
            valid_indices = (ranges >= msg.range_min) & (ranges <= msg.range_max)
            valid_ranges = ranges[valid_indices]
            valid_angles = angles[valid_indices]

            # Convert to Cartesian coordinates
            x_coords = valid_ranges * np.cos(valid_angles)
            y_coords = valid_ranges * np.sin(valid_angles)

            # Detect obstacles
            obstacles = self.detect_obstacles(x_coords, y_coords, msg.header)

            # Publish obstacle data
            if obstacles.size > 0:
                obstacle_msg = self.create_pointcloud2(obstacles, msg.header)
                self.obstacle_publisher.publish(obstacle_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def detect_obstacles(self, x_coords, y_coords, header):
        """Detect obstacles from LiDAR data"""
        # Simple clustering-based obstacle detection
        points = np.column_stack((x_coords, y_coords))

        if len(points) == 0:
            return np.array([])

        # Use KDTree for efficient nearest neighbor search
        tree = KDTree(points)

        # Find clusters of points (potential obstacles)
        visited = set()
        obstacles = []

        for i, point in enumerate(points):
            if i in visited:
                continue

            # Find neighbors within a certain distance
            neighbors = tree.query_ball_point(point, r=0.3)  # 30cm clustering radius

            if len(neighbors) > 3:  # Minimum 3 points to form an obstacle
                # Calculate centroid of cluster
                cluster_points = points[neighbors]
                centroid = np.mean(cluster_points, axis=0)

                # Calculate cluster density
                cluster_size = len(neighbors)

                # Only consider as obstacle if within min distance
                distance_to_robot = np.linalg.norm(centroid)
                if distance_to_robot < self.min_obstacle_distance:
                    obstacles.append(centroid)

                visited.update(neighbors)

        return np.array(obstacles)

    def create_pointcloud2(self, points, header):
        """Create PointCloud2 message from numpy array"""
        # This is a simplified implementation
        # In practice, use sensor_msgs_py.point_cloud2.create_cloud_xyz32
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
            # Add z=0 for 2D LiDAR data
            packed = struct.pack('fff', point[0], point[1], 0.0)
            data.append(packed)

        cloud_msg.data = b''.join(data)
        return cloud_msg

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_processor.destroy_node()
        rclpy.shutdown()
```

## IMU Simulation

An Inertial Measurement Unit (IMU) provides measurements of linear acceleration and angular velocity, crucial for robot localization and control.

### Gazebo IMU Implementation

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.017</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.017</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.0</bias_mean>
          <bias_stddev>0.017</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/imu</namespace>
      <remapping>~/out:=data</remapping>
    </ros>
    <initial_orientation_as_reference>false</initial_orientation_as_reference>
  </plugin>
</sensor>
```

### IMU Data Processing and Filtering

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU data
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)

        # Publishers for processed data
        self.orientation_publisher = self.create_publisher(
            Imu,
            '/imu/orientation_filtered',
            10)

        self.linear_velocity_publisher = self.create_publisher(
            Vector3,
            '/imu/linear_velocity',
            10)

        # Complementary filter parameters
        self.complementary_filter_alpha = 0.98
        self.gravity = 9.81  # m/s^2

        # Initialize state
        self.previous_linear_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        self.integration_time = self.get_clock().now()

        # Orientation estimate (initially identity)
        self.orientation_estimate = R.identity()

        self.get_logger().info('IMU Processor initialized')

    def imu_callback(self, msg):
        """Process incoming IMU data"""
        try:
            # Extract measurements
            angular_velocity = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            linear_acceleration = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Update timestamp
            current_time = rclpy.time.Time.from_msg(msg.header.stamp)
            dt = (current_time.nanoseconds - self.integration_time.nanoseconds) / 1e9
            self.integration_time = current_time

            if dt <= 0:
                return  # Invalid time difference

            # Estimate orientation using gyroscope integration
            estimated_orientation = self.integrate_gyro(angular_velocity, dt)

            # Get orientation from accelerometer (pitch and roll only)
            accel_orientation = self.accelerometer_orientation(linear_acceleration)

            # Complementary filter to combine both estimates
            filtered_orientation = self.complementary_filter(
                estimated_orientation, accel_orientation)

            # Calculate linear velocity by integrating acceleration
            linear_velocity = self.integrate_acceleration(
                linear_acceleration, filtered_orientation, dt)

            # Publish results
            self.publish_results(filtered_orientation, linear_velocity, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def integrate_gyro(self, angular_velocity, dt):
        """Integrate gyroscope data to estimate orientation"""
        # Convert angular velocity to rotation vector
        rotation_vector = angular_velocity * dt

        # Convert to quaternion
        rotation = R.from_rotvec(rotation_vector)

        # Update orientation estimate
        self.orientation_estimate = self.orientation_estimate * rotation

        return self.orientation_estimate

    def accelerometer_orientation(self, linear_acceleration):
        """Estimate pitch and roll from accelerometer data"""
        ax, ay, az = linear_acceleration

        # Calculate pitch and roll from accelerometer
        pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        roll = math.atan2(ay, az)

        # Create rotation object with only pitch and roll
        # yaw remains unchanged from gyroscope integration
        current_euler = self.orientation_estimate.as_euler('xyz')
        corrected_euler = [roll, pitch, current_euler[2]]  # Keep integrated yaw

        return R.from_euler('xyz', corrected_euler)

    def complementary_filter(self, gyro_estimate, accel_estimate):
        """Apply complementary filter to combine estimates"""
        # Convert to Euler angles for filtering
        gyro_euler = gyro_estimate.as_euler('xyz')
        accel_euler = accel_estimate.as_euler('xyz')

        # Apply complementary filter
        # Use gyroscope for yaw (no gravity reference)
        # Use accelerometer for pitch and roll with gyroscope for short-term stability
        filtered_euler = [
            self.complementary_filter_alpha * gyro_euler[0] + (1 - self.complementary_filter_alpha) * accel_euler[0],  # roll
            self.complementary_filter_alpha * gyro_euler[1] + (1 - self.complementary_filter_alpha) * accel_euler[1],  # pitch
            gyro_euler[2]  # yaw (from gyroscope only)
        ]

        # Convert back to rotation object
        self.orientation_estimate = R.from_euler('xyz', filtered_euler)
        return self.orientation_estimate

    def integrate_acceleration(self, linear_acceleration, orientation, dt):
        """Integrate linear acceleration to get velocity"""
        # Transform acceleration from body frame to world frame
        rotation_matrix = orientation.as_matrix()
        world_frame_accel = rotation_matrix @ linear_acceleration

        # Subtract gravity (assuming z-axis is up)
        world_frame_accel[2] -= self.gravity

        # Integrate acceleration to get velocity
        delta_velocity = world_frame_accel * dt
        self.previous_linear_velocity.x += delta_velocity[0]
        self.previous_linear_velocity.y += delta_velocity[1]
        self.previous_linear_velocity.z += delta_velocity[2]

        return self.previous_linear_velocity

    def publish_results(self, orientation, linear_velocity, header):
        """Publish processed IMU results"""
        # Publish filtered orientation
        filtered_imu = Imu()
        filtered_imu.header = header
        quat = orientation.as_quat()
        filtered_imu.orientation.x = quat[0]
        filtered_imu.orientation.y = quat[1]
        filtered_imu.orientation.z = quat[2]
        filtered_imu.orientation.w = quat[3]

        # Copy angular velocity and linear acceleration
        # (these are the raw measurements, not processed)
        filtered_imu.angular_velocity = Vector3()  # Raw values would come from original msg
        filtered_imu.linear_acceleration = Vector3()  # Raw values would come from original msg

        self.orientation_publisher.publish(filtered_imu)

        # Publish linear velocity
        self.linear_velocity_publisher.publish(linear_velocity)

def main(args=None):
    rclpy.init(args=args)
    imu_processor = IMUProcessor()

    try:
        rclpy.spin(imu_processor)
    except KeyboardInterrupt:
        pass
    finally:
        imu_processor.destroy_node()
        rclpy.shutdown()
```

## Depth Camera Simulation

Depth cameras provide both color and depth information, essential for 3D scene understanding and navigation.

### Gazebo Depth Camera Implementation

```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <ros>
      <namespace>/camera</namespace>
      <remapping>~/rgb/image_raw:=image_raw</remapping>
      <remapping>~/depth/image_raw:=depth/image_raw</remapping>
      <remapping>~/depth/camera_info:=depth/camera_info</remapping>
    </ros>
    <update_rate>30.0</update_rate>
    <baseline>0.2</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <point_cloud_cutoff>0.1</point_cloud_cutoff>
    <point_cloud_cutoff_max>3.0</point_cloud_cutoff_max>
    <frame_name>camera_depth_optical_frame</frame_name>
  </plugin>
</sensor>
```

### Depth Camera Data Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
import struct

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')

        # Create CV bridge
        self.cv_bridge = CvBridge()

        # Subscribe to depth camera data
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10)

        # Publishers
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2,
            '/camera/pointcloud',
            10)

        self.obstacle_publisher = self.create_publisher(
            PointStamped,
            '/camera/closest_obstacle',
            10)

        # State variables
        self.camera_matrix = None
        self.latest_depth_image = None
        self.latest_rgb_image = None

        self.get_logger().info('Depth Camera Processor initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        """Process RGB image data"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_rgb_image = cv_image

            # Process image for object detection (simplified)
            processed_image = self.process_rgb_image(cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth image data"""
        try:
            # Convert ROS Image to OpenCV
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
            self.latest_depth_image = depth_image

            # Process depth data
            if self.camera_matrix is not None:
                pointcloud = self.depth_to_pointcloud(depth_image, self.camera_matrix, msg.header)
                if pointcloud is not None:
                    self.pointcloud_publisher.publish(pointcloud)

                # Find closest obstacle
                closest_point = self.find_closest_obstacle(depth_image)
                if closest_point is not None:
                    point_msg = PointStamped()
                    point_msg.header = msg.header
                    point_msg.point.x = closest_point[0]
                    point_msg.point.y = closest_point[1]
                    point_msg.point.z = closest_point[2]
                    self.obstacle_publisher.publish(point_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def process_rgb_image(self, image):
        """Process RGB image for object detection"""
        # Simple example: detect colors in certain ranges
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for a specific color (e.g., blue)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on image
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    def depth_to_pointcloud(self, depth_image, camera_matrix, header):
        """Convert depth image to point cloud"""
        if depth_image is None or camera_matrix is None:
            return None

        height, width = depth_image.shape

        # Get camera intrinsic parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Generate coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Convert pixel coordinates to camera coordinates
        z_cam = depth_image
        x_cam = (u_coords - cx) * z_cam / fx
        y_cam = (v_coords - cy) * z_cam / fy

        # Flatten arrays
        x_flat = x_cam.flatten()
        y_flat = y_cam.flatten()
        z_flat = z_cam.flatten()

        # Remove invalid points (where depth is 0 or infinity)
        valid_mask = (z_flat > 0) & (z_flat < np.inf)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]

        # Create PointCloud2 message
        points = np.column_stack((x_valid, y_valid, z_valid)).astype(np.float32)

        if len(points) == 0:
            return None

        return self.create_pointcloud2(points, header)

    def find_closest_obstacle(self, depth_image):
        """Find the closest obstacle in the depth image"""
        if depth_image is None:
            return None

        # Find minimum depth value (closest point)
        min_depth = np.min(depth_image[depth_image > 0])  # Ignore invalid depths

        if min_depth == np.inf or min_depth <= 0:
            return None

        # Find coordinates of closest point
        min_coords = np.unravel_index(np.argmin(depth_image, axis=None), depth_image.shape)
        v, u = min_coords

        # Convert to 3D coordinates using camera matrix
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            x = (u - cx) * min_depth / fx
            y = (v - cy) * min_depth / fy
            z = min_depth

            return [x, y, z]

        return [0, 0, min_depth]  # Simplified if no camera matrix

    def create_pointcloud2(self, points, header):
        """Create PointCloud2 message from numpy array"""
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = points.shape[0]
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
    depth_processor = DepthCameraProcessor()

    try:
        rclpy.spin(depth_processor)
    except KeyboardInterrupt:
        pass
    finally:
        depth_processor.destroy_node()
        rclpy.shutdown()
```

## Unity Sensor Simulation

Unity provides different approaches for sensor simulation, particularly for more advanced graphics and physics:

### LiDAR Simulation in Unity

```csharp
using UnityEngine;
using System.Collections.Generic;

public class UnityLidarSimulation : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int resolution = 360;  // Number of rays
    public float minAngle = -90f; // Minimum angle in degrees
    public float maxAngle = 90f;  // Maximum angle in degrees
    public float maxDistance = 30f; // Maximum detection distance
    public LayerMask detectionMask = -1; // Layers to detect

    [Header("Visualization")]
    public bool visualizeRays = true;
    public GameObject rayVisualizationPrefab;

    private float[] ranges;
    private List<GameObject> visualizations;

    void Start()
    {
        ranges = new float[resolution];
        visualizations = new List<GameObject>();

        // Initialize visualization objects if needed
        if (visualizeRays && rayVisualizationPrefab != null)
        {
            for (int i = 0; i < resolution; i++)
            {
                GameObject viz = Instantiate(rayVisualizationPrefab);
                viz.SetActive(false);
                visualizations.Add(viz);
            }
        }
    }

    void Update()
    {
        SimulateLidar();
    }

    void SimulateLidar()
    {
        float angleStep = (maxAngle - minAngle) / resolution;

        for (int i = 0; i < resolution; i++)
        {
            float currentAngle = minAngle + i * angleStep;
            float angleRad = currentAngle * Mathf.Deg2Rad;

            Vector3 direction = new Vector3(
                Mathf.Cos(angleRad),
                0f,
                Mathf.Sin(angleRad)
            );

            // Rotate direction according to lidar's forward direction
            direction = transform.TransformDirection(direction);

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxDistance, detectionMask))
            {
                ranges[i] = hit.distance;

                // Update visualization
                if (visualizeRays && i < visualizations.Count)
                {
                    GameObject viz = visualizations[i];
                    viz.SetActive(true);
                    viz.transform.position = transform.position + direction * hit.distance;
                    viz.transform.LookAt(transform);
                }
            }
            else
            {
                ranges[i] = float.MaxValue; // No hit

                // Hide visualization
                if (visualizeRays && i < visualizations.Count)
                {
                    visualizations[i].SetActive(false);
                }
            }
        }

        // Publish ranges to ROS (via ROS# or Unity Robotics package)
        PublishLidarData();
    }

    void PublishLidarData()
    {
        // This would typically interface with ROS via Unity Robotics packages
        // For now, just log the data
        Debug.Log($"LiDAR Range[0]: {ranges[0]}, Range[{resolution-1}]: {ranges[ranges.Length-1]}");
    }

    public float[] GetRanges()
    {
        return ranges;
    }

    public float GetRange(int index)
    {
        if (index >= 0 && index < ranges.Length)
        {
            return ranges[index];
        }
        return float.MaxValue;
    }
}
```

### IMU Simulation in Unity

```csharp
using UnityEngine;

public class UnityIMUModule : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float noiseLevel = 0.01f;
    public float biasLevel = 0.001f;

    [Header("Gravity Compensation")]
    public bool compensateGravity = true;

    private Vector3 lastPosition;
    private Vector3 lastVelocity;
    private float lastTime;

    // Simulated IMU readings
    private Vector3 simulatedAcceleration;
    private Vector3 simulatedAngularVelocity;
    private Quaternion simulatedOrientation;

    void Start()
    {
        lastPosition = transform.position;
        lastTime = Time.time;
        simulatedOrientation = transform.rotation;
    }

    void FixedUpdate()
    {
        SimulateIMU();
    }

    void SimulateIMU()
    {
        float deltaTime = Time.fixedDeltaTime;

        // Get current state
        Vector3 currentPosition = transform.position;
        Quaternion currentRotation = transform.rotation;

        // Calculate linear acceleration from position changes
        Vector3 currentVelocity = (currentPosition - lastPosition) / deltaTime;
        Vector3 linearAcceleration = (currentVelocity - lastVelocity) / deltaTime;

        // Add noise and bias to acceleration
        Vector3 noiseAcc = new Vector3(
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel)
        );

        Vector3 biasAcc = new Vector3(
            Random.Range(-biasLevel, biasLevel),
            Random.Range(-biasLevel, biasLevel),
            Random.Range(-biasLevel, biasLevel)
        );

        // Apply gravity compensation if enabled
        if (compensateGravity)
        {
            linearAcceleration -= Physics.gravity;
        }

        simulatedAcceleration = linearAcceleration + noiseAcc + biasAcc;

        // Simulate angular velocity (simplified)
        Quaternion deltaRotation = currentRotation * Quaternion.Inverse(simulatedOrientation);
        Vector3 angularVelocity = new Vector3(
            deltaRotation.eulerAngles.x * Mathf.Deg2Rad / deltaTime,
            deltaRotation.eulerAngles.y * Mathf.Deg2Rad / deltaTime,
            deltaRotation.eulerAngles.z * Mathf.Deg2Rad / deltaTime
        );

        // Add noise and bias to angular velocity
        Vector3 noiseAngVel = new Vector3(
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel),
            Random.Range(-noiseLevel, noiseLevel)
        );

        Vector3 biasAngVel = new Vector3(
            Random.Range(-biasLevel, biasLevel),
            Random.Range(-biasLevel, biasLevel),
            Random.Range(-biasLevel, biasLevel)
        );

        simulatedAngularVelocity = angularVelocity + noiseAngVel + biasAngVel;

        // Update orientation
        simulatedOrientation = currentRotation;

        // Store for next frame
        lastPosition = currentPosition;
        lastVelocity = currentVelocity;

        // Publish IMU data
        PublishIMUData();
    }

    void PublishIMUData()
    {
        // Log the simulated data
        Debug.Log($"IMU - Acc: {simulatedAcceleration}, AngVel: {simulatedAngularVelocity}");
    }

    public Vector3 GetLinearAcceleration()
    {
        return simulatedAcceleration;
    }

    public Vector3 GetAngularVelocity()
    {
        return simulatedAngularVelocity;
    }

    public Quaternion GetOrientation()
    {
        return simulatedOrientation;
    }
}
```

## Synthetic Data Generation

Generating synthetic sensor data is crucial for training perception systems:

### Synthetic LiDAR Data Generator

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random

class SyntheticLidarGenerator:
    def __init__(self, num_beams=360, max_range=30.0, angular_resolution=1.0):
        self.num_beams = num_beams
        self.max_range = max_range
        self.angular_resolution = angular_resolution

        # Precompute angles for efficiency
        self.angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)

    def generate_static_scene(self, objects, robot_position=(0, 0)):
        """
        Generate LiDAR scan for a static scene with objects

        Args:
            objects: List of objects in the scene (circles with center and radius)
            robot_position: (x, y) position of the robot

        Returns:
            ranges: Array of range measurements
        """
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)

        robot_x, robot_y = robot_position

        for beam_idx, angle in enumerate(self.angles):
            ray_x = np.cos(angle)
            ray_y = np.sin(angle)

            min_distance = self.max_range

            # Check intersection with each object
            for obj in objects:
                obj_type = obj['type']

                if obj_type == 'circle':
                    # Circle intersection
                    center_x, center_y = obj['center']
                    radius = obj['radius']

                    # Vector from robot to circle center
                    to_center_x = center_x - robot_x
                    to_center_y = center_y - robot_y

                    # Distance from ray to circle center
                    ray_to_center = to_center_x * ray_x + to_center_y * ray_y

                    # Closest point on ray to circle center
                    closest_x = robot_x + ray_to_center * ray_x
                    closest_y = robot_y + ray_to_center * ray_y

                    # Distance from closest point to circle center
                    dist_to_center = np.sqrt(
                        (closest_x - center_x)**2 + (closest_y - center_y)**2
                    )

                    if dist_to_center <= radius:
                        # Calculate intersection points
                        half_chord_len = np.sqrt(radius**2 - dist_to_center**2)

                        # Distances along ray to intersection points
                        d1 = ray_to_center - half_chord_len
                        d2 = ray_to_center + half_chord_len

                        # Take the closer positive intersection
                        if d1 > 0 and d1 < min_distance:
                            min_distance = d1
                        elif d2 > 0 and d2 < min_distance:
                            min_distance = d2

                elif obj_type == 'rectangle':
                    # Rectangle intersection (simplified)
                    center_x, center_y = obj['center']
                    width, height = obj['dimensions']

                    # Calculate intersection with rectangle edges
                    half_w, half_h = width/2, height/2
                    rect_left = center_x - half_w
                    rect_right = center_x + half_w
                    rect_bottom = center_y - half_h
                    rect_top = center_y + half_h

                    # Check intersection with each edge
                    t_values = []

                    # Left edge (x = rect_left)
                    if ray_x != 0:
                        t = (rect_left - robot_x) / ray_x
                        if t > 0:
                            y_intersect = robot_y + t * ray_y
                            if rect_bottom <= y_intersect <= rect_top:
                                t_values.append(t)

                    # Right edge (x = rect_right)
                    if ray_x != 0:
                        t = (rect_right - robot_x) / ray_x
                        if t > 0:
                            y_intersect = robot_y + t * ray_y
                            if rect_bottom <= y_intersect <= rect_top:
                                t_values.append(t)

                    # Bottom edge (y = rect_bottom)
                    if ray_y != 0:
                        t = (rect_bottom - robot_y) / ray_y
                        if t > 0:
                            x_intersect = robot_x + t * ray_x
                            if rect_left <= x_intersect <= rect_right:
                                t_values.append(t)

                    # Top edge (y = rect_top)
                    if ray_y != 0:
                        t = (rect_top - robot_y) / ray_y
                        if t > 0:
                            x_intersect = robot_x + t * ray_x
                            if rect_left <= x_intersect <= rect_right:
                                t_values.append(t)

                    if t_values:
                        min_t = min(t_values)
                        if min_t > 0 and min_t < min_distance:
                            min_distance = min_t

            ranges[beam_idx] = min(min_distance, self.max_range)

        return ranges

    def add_noise(self, ranges, noise_std=0.02, dropout_prob=0.01):
        """Add realistic noise to LiDAR ranges"""
        noisy_ranges = ranges.copy().astype(np.float32)

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, size=ranges.shape)
        noisy_ranges += noise

        # Apply range limits
        noisy_ranges = np.clip(noisy_ranges, 0, self.max_range)

        # Random dropout (simulates sensor failures or reflective surfaces)
        dropout_mask = np.random.random(size=ranges.shape) < dropout_prob
        noisy_ranges[dropout_mask] = self.max_range  # Set to max range for dropout

        return noisy_ranges

    def generate_dynamic_scene(self, static_objects, moving_objects, robot_position=(0, 0)):
        """
        Generate LiDAR scan for a scene with both static and moving objects
        """
        # Combine static and moving objects
        all_objects = static_objects + moving_objects

        ranges = self.generate_static_scene(all_objects, robot_position)
        return self.add_noise(ranges)

# Example usage
def example_synthetic_lidar():
    generator = SyntheticLidarGenerator(num_beams=720, max_range=20.0)

    # Define static objects in the scene
    static_objects = [
        {'type': 'circle', 'center': (3, 2), 'radius': 0.5},
        {'type': 'circle', 'center': (-2, 4), 'radius': 0.8},
        {'type': 'rectangle', 'center': (0, -3), 'dimensions': (4, 1)},
        {'type': 'rectangle', 'center': (5, 0), 'dimensions': (1, 3)}
    ]

    # Generate LiDAR scan
    ranges = generator.generate_static_scene(static_objects, robot_position=(0, 0))
    noisy_ranges = generator.add_noise(ranges)

    # Plot results
    plt.figure(figsize=(12, 5))

    # Polar plot
    plt.subplot(1, 2, 1, projection='polar')
    plt.plot(generator.angles, noisy_ranges, 'b-', alpha=0.7)
    plt.title('Synthetic LiDAR Scan')
    plt.grid(True)

    # Cartesian plot
    plt.subplot(1, 2, 2)
    x_coords = noisy_ranges * np.cos(generator.angles)
    y_coords = noisy_ranges * np.sin(generator.angles)

    # Filter out max range values for better visualization
    valid_mask = noisy_ranges < generator.max_range * 0.9
    plt.scatter(x_coords[valid_mask], y_coords[valid_mask], s=1, c='blue', alpha=0.6)

    # Plot objects
    for obj in static_objects:
        if obj['type'] == 'circle':
            circle = plt.Circle(obj['center'], obj['radius'], fill=False, color='red', linewidth=2)
            plt.gca().add_patch(circle)
        elif obj['type'] == 'rectangle':
            center_x, center_y = obj['center']
            width, height = obj['dimensions']
            rect = plt.Rectangle(
                (center_x - width/2, center_y - height/2),
                width, height,
                fill=False, color='green', linewidth=2
            )
            plt.gca().add_patch(rect)

    plt.axis('equal')
    plt.grid(True)
    plt.title('Cartesian View')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    plt.tight_layout()
    plt.show()

    return noisy_ranges

# Run example
# synthetic_scan = example_synthetic_lidar()
```

## Best Practices for Sensor Simulation

1. **Realistic Noise Modeling**: Include appropriate noise models that match real sensors
2. **Calibration**: Ensure simulated sensors are properly calibrated
3. **Validation**: Compare simulated data with real sensor data when possible
4. **Computational Efficiency**: Balance realism with simulation speed
5. **Domain Randomization**: Vary parameters to increase robustness of trained models

## Summary

This chapter covered sensor simulation for LiDAR, IMU, and depth cameras in both Gazebo and Unity environments. We explored how to implement realistic sensor models, process the data in ROS 2, and generate synthetic data for training and testing. Proper sensor simulation is crucial for developing robust perception and navigation systems.

## Exercises

1. Implement a thermal camera simulation in Gazebo.
2. Create a synthetic dataset generator for stereo vision.
3. How would you simulate sensor failures and degraded performance?

## Code Example: Multi-Sensor Fusion Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import threading

class MultiSensorFusionNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Initialize state variables
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.robot_velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, omega
        self.pose_covariance = np.eye(3) * 0.1  # Initial covariance

        # Buffers for sensor data synchronization
        self.lidar_buffer = deque(maxlen=5)
        self.imu_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)

        # Lock for thread safety
        self.state_lock = threading.Lock()

        # Subscribe to sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, '/lidar/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        # Publisher for fused pose
        self.pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10)

        # Timer for fusion update
        self.fusion_timer = self.create_timer(0.05, self.fusion_callback)  # 20 Hz

        self.get_logger().info('Multi-Sensor Fusion Node initialized')

    def lidar_callback(self, msg):
        """Process LiDAR data for environment mapping and localization"""
        with self.state_lock:
            # Store LiDAR data with timestamp for synchronization
            lidar_data = {
                'ranges': np.array(msg.ranges),
                'angles': np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges)),
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
            self.lidar_buffer.append(lidar_data)

    def imu_callback(self, msg):
        """Process IMU data for attitude estimation"""
        with self.state_lock:
            # Extract IMU measurements
            angular_velocity = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            linear_acceleration = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Store IMU data
            imu_data = {
                'angular_velocity': angular_velocity,
                'linear_acceleration': linear_acceleration,
                'orientation': np.array([
                    msg.orientation.x, msg.orientation.y,
                    msg.orientation.z, msg.orientation.w
                ]),
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
            self.imu_buffer.append(imu_data)

    def fusion_callback(self):
        """Main fusion algorithm"""
        with self.state_lock:
            # Get latest sensor data
            if not self.lidar_buffer or not self.imu_buffer:
                return

            # Perform sensor fusion (simplified Extended Kalman Filter approach)
            predicted_state = self.predict_state()
            updated_state = self.update_with_sensors(predicted_state)

            # Publish fused pose
            self.publish_fused_pose(updated_state)

    def predict_state(self):
        """Predict state using motion model"""
        # Simple motion model: constant velocity
        dt = 0.05  # Time step from timer

        new_pose = self.robot_pose.copy()
        new_pose[0] += self.robot_velocity[0] * dt * np.cos(new_pose[2]) - self.robot_velocity[1] * dt * np.sin(new_pose[2])
        new_pose[1] += self.robot_velocity[0] * dt * np.sin(new_pose[2]) + self.robot_velocity[1] * dt * np.cos(new_pose[2])
        new_pose[2] += self.robot_velocity[2] * dt  # Angular position update

        # Normalize angle
        new_pose[2] = ((new_pose[2] + np.pi) % (2 * np.pi)) - np.pi

        return new_pose

    def update_with_sensors(self, predicted_state):
        """Update state estimate with sensor measurements"""
        # This is a simplified update - in practice, you'd implement proper EKF equations
        updated_state = predicted_state.copy()

        # Example: Use LiDAR features for position correction
        if self.lidar_buffer:
            latest_lidar = self.lidar_buffer[-1]

            # Extract features from LiDAR data (simplified)
            valid_ranges = latest_lidar['ranges'][latest_lidar['ranges'] > 0.1]
            if len(valid_ranges) > 0:
                # Calculate average range as a simple feature
                avg_range = np.mean(valid_ranges)

                # This is a very simplified example - in reality you'd match features
                # to a map to get position corrections
                if avg_range < 2.0:  # Close to obstacles
                    # Could indicate position correction needed
                    pass

        # Example: Use IMU for orientation correction
        if self.imu_buffer:
            latest_imu = self.imu_buffer[-1]
            imu_rotation = R.from_quat(latest_imu['orientation'])
            imu_yaw = imu_rotation.as_euler('xyz')[2]

            # Blend predicted yaw with IMU measurement
            alpha = 0.7  # Trust IMU more for orientation
            updated_state[2] = alpha * imu_yaw + (1 - alpha) * predicted_state[2]

        return updated_state

    def publish_fused_pose(self, state):
        """Publish the fused pose estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose.position.x = float(state[0])
        pose_msg.pose.pose.position.y = float(state[1])
        pose_msg.pose.pose.position.z = 0.0  # Assume 2D

        # Convert yaw to quaternion
        qw = np.cos(state[2] / 2)
        qz = np.sin(state[2] / 2)
        pose_msg.pose.pose.orientation.w = qw
        pose_msg.pose.pose.orientation.z = qz

        # Set covariance (simplified)
        cov = self.pose_covariance.flatten()
        pose_msg.pose.covariance = cov.tolist()

        self.pose_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultiSensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```