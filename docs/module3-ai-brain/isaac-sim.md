---
title: NVIDIA Isaac Sim & Synthetic Data Generation
description: Understanding NVIDIA Isaac Sim for robotics simulation and synthetic data generation
sidebar_position: 1
---

# NVIDIA Isaac Sim & Synthetic Data Generation

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the architecture and capabilities of NVIDIA Isaac Sim
2. Set up basic simulation environments in Isaac Sim
3. Generate synthetic data for computer vision and robotics tasks
4. Integrate Isaac Sim with ROS 2 for robotics applications

## Introduction to NVIDIA Isaac Sim

NVIDIA Isaac Sim is a robotics simulator built on NVIDIA Omniverse, providing high-fidelity physics simulation and photorealistic rendering. It's designed for developing, testing, and validating AI-based robotics applications with:

- **PhysX Physics Engine**: Accurate physics simulation for robot dynamics
- **RTX Ray Tracing**: Photorealistic rendering for synthetic data generation
- **ROS 2 Bridge**: Seamless integration with ROS 2 ecosystem
- **Synthetic Data Generation**: Tools for creating labeled datasets
- **AI Training Environment**: Ready for reinforcement learning and imitation learning

## Isaac Sim Architecture

Isaac Sim has a modular architecture built on Omniverse:

- **Simulation Engine**: PhysX for physics, RTX for rendering
- **Extension System**: Python-based extensions for custom functionality
- **USD Scene Graph**: Universal Scene Description for 3D scene representation
- **ROS 2 Bridge**: Two-way communication with ROS 2 nodes
- **AI Framework Integration**: Support for TensorFlow, PyTorch, and Isaac ROS

## Setting Up Isaac Sim

### Basic Python Script Structure

```python
import omni
import carb
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np

# Initialize Isaac Sim
def setup_isaac_sim():
    # Create a world object
    world = World(stage_units_in_meters=1.0)

    # Get the assets root path
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets. Ensure Isaac Sim is properly installed.")
        return None

    return world

# Example: Loading a robot asset
def load_robot(world, robot_path="/Isaac/Robots/Franka/franka_alt_fingers.usd"):
    assets_root_path = get_assets_root_path()
    robot_prim_path = "/World/Robot"

    # Add robot to the stage
    add_reference_to_stage(
        usd_path=assets_root_path + robot_path,
        prim_path=robot_prim_path
    )

    # Load the robot into the world
    robot = world.scene.add_from_usd(
        usd_path=assets_root_path + robot_path,
        prim_path=robot_prim_path,
        position=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0, 1])
    )

    return robot

def main():
    # Setup Isaac Sim
    world = setup_isaac_sim()
    if world is None:
        return

    # Load a robot
    robot = load_robot(world)

    # Reset the world to initialize all objects
    world.reset()

    # Main simulation loop
    for i in range(1000):
        # Step the world
        world.step(render=True)

        # Perform actions every 100 steps
        if i % 100 == 0:
            print(f"Simulation step: {i}")

    # Cleanup
    world.clear()

if __name__ == "__main__":
    main()
```

## Synthetic Data Generation

Isaac Sim excels at generating synthetic data for training computer vision and robotics models:

### RGB, Depth, and Semantic Segmentation Data

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import cv2
import os

class SyntheticDataGenerator:
    def __init__(self, world, output_dir="synthetic_data"):
        self.world = world
        self.output_dir = output_dir
        self.camera = None
        self.setup_output_directory()

    def setup_output_directory(self):
        """Create output directory for synthetic data"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth", exist_ok=True)
        os.makedirs(f"{self.output_dir}/seg", exist_ok=True)

    def setup_camera(self, prim_path="/World/Camera", position=[1, 1, 1], rotation=[0, 0, 0]):
        """Setup camera for data capture"""
        self.camera = Camera(
            prim_path=prim_path,
            position=position,
            frequency=20,  # 20 Hz capture rate
        )

        # Add the camera to the world
        self.world.scene.add(self.camera)

        # Enable different types of data capture
        self.camera.add_ground_truth_to_frame({
            "rgb": "/Render/rgb",
            "depth": "/Render/depth",
            "semantic_segmentation": "/Render/semantic_segmentation"
        })

        return self.camera

    def capture_data_frame(self, frame_id):
        """Capture a single frame of synthetic data"""
        # Step the world to render the frame
        self.world.step(render=True)

        # Get the data
        data = self.camera.get_frame()

        # Save RGB image
        if "rgb" in data:
            rgb_img = data["rgb"]
            cv2.imwrite(f"{self.output_dir}/rgb/frame_{frame_id:06d}.png",
                       cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Save depth image
        if "depth" in data:
            depth_img = data["depth"]
            # Normalize depth for visualization
            depth_normalized = ((depth_img - depth_img.min()) /
                               (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
            cv2.imwrite(f"{self.output_dir}/depth/frame_{frame_id:06d}.png", depth_normalized)

        # Save semantic segmentation
        if "semantic_segmentation" in data:
            seg_img = data["semantic_segmentation"]["data"]
            cv2.imwrite(f"{self.output_dir}/seg/frame_{frame_id:06d}.png", seg_img)

        return data

    def generate_dataset(self, num_frames=1000):
        """Generate a synthetic dataset"""
        print(f"Generating {num_frames} frames of synthetic data...")

        for i in range(num_frames):
            # Move camera or objects to create variety
            self.move_camera_randomly()

            # Capture the frame
            self.capture_data_frame(i)

            if i % 100 == 0:
                print(f"Captured {i}/{num_frames} frames")

        print(f"Dataset generation complete! Data saved to {self.output_dir}")

    def move_camera_randomly(self):
        """Move camera to random positions for data variety"""
        # Random position around the scene
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        z = np.random.uniform(1, 3)

        # Random rotation
        rot_x = np.random.uniform(-0.5, 0.5)
        rot_y = np.random.uniform(-0.5, 0.5)
        rot_z = np.random.uniform(-np.pi, np.pi)

        self.camera.set_world_pose(position=np.array([x, y, z]),
                                 orientation=rot_to_quat([rot_x, rot_y, rot_z]))

def rot_to_quat(rotation):
    """Convert rotation vector to quaternion"""
    # Simplified conversion - in practice, use proper rotation matrix to quaternion conversion
    return np.array([0, 0, 0, 1])  # Placeholder

# Example usage
def main():
    world = World(stage_units_in_meters=1.0)

    # Setup robot and environment (simplified)
    # ... add robot and objects to the scene ...

    # Create synthetic data generator
    data_gen = SyntheticDataGenerator(world)

    # Setup camera
    camera = data_gen.setup_camera(position=[1, 1, 1.5])

    # Generate dataset
    data_gen.generate_dataset(num_frames=500)

    # Cleanup
    world.clear()

if __name__ == "__main__":
    main()
```

## Isaac ROS Integration

Isaac ROS provides optimized perception and manipulation capabilities that work with Isaac Sim:

### Example: Isaac ROS Perception Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Publishers for Isaac ROS pipeline
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_rect_color', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/rgb/camera_info', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)  # 10 Hz

        # Simulated camera parameters
        self.camera_matrix = np.array([
            [616.17865, 0.0, 311.10156],
            [0.0, 616.17865, 229.35829],
            [0.0, 0.0, 1.0]
        ])

        self.get_logger().info("Isaac Perception Node initialized")

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        # Create and publish camera info
        camera_info = CameraInfo()
        camera_info.header.stamp = self.get_clock().now().to_msg()
        camera_info.header.frame_id = 'camera_link'
        camera_info.height = 480
        camera_info.width = 640
        camera_info.k = self.camera_matrix.flatten().tolist()

        self.camera_info_pub.publish(camera_info)

        # Publish TF transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.1
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        # In a real implementation, this would publish actual sensor data from Isaac Sim
        # For now, we'll just log that data would be published
        self.get_logger().debug("Published simulated sensor data")

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Reinforcement Learning Environment Example

Isaac Sim can be used to create reinforcement learning environments:

```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import torch
import gym
from gym import spaces

class IsaacRLEnv(gym.Env):
    """Gym environment wrapper for Isaac Sim"""

    def __init__(self, render=False):
        super().__init__()

        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.render = render

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32  # 7-DOF robot
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32  # joint pos + vel
        )

        # Robot parameters
        self.robot = None
        self.default_dof_pos = np.zeros(7)

        # Setup the environment
        self.setup_scene()

    def setup_scene(self):
        """Setup the RL environment scene"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets")

        # Add ground plane
        plane_path = assets_root_path + "/Isaac/Props/Prismarine/Stage/usd/ground_plane.usd"
        add_reference_to_stage(usd_path=plane_path, prim_path="/World/GroundPlane")

        # Add robot
        robot_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")

        # Load robot into the world
        self.robot = self.world.scene.add(
            ArticulationView(
                prim_path="/World/Robot",
                name="franka_robot",
                reset_xform_properties=False,
            )
        )

        # Reset the world to initialize everything
        self.world.reset()

        # Set default joint positions
        self.robot.set_joint_positions(self.default_dof_pos)

    def reset(self):
        """Reset the environment"""
        # Reset the world
        self.world.reset()

        # Set random initial joint positions
        initial_pos = np.random.uniform(-0.5, 0.5, size=(7,))
        self.robot.set_joint_positions(initial_pos)

        # Step the world to apply changes
        self.world.step(render=self.render)

        # Return initial observation
        return self.get_observation()

    def step(self, action):
        """Execute one step in the environment"""
        # Apply action (joint position targets)
        current_pos = self.robot.get_joint_positions()
        target_pos = current_pos + action * 0.05  # Scale action appropriately

        self.robot.set_joint_positions(target_pos)

        # Step the simulation
        self.world.step(render=self.render)

        # Get observation
        obs = self.get_observation()

        # Calculate reward (simplified - in practice this would be task-specific)
        reward = self.calculate_reward()

        # Check if episode is done
        done = self.is_done()

        # Info dictionary
        info = {}

        return obs, reward, done, info

    def get_observation(self):
        """Get the current observation"""
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()

        # Concatenate position and velocity
        obs = np.concatenate([joint_pos, joint_vel])
        return obs.astype(np.float32)

    def calculate_reward(self):
        """Calculate reward for current state"""
        # Simplified reward function
        # In practice, this would be task-specific
        return 0.0

    def is_done(self):
        """Check if the episode is done"""
        # Simplified termination condition
        return False

    def close(self):
        """Clean up the environment"""
        self.world.clear()

# Example usage of the RL environment
def run_rl_example():
    """Run a simple example with the RL environment"""
    env = IsaacRLEnv(render=False)

    # Run a few episodes
    for episode in range(3):
        obs = env.reset()
        total_reward = 0

        for step in range(100):
            # Random action
            action = env.action_space.sample()

            # Take a step
            obs, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    run_rl_example()
```

## Best Practices for Isaac Sim

1. **Scene Complexity**: Balance scene complexity with simulation performance
2. **Asset Optimization**: Use appropriate level of detail for different tasks
3. **Physics Parameters**: Tune physics parameters for your specific robot and tasks
4. **Data Quality**: Ensure synthetic data quality matches real-world requirements
5. **Validation**: Always validate simulation results with real-world data when possible

## Summary

This chapter introduced NVIDIA Isaac Sim for robotics simulation and synthetic data generation. We covered the architecture, setup, synthetic data generation capabilities, and integration with ROS 2 and reinforcement learning frameworks. Isaac Sim provides a powerful platform for developing and testing AI-based robotics applications.

## Exercises

1. Create an Isaac Sim scene with multiple objects and generate a synthetic dataset with 1000 frames.
2. Implement a simple RL environment for a 2-DOF planar manipulator in Isaac Sim.
3. How would you modify the synthetic data generation pipeline to include multiple camera viewpoints?

## Code Example: Isaac Sim Object Detection Training Dataset

```python
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np
import cv2
import json
import os
from PIL import Image
import random

class ObjectDetectionDatasetGenerator:
    def __init__(self, world, output_dir="object_detection_dataset"):
        self.world = world
        self.output_dir = output_dir
        self.camera = None
        self.objects = []
        self.annotations = []

        # Create output directories
        os.makedirs(f"{self.output_dir}/images", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels", exist_ok=True)

    def setup_camera(self):
        """Setup camera for object detection dataset"""
        self.camera = Camera(
            prim_path="/World/Camera",
            position=[2, 2, 1.5],
            frequency=10,  # 10 Hz
        )
        self.world.scene.add(self.camera)

        # Enable RGB capture
        self.camera.add_ground_truth_to_frame({"rgb": "/Render/rgb"})

    def add_objects_to_scene(self, num_objects=5):
        """Add random objects to the scene"""
        # In a real implementation, you would add actual 3D objects
        # For this example, we'll simulate object positions
        object_types = ["box", "cylinder", "sphere"]

        for i in range(num_objects):
            obj_type = random.choice(object_types)
            position = [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(0.1, 0.5)
            ]

            self.objects.append({
                "id": i,
                "type": obj_type,
                "position": position,
                "size": random.uniform(0.1, 0.3)
            })

    def capture_annotated_frame(self, frame_id):
        """Capture a frame with bounding box annotations"""
        # Step the world
        self.world.step(render=True)

        # Get RGB image
        data = self.camera.get_frame()
        rgb_img = data["rgb"]

        # Generate random bounding boxes (in a real implementation, these would come from ground truth)
        height, width = rgb_img.shape[:2]

        # Simulate object detection annotations
        annotations = []
        for obj in self.objects:
            # Generate random bounding box (in practice, this would come from ground truth)
            x_center = random.uniform(0.2, 0.8) * width
            y_center = random.uniform(0.2, 0.8) * height
            bbox_width = random.uniform(0.1, 0.3) * width
            bbox_height = random.uniform(0.1, 0.3) * height

            x_min = max(0, x_center - bbox_width/2)
            y_min = max(0, y_center - bbox_height/2)
            x_max = min(width, x_center + bbox_width/2)
            y_max = min(height, y_center + bbox_height/2)

            annotation = {
                "category": obj["type"],
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "area": float((x_max - x_min) * (y_max - y_min)),
                "iscrowd": 0
            }
            annotations.append(annotation)

        # Save image
        img_filename = f"{self.output_dir}/images/frame_{frame_id:06d}.jpg"
        cv2.imwrite(img_filename, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # Save annotation
        annotation_data = {
            "image_id": frame_id,
            "width": width,
            "height": height,
            "annotations": annotations
        }

        annot_filename = f"{self.output_dir}/labels/frame_{frame_id:06d}.json"
        with open(annot_filename, 'w') as f:
            json.dump(annotation_data, f, indent=2)

        return img_filename, annot_filename

    def generate_dataset(self, num_frames=100):
        """Generate object detection training dataset"""
        print(f"Generating {num_frames} annotated frames for object detection...")

        for i in range(num_frames):
            # Clear previous objects and add new ones
            self.objects = []
            self.add_objects_to_scene(random.randint(1, 5))

            # Capture annotated frame
            img_path, annot_path = self.capture_annotated_frame(i)

            if i % 20 == 0:
                print(f"Generated {i}/{num_frames} frames")

        print(f"Dataset generation complete! Data saved to {self.output_dir}")

        # Create a simple dataset info file
        dataset_info = {
            "name": "Isaac Sim Object Detection Dataset",
            "description": "Synthetic dataset for object detection training",
            "num_frames": num_frames,
            "image_size": [640, 480],  # Default camera resolution
            "object_categories": ["box", "cylinder", "sphere"],
            "created_at": "2025-12-10"
        }

        with open(f"{self.output_dir}/dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

def main():
    # Initialize Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Create dataset generator
    dataset_gen = ObjectDetectionDatasetGenerator(world)

    # Setup camera
    dataset_gen.setup_camera()

    # Generate dataset
    dataset_gen.generate_dataset(num_frames=50)

    # Cleanup
    world.clear()

if __name__ == "__main__":
    main()
```