---
title: Gazebo Physics Simulation & Environment Setup
description: Understanding Gazebo physics simulation and environment building for digital twins
sidebar_position: 1
---

# Gazebo Physics Simulation & Environment Setup

## Learning Objectives

By the end of this chapter, you will be able to:
1. Set up a Gazebo simulation environment
2. Create custom worlds and environments for simulation
3. Configure physics properties and parameters
4. Integrate ROS 2 with Gazebo for robot simulation

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development for:

- **Testing algorithms**: Validate robot behaviors in a safe, virtual environment
- **Robot design**: Evaluate robot designs before building physical prototypes
- **Training**: Generate synthetic data for machine learning
- **Education**: Provide a platform for learning robotics concepts

## Gazebo Architecture

Gazebo has a client-server architecture:

- **Server (gzserver)**: Runs the physics simulation, sensors, and robot dynamics
- **Client (gzclient)**: Provides the graphical user interface
- **Plugins**: Extend functionality for custom sensors, controllers, and communication

## Setting Up a Gazebo Environment

### Creating a World File

Gazebo worlds are defined in SDF (Simulation Description Format) files, which are XML-based:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Include default world properties -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a simple room -->
    <model name="room">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>

      <!-- Floor -->
      <link name="floor">
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Walls -->
      <link name="wall1">
        <pose>0 5.05 2.5 0 0 0</pose>
        <collision name="wall1_collision">
          <geometry>
            <box>
              <size>10 0.1 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall1_visual">
          <geometry>
            <box>
              <size>10 0.1 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Add more walls as needed -->
    </model>
  </world>
</sdf>
```

## Physics Configuration

Gazebo provides extensive physics configuration options:

```xml
<world name="physics_world">
  <!-- Physics engine configuration -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>

    <!-- ODE-specific parameters -->
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.0</sor>
      </solver>
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

## Python Integration with Gazebo

### Using Gazebo Services from Python

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState, GetEntityState
from geometry_msgs.msg import Pose, Twist
import time

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # Create clients for Gazebo services
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        self.pause_physics_client = self.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.create_client(Empty, '/unpause_physics')

        # Service for setting entity state
        self.set_entity_state_client = self.create_client(
            SetEntityState, '/set_entity_state')

        # Service for getting entity state
        self.get_entity_state_client = self.create_client(
            GetEntityState, '/get_entity_state')

    def reset_simulation(self):
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting again...')

        request = Empty.Request()
        future = self.reset_simulation_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def set_robot_position(self, entity_name, x, y, z):
        while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set entity state service not available, waiting...')

        request = SetEntityState.Request()
        request.state.name = entity_name
        request.state.pose.position.x = x
        request.state.pose.position.y = y
        request.state.pose.position.z = z
        request.state.pose.orientation.w = 1.0  # No rotation

        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def get_robot_position(self, entity_name, reference_frame="world"):
        while not self.get_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get entity state service not available, waiting...')

        request = GetEntityState.Request()
        request.name = entity_name
        request.reference_frame = reference_frame

        future = self.get_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.success:
            pose = result.state.pose
            return (pose.position.x, pose.position.y, pose.position.z)
        else:
            self.get_logger().error(f"Failed to get state: {result.status_message}")
            return None

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()

    # Example usage
    controller.reset_simulation()
    time.sleep(1)

    # Move robot to new position
    controller.set_robot_position("my_robot", 1.0, 2.0, 0.0)
    time.sleep(1)

    # Get current position
    position = controller.get_robot_position("my_robot")
    if position:
        print(f"Robot position: {position}")

    controller.destroy_node()
    rclpy.shutdown()
```

## Creating Custom Sensors in Gazebo

Gazebo supports various sensor types that can be integrated with ROS 2:

```xml
<model name="sensor_robot">
  <!-- Example: RGB camera sensor -->
  <link name="camera_link">
    <pose>0.1 0 0.1 0 0 0</pose>
    <sensor name="camera" type="camera">
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <topic_name>/camera/image_raw</topic_name>
      </plugin>
    </sensor>
  </link>

  <!-- Example: IMU sensor -->
  <link name="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <frame_name>imu_link</frame_name>
        <topic_name>/imu/data</topic_name>
      </plugin>
    </sensor>
  </link>
</model>
```

## Environment Building Techniques

### Procedural Environment Generation

```python
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

class GazeboEnvironmentGenerator:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_random_obstacles(self, num_obstacles=5, area_size=(10, 10)):
        """
        Add randomly positioned obstacles to the environment
        """
        for i in range(num_obstacles):
            # Random position within area
            x = np.random.uniform(-area_size[0]/2, area_size[0]/2)
            y = np.random.uniform(-area_size[1]/2, area_size[1]/2)

            # Random size
            size_x = np.random.uniform(0.5, 2.0)
            size_y = np.random.uniform(0.5, 2.0)
            size_z = np.random.uniform(1.0, 3.0)

            obstacle = {
                'name': f'obstacle_{i}',
                'type': 'box',
                'position': (x, y, size_z/2),
                'size': (size_x, size_y, size_z),
                'color': (np.random.random(), np.random.random(), np.random.random(), 1.0)
            }
            self.objects.append(obstacle)

    def add_light_source(self, name, position, color=(1.0, 1.0, 1.0, 1.0)):
        """
        Add a light source to the environment
        """
        light = {
            'name': name,
            'position': position,
            'color': color
        }
        self.lights.append(light)

    def generate_sdf_world(self, world_name="generated_world"):
        """
        Generate an SDF world file based on the current configuration
        """
        sdf = ET.Element('sdf', version='1.7')
        world = ET.SubElement(sdf, 'world', name=world_name)

        # Add default ground plane and sun
        ground_include = ET.SubElement(world, 'include')
        ground_uri = ET.SubElement(ground_include, 'uri')
        ground_uri.text = 'model://ground_plane'

        sun_include = ET.SubElement(world, 'include')
        sun_uri = ET.SubElement(sun_include, 'uri')
        sun_uri.text = 'model://sun'

        # Add objects
        for obj in self.objects:
            model = ET.SubElement(world, 'model', name=obj['name'])

            # Position
            pose = ET.SubElement(model, 'pose')
            pose.text = f"{obj['position'][0]} {obj['position'][1]} {obj['position'][2]} 0 0 0"

            link = ET.SubElement(model, 'link', name=f"{obj['name']}_link")

            # Collision
            collision = ET.SubElement(link, 'collision', name=f"{obj['name']}_collision")
            geom_collision = ET.SubElement(collision, 'geometry')
            box_collision = ET.SubElement(geom_collision, obj['type'])
            size_collision = ET.SubElement(box_collision, 'size')
            size_collision.text = f"{obj['size'][0]} {obj['size'][1]} {obj['size'][2]}"

            # Visual
            visual = ET.SubElement(link, 'visual', name=f"{obj['name']}_visual")
            geom_visual = ET.SubElement(visual, 'geometry')
            box_visual = ET.SubElement(geom_visual, obj['type'])
            size_visual = ET.SubElement(box_visual, 'size')
            size_visual.text = f"{obj['size'][0]} {obj['size'][1]} {obj['size'][2]}"

            material = ET.SubElement(visual, 'material')
            ambient = ET.SubElement(material, 'ambient')
            ambient.text = f"{obj['color'][0]} {obj['color'][1]} {obj['color'][2]} {obj['color'][3]}"
            diffuse = ET.SubElement(material, 'diffuse')
            diffuse.text = f"{obj['color'][0]} {obj['color'][1]} {obj['color'][2]} {obj['color'][3]}"

        # Add lights
        for light in self.lights:
            light_elem = ET.SubElement(world, 'light', type='point', name=light['name'])
            pose = ET.SubElement(light_elem, 'pose')
            pose.text = f"{light['position'][0]} {light['position'][1]} {light['position'][2]} 0 0 0"
            diffuse_color = ET.SubElement(light_elem, 'diffuse')
            diffuse_color.text = f"{light['color'][0]} {light['color'][1]} {light['color'][2]} {light['color'][3]}"

        # Convert to pretty string
        rough_string = ET.tostring(sdf, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

# Example usage
generator = GazeboEnvironmentGenerator()
generator.add_random_obstacles(num_obstacles=10, area_size=(15, 15))
generator.add_light_source("main_light", (5, 5, 10), (1.0, 0.9, 0.8, 1.0))

sdf_content = generator.generate_sdf_world("random_obstacle_world")
print(sdf_content)
```

## Physics Parameter Tuning

Different simulation scenarios require different physics parameters:

### High-Fidelity Simulation (Slow but Accurate)
```xml
<physics type="ode">
  <max_step_size>0.0001</max_step_size>
  <real_time_factor>0.1</real_time_factor>
  <real_time_update_rate>10000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
      <sor>1.3</sor>
    </solver>
  </ode>
</physics>
```

### Fast Simulation (Less Accurate but Faster)
```xml
<physics type="ode">
  <max_step_size>0.01</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>100</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>
      <sor>1.0</sor>
    </solver>
  </ode>
</physics>
```

## Best Practices for Gazebo Simulation

1. **Start Simple**: Begin with basic environments and gradually add complexity
2. **Physics Tuning**: Balance accuracy and performance based on your needs
3. **Sensor Noise**: Add realistic noise models to make simulation more realistic
4. **Validation**: Compare simulation results with real-world data when possible
5. **Modularity**: Create reusable models and worlds for different experiments

## Summary

This chapter covered Gazebo physics simulation and environment setup. We learned how to create custom worlds, configure physics parameters, and integrate Gazebo with ROS 2 using Python. Proper environment setup is crucial for effective robot development and testing.

## Exercises

1. Create a Gazebo world file with multiple rooms and doorways for robot navigation.
2. Implement a Python script that dynamically adds obstacles to a running Gazebo simulation.
3. How would you modify the physics parameters for simulating a robot on a slippery surface?

## Code Example: Dynamic Obstacle Manager

```python
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
import random
import time

class DynamicObstacleManager(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_manager')

        # Create clients for spawning and deleting entities
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

        # Timer to periodically add/remove obstacles
        self.timer = self.create_timer(5.0, self.manage_obstacles)

        self.spawned_entities = []
        self.max_obstacles = 5

    def spawn_obstacle(self, name, x, y, z):
        """Spawn an obstacle at the specified position"""
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting...')

        # Create a simple box model SDF
        model_sdf = f"""
        <sdf version="1.7">
          <model name="{name}">
            <pose>{x} {y} {z} 0 0 0</pose>
            <link name="box_link">
              <collision name="box_collision">
                <geometry>
                  <box>
                    <size>1 1 1</size>
                  </box>
                </geometry>
              </collision>
              <visual name="box_visual">
                <geometry>
                  <box>
                    <size>1 1 1</size>
                  </box>
                </geometry>
                <material>
                  <ambient>0.8 0.2 0.2 1</ambient>
                  <diffuse>0.8 0.2 0.2 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>"""

        request = SpawnEntity.Request()
        request.name = name
        request.xml = model_sdf
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = z

        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.success:
            self.spawned_entities.append(name)
            self.get_logger().info(f"Spawned obstacle: {name}")
        else:
            self.get_logger().error(f"Failed to spawn obstacle: {result.status_message}")

    def delete_random_obstacle(self):
        """Delete a random obstacle from the simulation"""
        if not self.spawned_entities:
            return

        entity_name = random.choice(self.spawned_entities)

        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Delete service not available, waiting...')

        request = DeleteEntity.Request()
        request.name = entity_name

        future = self.delete_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.success:
            self.spawned_entities.remove(entity_name)
            self.get_logger().info(f"Deleted obstacle: {entity_name}")
        else:
            self.get_logger().error(f"Failed to delete obstacle: {result.status_message}")

    def manage_obstacles(self):
        """Periodically add or remove obstacles"""
        current_count = len(self.spawned_entities)

        if current_count < self.max_obstacles:
            # Add a new obstacle
            name = f"dynamic_obstacle_{len(self.spawned_entities)}"
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z = 0.5  # Half the box height

            self.spawn_obstacle(name, x, y, z)
        else:
            # Remove a random obstacle
            self.delete_random_obstacle()

def main(args=None):
    rclpy.init(args=args)
    manager = DynamicObstacleManager()

    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up by deleting all spawned entities
        for entity_name in manager.spawned_entities[:]:  # Copy the list
            manager.delete_random_obstacle()

        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```