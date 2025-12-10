---
title: URDF & Python Agent Integration
description: Understanding Unified Robot Description Format and its integration with Python agents
sidebar_position: 3
---

# URDF & Python Agent Integration

## Learning Objectives

By the end of this chapter, you will be able to:
1. Create and understand URDF files for robot description
2. Use Python to parse and manipulate URDF models
3. Integrate URDF with ROS 2 for robot state publishing
4. Understand how agents can work with URDF models

## What is URDF?

URDF (Unified Robot Description Format) is an XML format for representing a robot model. URDF is used in ROS to describe the physical and visual properties of a robot, including:

- **Links**: Rigid parts of the robot (e.g., base, arm segments)
- **Joints**: Connections between links (e.g., revolute, prismatic)
- **Visual**: How the robot appears in simulation
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties

## URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.2 0.0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Working with URDF in Python

### Parsing URDF Files

```python
import xml.etree.ElementTree as ET

def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    robot_name = root.get('name')
    print(f"Robot name: {robot_name}")

    # Parse links
    for link in root.findall('link'):
        link_name = link.get('name')
        print(f"Link: {link_name}")

        # Parse visual elements
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            if geometry is not None:
                for geom_type in ['box', 'cylinder', 'sphere', 'mesh']:
                    geom = geometry.find(geom_type)
                    if geom is not None:
                        print(f"  Geometry: {geom_type}")
                        for attr, value in geom.attrib.items():
                            print(f"    {attr}: {value}")

    # Parse joints
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        print(f"Joint: {joint_name} ({joint_type})")

# Usage
parse_urdf('path/to/robot.urdf')
```

### Creating URDF Programmatically

```python
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_simple_robot_urdf():
    # Create root robot element
    robot = ET.Element('robot', name='simple_robot')

    # Create base link
    base_link = ET.SubElement(robot, 'link', name='base_link')

    # Add visual properties to base link
    visual = ET.SubElement(base_link, 'visual')
    geometry = ET.SubElement(visual, 'geometry')
    box = ET.SubElement(geometry, 'box', size='0.5 0.5 0.2')

    # Add collision properties
    collision = ET.SubElement(base_link, 'collision')
    collision_geometry = ET.SubElement(collision, 'geometry')
    collision_box = ET.SubElement(collision_geometry, 'box', size='0.5 0.5 0.2')

    # Add inertial properties
    inertial = ET.SubElement(base_link, 'inertial')
    mass = ET.SubElement(inertial, 'mass', value='1.0')
    inertia = ET.SubElement(inertial, 'inertia',
                           ixx='0.0833', ixy='0.0', ixz='0.0',
                           iyy='0.0833', iyz='0.0', izz='0.0833')

    # Create wheel link
    wheel_link = ET.SubElement(robot, 'link', name='wheel_link')
    wheel_visual = ET.SubElement(wheel_link, 'visual')
    wheel_geom = ET.SubElement(wheel_visual, 'geometry')
    wheel_cyl = ET.SubElement(wheel_geom, 'cylinder', radius='0.1', length='0.05')

    # Create joint connecting base to wheel
    joint = ET.SubElement(robot, 'joint', name='base_to_wheel', type='continuous')
    parent = ET.SubElement(joint, 'parent', link='base_link')
    child = ET.SubElement(joint, 'child', link='wheel_link')
    origin = ET.SubElement(joint, 'origin', xyz='0.2 0.0 -0.1', rpy='0 0 0')
    axis = ET.SubElement(joint, 'axis', xyz='0 0 1')

    # Convert to string and pretty print
    rough_string = ET.tostring(robot, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

# Usage
urdf_content = create_simple_robot_urdf()
print(urdf_content)
```

## URDF with ROS 2 and Robot State Publisher

The robot_state_publisher package in ROS 2 takes a URDF and joint positions and publishes the appropriate TF transforms.

### Python Node for Robot State Publishing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create joint state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing state
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        # Initialize joint names and positions
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.joint_positions = [0.0, 0.0, 0.0]

        self.get_logger().info('Robot State Publisher started')

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update joint positions (for example purposes)
        time_sec = self.get_clock().now().nanoseconds / 1e9
        self.joint_positions[0] = math.sin(time_sec)
        self.joint_positions[1] = math.cos(time_sec)
        self.joint_positions[2] = math.sin(time_sec * 0.5)

        # Publish joint state
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Python Agent Integration with URDF

Python agents can work with URDF models to perform tasks like:

- Robot kinematics calculations
- Path planning based on robot dimensions
- Collision detection
- Simulation preparation

### Example: URDF Kinematics Agent

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class URDFKinematicsAgent:
    def __init__(self, urdf_file):
        self.urdf_file = urdf_file
        self.robot_model = self.load_urdf(urdf_file)
        self.joint_limits = self.extract_joint_limits()

    def load_urdf(self, urdf_file):
        # In a real implementation, this would parse the URDF file
        # and create a model representation
        print(f"Loading URDF from {urdf_file}")
        return {"links": [], "joints": []}

    def extract_joint_limits(self):
        # Extract joint limits from URDF
        return {
            "joint1": {"min": -2.0, "max": 2.0},
            "joint2": {"min": -1.5, "max": 1.5},
            "joint3": {"min": -3.0, "max": 3.0}
        }

    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector position given joint angles
        This is a simplified example
        """
        # For a simple 2-link planar manipulator:
        # Link lengths
        l1, l2 = 1.0, 0.8

        # Joint angles
        theta1, theta2 = joint_angles[0], joint_angles[1]

        # Calculate end-effector position
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

        return np.array([x, y, 0.0])  # z=0 for planar robot

    def inverse_kinematics(self, target_position):
        """
        Calculate joint angles for a target end-effector position
        """
        x, y, _ = target_position

        # For a simple 2-link planar manipulator
        l1, l2 = 1.0, 0.8

        # Calculate distance from origin to target
        r = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > l1 + l2:
            print("Target position is not reachable")
            return None

        if r < abs(l1 - l2):
            print("Target position is not reachable (too close)")
            return None

        # Calculate joint angles
        cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)

        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return np.array([theta1, theta2, 0.0])  # Add dummy third joint

    def check_joint_limits(self, joint_angles):
        """
        Check if joint angles are within limits
        """
        for i, angle in enumerate(joint_angles):
            joint_name = f"joint{i+1}"
            if joint_name in self.joint_limits:
                limits = self.joint_limits[joint_name]
                if angle < limits["min"] or angle > limits["max"]:
                    return False
        return True

# Usage example
if __name__ == "__main__":
    agent = URDFKinematicsAgent("robot.urdf")

    # Test forward kinematics
    joint_angles = [np.pi/4, np.pi/6, 0.0]
    ee_pos = agent.forward_kinematics(joint_angles)
    print(f"End-effector position: {ee_pos}")

    # Test inverse kinematics
    target_pos = np.array([1.2, 0.8, 0.0])
    joint_angles = agent.inverse_kinematics(target_pos)
    if joint_angles is not None:
        print(f"Joint angles for target: {joint_angles}")
        new_pos = agent.forward_kinematics(joint_angles)
        print(f"Verification - new position: {new_pos}")

    # Check joint limits
    is_valid = agent.check_joint_limits(joint_angles)
    print(f"Joint angles valid: {is_valid}")
```

## Best Practices for URDF

1. **Units**: Use meters for lengths, kilograms for mass, and radians for angles
2. **Inertial Properties**: Accurate inertial properties are crucial for simulation
3. **Collision vs Visual**: Use simpler collision geometry for performance
4. **Joint Limits**: Always specify joint limits to prevent damage
5. **Naming**: Use consistent and descriptive names

## Summary

This chapter covered URDF (Unified Robot Description Format) and its integration with Python agents. We learned how to parse, create, and work with URDF files programmatically, and how to integrate them with ROS 2 systems. Python agents can leverage URDF models for various robotics tasks including kinematics, path planning, and simulation.

## Exercises

1. Create a URDF file for a simple 3-link robot arm and write a Python script to parse it.
2. Implement a Python agent that can calculate the Jacobian matrix for a robot described in URDF.
3. How would you modify the URDF to include a gripper at the end of a robot arm?

## Code Example: URDF Validation Agent

```python
import xml.etree.ElementTree as ET
import math

class URDFValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_urdf(self, file_path):
        """
        Validate URDF file for common errors
        """
        self.errors = []
        self.warnings = []

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Check robot name
            if 'name' not in root.attrib or not root.attrib['name']:
                self.errors.append("Robot must have a name attribute")

            # Validate links
            links = root.findall('link')
            link_names = set()
            for link in links:
                name = link.get('name')
                if not name:
                    self.errors.append("Link must have a name attribute")
                elif name in link_names:
                    self.errors.append(f"Duplicate link name: {name}")
                else:
                    link_names.add(name)

                # Validate visual elements
                for visual in link.findall('visual'):
                    self.validate_visual_element(visual)

                # Validate collision elements
                for collision in link.findall('collision'):
                    self.validate_collision_element(collision)

                # Validate inertial elements
                for inertial in link.findall('inertial'):
                    self.validate_inertial_element(inertial)

            # Validate joints
            joints = root.findall('joint')
            joint_names = set()
            for joint in joints:
                name = joint.get('name')
                if not name:
                    self.errors.append("Joint must have a name attribute")
                elif name in joint_names:
                    self.errors.append(f"Duplicate joint name: {name}")
                else:
                    joint_names.add(name)

                joint_type = joint.get('type')
                if joint_type not in ['revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar']:
                    self.errors.append(f"Invalid joint type: {joint_type}")

                # Check parent and child links exist
                parent_elem = joint.find('parent')
                child_elem = joint.find('child')

                if parent_elem is None or 'link' not in parent_elem.attrib:
                    self.errors.append(f"Joint {name} must have a parent link")
                elif parent_elem.attrib['link'] not in link_names:
                    self.errors.append(f"Joint {name} parent link {parent_elem.attrib['link']} does not exist")

                if child_elem is None or 'link' not in child_elem.attrib:
                    self.errors.append(f"Joint {name} must have a child link")
                elif child_elem.attrib['link'] not in link_names:
                    self.errors.append(f"Joint {name} child link {child_elem.attrib['link']} does not exist")

            return len(self.errors) == 0

        except ET.ParseError as e:
            self.errors.append(f"XML Parse Error: {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"Error validating URDF: {str(e)}")
            return False

    def validate_visual_element(self, visual_elem):
        geometry = visual_elem.find('geometry')
        if geometry is None:
            self.errors.append("Visual element must have a geometry child")
            return

        geom_types = ['box', 'cylinder', 'sphere', 'mesh']
        found_geom = False
        for geom_type in geom_types:
            if geometry.find(geom_type) is not None:
                found_geom = True
                break

        if not found_geom:
            self.errors.append("Visual geometry must have one of: box, cylinder, sphere, mesh")

    def validate_collision_element(self, collision_elem):
        geometry = collision_elem.find('geometry')
        if geometry is None:
            self.errors.append("Collision element must have a geometry child")
            return

        geom_types = ['box', 'cylinder', 'sphere', 'mesh']
        found_geom = False
        for geom_type in geom_types:
            if geometry.find(geom_type) is not None:
                found_geom = True
                break

        if not found_geom:
            self.errors.append("Collision geometry must have one of: box, cylinder, sphere, mesh")

    def validate_inertial_element(self, inertial_elem):
        mass_elem = inertial_elem.find('mass')
        if mass_elem is None or 'value' not in mass_elem.attrib:
            self.errors.append("Inertial element must have a mass child with value attribute")
            return

        try:
            mass_value = float(mass_elem.attrib['value'])
            if mass_value <= 0:
                self.errors.append("Mass must be positive")
        except ValueError:
            self.errors.append("Mass value must be a number")

        inertia_elem = inertial_elem.find('inertia')
        if inertia_elem is None:
            self.errors.append("Inertial element must have an inertia child")
            return

        required_inertia_attrs = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
        for attr in required_inertia_attrs:
            if attr not in inertia_elem.attrib:
                self.errors.append(f"Inertia element missing {attr} attribute")
                continue

            try:
                value = float(inertia_elem.attrib[attr])
                if attr in ['ixx', 'iyy', 'izz'] and value <= 0:
                    self.errors.append(f"{attr} should be positive")
            except ValueError:
                self.errors.append(f"{attr} value must be a number")

    def get_validation_report(self):
        report = {
            'errors': self.errors,
            'warnings': self.warnings,
            'valid': len(self.errors) == 0
        }
        return report

# Usage example
validator = URDFValidator()
is_valid = validator.validate_urdf("robot.urdf")
report = validator.get_validation_report()

print(f"URDF is valid: {is_valid}")
if report['errors']:
    print("Errors found:")
    for error in report['errors']:
        print(f"  - {error}")
```