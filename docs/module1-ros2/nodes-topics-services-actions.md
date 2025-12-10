---
title: ROS 2 Nodes, Topics, Services, and Actions
description: Understanding the core communication patterns in ROS 2
sidebar_position: 2
---

# ROS 2 Nodes, Topics, Services, and Actions

## Learning Objectives

By the end of this chapter, you will be able to:
1. Create and run a ROS 2 node
2. Implement topic-based communication between nodes
3. Implement service-based communication
4. Understand when to use actions vs services
5. Design appropriate communication patterns for robotic applications

## Introduction to ROS 2 Communication Patterns

ROS 2 provides several communication patterns for different types of interactions between nodes. Each pattern serves a specific purpose and should be chosen based on the requirements of your robotic application.

### Communication Pattern Selection Guide

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| Topics | Publish/Subscribe | Streaming data, broadcasting | Asynchronous, decoupled |
| Services | Request/Response | Getting specific result | Synchronous, blocking |
| Actions | Goal/Result/Feedback | Long-running tasks | Cancellable, with progress |

## Nodes: The Foundation of ROS 2

A node is a process that performs computation. In ROS 2, nodes are the fundamental building blocks of an application. Each node is designed to perform a specific task and can communicate with other nodes to achieve more complex behaviors.

### Creating a Node in Python

In Python, a node is created by inheriting from the `Node` class:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle Management

ROS 2 provides lifecycle management for nodes to handle complex startup and shutdown procedures:

```python
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class LifecyclePublisher(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_publisher')
        self.pub = None

    def on_configure(self, state):
        self.get_logger().info("Configuring LifecyclePublisher")
        self.pub = self.create_publisher(String, "lifecycle_chatter", 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info("Activating LifecyclePublisher")
        return super().on_activate(state)

    def on_deactivate(self, state):
        self.get_logger().info("Deactivating LifecyclePublisher")
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        self.get_logger().info("Cleaning up LifecyclePublisher")
        self.destroy_publisher(self.pub)
        return TransitionCallbackReturn.SUCCESS
```

## Topics: Publish/Subscribe Communication

Topics enable asynchronous, decoupled communication between nodes. The publisher-subscriber pattern allows nodes to send and receive messages without having direct knowledge of each other.

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %s' % self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

### Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile with reliable delivery
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Create publisher with specific QoS
publisher = self.create_publisher(String, 'topic', qos_profile)

# Create subscription with matching QoS
subscription = self.create_subscription(
    String,
    'topic',
    self.callback,
    qos_profile
)
```

## Services: Request/Response Communication

Services provide synchronous, request-response communication. They are useful when you need a specific response to a request, unlike topics which are asynchronous.

### Service Server Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {request.a} + {request.b} = {response.sum}')
        return response
```

### Service Client Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions: Goal-Based Communication

Actions are used for long-running tasks that may take a significant amount of time to complete. They provide feedback during execution and can be canceled.

### Action Server Example

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

### Action Client Example

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')
```

## Communication Pattern Selection

Choosing the right communication pattern is crucial for effective robot design:

### When to Use Topics
- Broadcasting sensor data (camera images, laser scans)
- Publishing robot state (joint positions, odometry)
- Continuous data streams
- Many-to-many communication

### When to Use Services
- Requesting specific computations (path planning)
- Getting current state information
- One-time operations
- Synchronous responses needed

### When to Use Actions
- Long-running operations (navigation to goal)
- Operations that provide feedback (manipulation tasks)
- Cancellable operations
- Tasks with intermediate results

## Advanced Communication Concepts

### Message Filters

Message filters help synchronize messages from multiple topics:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber

def sync_callback(image_msg, depth_msg):
    # Process synchronized image and depth messages
    pass

image_sub = Subscriber(node, Image, 'camera/image')
depth_sub = Subscriber(node, Image, 'camera/depth')

ats = ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.1)
ats.registerCallback(sync_callback)
```

### Transforms (TF2)

TF2 manages coordinate transformations between different frames:

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_transform)

    def broadcast_transform(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'turtle1'
        t.child_frame_id = 'carrot1'

        t.transform.translation.x = 1.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

## Best Practices

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent names following ROS conventions
3. **Message Types**: Use appropriate message types for your data
4. **Error Handling**: Implement proper error handling and logging
5. **Resource Management**: Clean up resources properly in node destruction
6. **QoS Configuration**: Choose appropriate QoS settings for your use case
7. **Security**: Implement proper security measures for network communication

## Summary

This chapter covered the fundamental communication patterns in ROS 2: nodes, topics, services, and actions. Each pattern serves different purposes and should be chosen based on the specific requirements of your robot's communication needs. Proper selection and implementation of these patterns are crucial for creating robust and maintainable robotic systems.

## Exercises

1. Create a ROS 2 node that publishes sensor data to a topic and another node that subscribes to this topic and processes the data.
2. Implement a service that takes a string and returns the reversed version.
3. When would you use an action instead of a service? Give an example.
4. Design a communication architecture for a mobile robot with camera, lidar, and arm.

## Code Example: Complete Publisher-Subscriber System

```python
#!/usr/bin/env python3
# publisher_subscriber_example.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import numpy as np

class SensorDataPublisher(Node):
    def __init__(self):
        super().__init__('sensor_data_publisher')

        # Create publishers for different sensor data
        self.laser_pub = self.create_publisher(LaserScan, 'laser_scan', 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)

        # Timer for publishing data
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

        self.counter = 0
        self.get_logger().info('Sensor Data Publisher started')

    def publish_sensor_data(self):
        # Publish laser scan data
        laser_msg = LaserScan()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'laser_frame'

        # Set laser scan parameters
        laser_msg.angle_min = -np.pi / 2
        laser_msg.angle_max = np.pi / 2
        laser_msg.angle_increment = np.pi / 180  # 1 degree increments
        laser_msg.time_increment = 0.0
        laser_msg.scan_time = 0.1
        laser_msg.range_min = 0.1
        laser_msg.range_max = 10.0

        # Generate simulated scan data
        num_scans = int((laser_msg.angle_max - laser_msg.angle_min) / laser_msg.angle_increment) + 1
        ranges = []
        for i in range(num_scans):
            # Simulate some obstacles
            distance = 2.0 + 0.5 * np.sin(self.counter * 0.1 + i * 0.01)
            ranges.append(distance)

        laser_msg.ranges = ranges
        laser_msg.intensities = [100.0] * len(ranges)  # Dummy intensities

        self.laser_pub.publish(laser_msg)

        # Publish robot status
        status_msg = String()
        status_msg.data = f'Active - Counter: {self.counter}'
        self.status_pub.publish(status_msg)

        self.counter += 1


class SensorDataSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_data_subscriber')

        # Create subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.laser_callback,
            10)

        self.status_sub = self.create_subscription(
            String,
            'robot_status',
            self.status_callback,
            10)

        self.get_logger().info('Sensor Data Subscriber started')

    def laser_callback(self, msg):
        # Process laser scan data
        if len(msg.ranges) > 0:
            min_range = min([r for r in msg.ranges if r != float('inf')], default=float('inf'))
            if min_range < 0.5:
                self.get_logger().warn(f'OBSTACLE DETECTED: {min_range:.2f}m')
            else:
                self.get_logger().info(f'Min range: {min_range:.2f}m')

    def status_callback(self, msg):
        self.get_logger().info(f'Robot status: {msg.data}')


def main(args=None):
    rclpy.init(args=args)

    publisher_node = SensorDataPublisher()
    subscriber_node = SensorDataSubscriber()

    # Create executor and add nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher_node)
    executor.add_node(subscriber_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # Cleanup
    publisher_node.destroy_node()
    subscriber_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```