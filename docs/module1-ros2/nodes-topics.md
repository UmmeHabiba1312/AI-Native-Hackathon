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

## ROS 2 Nodes

A node is a process that performs computation. In ROS 2, nodes are the fundamental building blocks of an application. Each node is designed to perform a specific task and can communicate with other nodes to achieve more complex behaviors.

### Creating a Node

In Python, a node is created by inheriting from the `Node` class:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Node initialization code here

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Topics and Publishers/Subscribers

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
        msg.data = 'Hello World'
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

## Services

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

## Actions

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

## Communication Patterns Comparison

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| Topics | Publish/Subscribe | Streaming data, broadcasting | Asynchronous, decoupled |
| Services | Request/Response | Getting specific result | Synchronous, blocking |
| Actions | Goal/Result/Feedback | Long-running tasks | Cancellable, with progress |

## Best Practices

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent names
3. **Message Types**: Use appropriate message types for your data
4. **Error Handling**: Implement proper error handling and logging
5. **Resource Management**: Clean up resources properly in node destruction

## Summary

This chapter covered the fundamental communication patterns in ROS 2: nodes, topics, services, and actions. Each pattern serves different purposes and should be chosen based on the specific requirements of your robot's communication needs.

## Exercises

1. Create a ROS 2 node that publishes sensor data to a topic and another node that subscribes to this topic and processes the data.
2. Implement a service that takes a string and returns the reversed version.
3. When would you use an action instead of a service? Give an example.

## Code Example: Complete Publisher-Subscriber System

```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DataPublisher(Node):
    def __init__(self):
        super().__init__('data_publisher')
        self.publisher_ = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading {self.counter}: {self.counter * 10}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    data_publisher = DataPublisher()
    rclpy.spin(data_publisher)
    data_publisher.destroy_node()
    rclpy.shutdown()

# subscriber_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received sensor data: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    data_subscriber = DataSubscriber()
    rclpy.spin(data_subscriber)
    data_subscriber.destroy_node()
    rclpy.shutdown()
```