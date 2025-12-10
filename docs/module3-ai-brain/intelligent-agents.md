---
title: Intelligent Agents for Humanoid Control
description: Developing AI agents for autonomous humanoid robot control and decision making
sidebar_position: 5
---

# Intelligent Agents for Humanoid Control

## Learning Objectives

By the end of this chapter, you will be able to:
1. Design and implement AI agents for humanoid robot control and decision making
2. Create reinforcement learning agents for humanoid locomotion and balance
3. Implement multi-agent systems for coordinated humanoid behavior
4. Develop decision-making frameworks for autonomous humanoid navigation
5. Integrate AI agents with ROS 2 control systems for real-time operation

## Introduction to AI Agents for Humanoid Control

AI agents represent a paradigm shift in robotics control, moving from purely reactive or pre-programmed behaviors to adaptive, learning systems that can handle complex environments and tasks. For humanoid robots, AI agents offer several advantages:

- **Adaptive Control**: Ability to adjust to changing conditions and environments
- **Learning Capabilities**: Improvement over time through experience
- **Complex Decision Making**: Handling multiple objectives and constraints
- **Robustness**: Ability to recover from disturbances and failures

### Types of AI Agents for Humanoids

1. **Reinforcement Learning Agents**: Learn optimal control policies through trial and error
2. **Imitation Learning Agents**: Learn from demonstrations and expert behavior
3. **Planning Agents**: Make high-level decisions about navigation and task execution
4. **Multi-Agent Systems**: Coordinate multiple agents for complex behaviors
5. **Hybrid Agents**: Combine multiple AI techniques for robust performance

## Reinforcement Learning for Humanoid Control

### Deep Reinforcement Learning Architecture

Deep Reinforcement Learning (DRL) has shown remarkable success in learning complex humanoid behaviors. The key components include:

- **State Space**: Robot joint positions, velocities, IMU readings, task-specific information
- **Action Space**: Joint torques, positions, or velocities
- **Reward Function**: Encourages desired behaviors (balance, forward motion, energy efficiency)
- **Policy Network**: Maps states to actions
- **Value Network**: Estimates expected future rewards

### DDPG for Continuous Control

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor network for humanoid control"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    """Critic network for humanoid control"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class HumanoidDDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.device = device

        # Hyperparameters
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2

        self.total_it = 0

    def select_action(self, state):
        """Select action using the actor network"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        """Train the agent using experience replay"""
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """Save the model"""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """Load the model"""
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
```

### SAC for Sample-Efficient Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        return action[0].detach().cpu().numpy()

class HumanoidSACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, alpha=0.2, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.value_optim = optim.Adam(self.value.parameters(), lr=lr)

        self.target_entropy = -torch.prod(torch.Tensor(action_dim)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def update(self, replay_buffer, batch_size=128):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        not_done = torch.FloatTensor(not_done).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, log_prob, _, _, _ = self.actor.evaluate(next_state)

            next_q_value = self.critic_target(next_state, next_action)
            next_v = next_q_value - self.alpha * log_prob
            expected_q = reward + (not_done * self.gamma * next_v)

        # q loss
        q1, q2 = self.critic(state, action)
        value_loss = F.mse_loss(q1, expected_q) + F.mse_loss(q2, expected_q)

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # Update value
        predicted_value = self.value(state)
        v = q1 - self.alpha * log_prob
        value_loss = F.mse_loss(predicted_value, v.detach())

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # Update policy
        new_action, log_prob, _, mean, log_std = self.actor.evaluate(state)

        q1, q2 = self.critic(state, new_action)
        q_min = torch.min(q1, q2)
        policy_loss = ((self.alpha * log_prob) - q_min).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        # Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

## Humanoid-Specific Environment Implementation

### PyBullet Humanoid Environment

```python
import pybullet as p
import pybullet_data
import numpy as np
import math
import time

class HumanoidEnv:
    def __init__(self, render=False):
        self.render = render
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load humanoid model
        self.robot_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            [0, 0, 1.5],
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION
        )

        # Get joint information
        self.joint_indices = []
        self.joint_names = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_ranges = []
        self.rest_poses = []

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]

            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                self.joint_lower_limits.append(joint_info[8])
                self.joint_upper_limits.append(joint_info[9])
                self.joint_ranges.append(joint_info[9] - joint_info[8])
                self.rest_poses.append(joint_info[5])

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Set simulation parameters
        p.setTimeStep(1.0/60.0)  # 60 Hz simulation
        p.setRealTimeSimulation(0)  # Use stepping

        # Initialize environment parameters
        self.target_velocity = [0.0, 0.0, 0.0]  # Target forward velocity
        self.initial_position = [0, 0, 1.5]
        self.max_episode_steps = 1000
        self.step_count = 0

        # Define important links for balance
        self.torso_link_index = self.get_link_index('torso')
        self.left_foot_link_index = self.get_link_index('left_foot')
        self.right_foot_link_index = self.get_link_index('right_foot')

    def get_link_index(self, link_name):
        """Get link index by name"""
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == link_name:
                return i
        return -1

    def reset(self):
        """Reset the environment"""
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reload environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            self.initial_position,
            p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_SELF_COLLISION
        )

        # Reset joint positions to neutral
        for i, joint_index in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_index,
                self.rest_poses[i]
            )

        # Set joint motors to position control
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[0] * len(self.joint_indices),
            forces=[200] * len(self.joint_indices)  # Max torque
        )

        self.step_count = 0
        return self.get_state()

    def get_state(self):
        """Get current state of the robot"""
        # Get joint states
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Get base position and orientation
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)

        # Get IMU-like data (orientation and angular velocity)
        orientation = p.getEulerFromQuaternion(base_orn)

        # Get center of mass information
        com_pos = self.get_center_of_mass()

        # Get foot contact information
        left_contact = bool(p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_link_index))
        right_contact = bool(p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_link_index))

        # Normalize joint positions and velocities
        normalized_positions = []
        normalized_velocities = []

        for i, pos in enumerate(joint_positions):
            if self.joint_ranges[i] != 0:
                normalized_pos = 2 * (pos - self.joint_lower_limits[i]) / self.joint_ranges[i] - 1
                normalized_positions.append(normalized_pos)
            else:
                normalized_positions.append(0.0)

        for vel in joint_velocities:
            normalized_velocities.append(np.clip(vel / 10.0, -1.0, 1.0))  # Assuming max velocity of 10 rad/s

        # Construct state vector
        state = []
        state.extend(normalized_positions)
        state.extend(normalized_velocities)
        state.extend(list(base_pos))  # Base position
        state.extend(list(orientation))  # Base orientation (roll, pitch, yaw)
        state.extend(list(base_lin_vel))  # Base linear velocity
        state.extend(list(base_ang_vel))  # Base angular velocity
        state.extend([left_contact, right_contact])  # Contact information
        state.extend(list(com_pos))  # Center of mass position

        return np.array(state, dtype=np.float32)

    def get_center_of_mass(self):
        """Calculate center of mass of the robot"""
        total_mass = 0
        weighted_pos = np.array([0.0, 0.0, 0.0])

        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getDynamicsInfo(self.robot_id, i)
            mass = info[0]
            if mass > 0:  # Only consider links with mass
                link_state = p.getLinkState(self.robot_id, i)
                pos = np.array(link_state[0])
                weighted_pos += mass * pos
                total_mass += mass

        # Include base link
        base_mass = p.getDynamicsInfo(self.robot_id, -1)[0]
        if base_mass > 0:
            base_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
            weighted_pos += base_mass * base_pos
            total_mass += base_mass

        if total_mass > 0:
            com = weighted_pos / total_mass
        else:
            com = np.array([0.0, 0.0, 0.0])

        return com

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Apply action (torques)
        action = np.clip(action, -1, 1)  # Clip to [-1, 1]

        # Convert normalized action to actual joint torques
        torques = []
        for i, torque_norm in enumerate(action):
            max_torque = 200  # Maximum torque for each joint
            torques.append(torque_norm * max_torque)

        # Apply torques using torque control
        p.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            p.TORQUE_CONTROL,
            forces=torques
        )

        # Step simulation
        p.stepSimulation()
        self.step_count += 1

        # Get new state
        next_state = self.get_state()

        # Calculate reward
        reward = self.calculate_reward(next_state, action)

        # Check if episode is done
        done = self.is_done(next_state)

        # Additional info
        info = {
            'step_count': self.step_count,
            'episode_done': done,
            'reward': reward
        }

        return next_state, reward, done, info

    def calculate_reward(self, state, action):
        """Calculate reward based on current state and action"""
        # Extract relevant state components
        base_pos = state[6:9]  # Base position (after joint positions and velocities)
        base_orn = state[9:12]  # Base orientation (roll, pitch, yaw)
        base_lin_vel = state[12:15]  # Base linear velocity
        base_ang_vel = state[15:18]  # Base angular velocity
        left_contact = bool(state[18])  # Left foot contact
        right_contact = bool(state[19])  # Right foot contact
        com_pos = state[20:23]  # Center of mass position

        # Reward for forward velocity
        forward_vel_reward = base_lin_vel[0] * 10.0  # Encourage forward motion

        # Penalty for falling (deviation from upright position)
        pitch_penalty = abs(base_orn[1]) * 5.0  # Penalize pitch deviation
        roll_penalty = abs(base_orn[0]) * 5.0   # Penalize roll deviation

        # Penalty for angular velocity (instability)
        angular_vel_penalty = np.linalg.norm(base_ang_vel) * 2.0

        # Reward for maintaining balance (keeping CoM over support polygon)
        com_height = com_pos[2]
        com_height_reward = max(0, (com_height - 0.5) * 5.0)  # Reward for keeping CoM high

        # Penalty for excessive joint torques (energy efficiency)
        torque_penalty = np.mean(np.abs(action)) * 0.1

        # Reward for maintaining contact with ground (walking)
        contact_reward = (left_contact + right_contact) * 1.0

        # Total reward
        reward = (forward_vel_reward
                 - pitch_penalty
                 - roll_penalty
                 - angular_vel_penalty
                 + com_height_reward
                 - torque_penalty
                 + contact_reward)

        # Additional penalty for falling
        if abs(base_pos[2]) < 0.5:  # Robot fell (base too low)
            reward -= 100
        elif abs(base_orn[1]) > 1.0:  # Excessive pitch
            reward -= 50
        elif abs(base_orn[0]) > 1.0:  # Excessive roll
            reward -= 50

        return reward

    def is_done(self, state):
        """Check if episode is done"""
        base_pos = state[6:9]
        base_orn = state[9:12]

        # Episode ends if robot falls
        if base_pos[2] < 0.3:  # Robot base too low (fell)
            return True

        # Episode ends if robot is excessively tilted
        if abs(base_orn[0]) > 1.0 or abs(base_orn[1]) > 1.0:  # Too much roll or pitch
            return True

        # Episode ends if max steps reached
        if self.step_count >= self.max_episode_steps:
            return True

        return False

    def close(self):
        """Close the environment"""
        p.disconnect(self.client)
```

## Multi-Agent Systems for Humanoid Coordination

### Distributed Control Architecture

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
import numpy as np
import threading
import time

class HumanoidAgentNode(Node):
    """Base class for humanoid control agents"""

    def __init__(self, agent_name):
        super().__init__(f'{agent_name}_agent')

        self.agent_name = agent_name
        self.agent_id = self.generate_agent_id()
        self.agent_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
            'velocity': np.array([0.0, 0.0, 0.0]),
            'balance_state': 'stable',
            'health': 100.0,
            'last_update': self.get_clock().now()
        }

        # Communication with other agents
        self.agent_registry = {}  # Other agents in the system
        self.agent_states = {}    # States of other agents

        # Publishers and subscribers
        self.joint_cmd_pub = self.create_publisher(JointState, f'/{agent_name}/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{agent_name}/cmd_vel', 10)
        self.state_pub = self.create_publisher(Float64MultiArray, f'/{agent_name}/state', 10)

        self.joint_state_sub = self.create_subscription(JointState, f'/{agent_name}/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, f'/{agent_name}/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, f'/{agent_name}/odom', self.odom_callback, 10)

        # Agent communication
        self.agent_state_sub = self.create_subscription(Float64MultiArray, '/agent_states', self.agent_state_callback, 10)
        self.agent_comm_pub = self.create_publisher(Float64MultiArray, '/agent_communication', 10)

        # Agent control timer
        self.control_timer = self.create_timer(0.02, self.agent_control_loop)  # 50 Hz

        self.get_logger().info(f'{agent_name} agent initialized')

    def generate_agent_id(self):
        """Generate unique agent ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.agent_state['joint_positions'] = list(msg.position)
        self.agent_state['joint_velocities'] = list(msg.velocity)
        self.agent_state['joint_efforts'] = list(msg.effort)

    def imu_callback(self, msg):
        """Update IMU data and balance state"""
        self.agent_state['linear_acceleration'] = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]
        self.agent_state['angular_velocity'] = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        self.agent_state['orientation'] = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

        # Update balance state based on IMU data
        self.update_balance_state()

    def odom_callback(self, msg):
        """Update odometry data"""
        self.agent_state['position'] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        self.agent_state['velocity'] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ]

    def update_balance_state(self):
        """Update balance state based on IMU data"""
        # Calculate tilt angles from orientation
        quat = self.agent_state['orientation']
        # Convert quaternion to Euler angles
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion(quat)
        roll, pitch, yaw = euler

        # Check if robot is stable
        max_tilt = 0.3  # 17 degrees max tilt
        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            self.agent_state['balance_state'] = 'unstable'
        else:
            self.agent_state['balance_state'] = 'stable'

    def agent_state_callback(self, msg):
        """Receive state from other agents"""
        try:
            # Message format: [agent_id, x, y, z, qw, qx, qy, qz, vx, vy, vz, balance_state, health]
            data = msg.data
            if len(data) >= 13:
                agent_id = str(int(data[0]))  # Agent ID as integer converted to string

                # Update other agent's state
                self.agent_states[agent_id] = {
                    'position': np.array(data[1:4]),
                    'orientation': np.array(data[4:8]),
                    'velocity': np.array(data[8:11]),
                    'balance_state': 'stable' if data[11] > 0.5 else 'unstable',
                    'health': data[12],
                    'timestamp': self.get_clock().now()
                }
        except Exception as e:
            self.get_logger().error(f'Error parsing agent state: {e}')

    def agent_control_loop(self):
        """Main agent control loop"""
        try:
            # Update agent state timestamp
            self.agent_state['last_update'] = self.get_clock().now()

            # Execute agent-specific behavior
            self.execute_agent_behavior()

            # Publish agent state for other agents
            self.publish_agent_state()

            # Check for coordination opportunities
            self.check_coordination_opportunities()

        except Exception as e:
            self.get_logger().error(f'Agent control loop error: {e}')

    def execute_agent_behavior(self):
        """Execute agent-specific behavior - to be overridden by subclasses"""
        pass

    def publish_agent_state(self):
        """Publish agent state for other agents"""
        state_msg = Float64MultiArray()
        state_msg.data = [
            float(self.agent_id),  # Agent ID
            self.agent_state['position'][0],
            self.agent_state['position'][1],
            self.agent_state['position'][2],
            self.agent_state['orientation'][0],
            self.agent_state['orientation'][1],
            self.agent_state['orientation'][2],
            self.agent_state['orientation'][3],
            self.agent_state['velocity'][0],
            self.agent_state['velocity'][1],
            self.agent_state['velocity'][2],
            1.0 if self.agent_state['balance_state'] == 'stable' else 0.0,
            self.agent_state['health']
        ]
        self.state_pub.publish(state_msg)

    def check_coordination_opportunities(self):
        """Check for opportunities to coordinate with other agents"""
        # Check for nearby agents
        for other_agent_id, other_state in self.agent_states.items():
            if other_agent_id != self.agent_id:
                distance = np.linalg.norm(
                    self.agent_state['position'] - other_state['position']
                )

                if distance < 2.0:  # Within 2m of another agent
                    self.handle_agent_proximity(other_agent_id, other_state, distance)

    def handle_agent_proximity(self, other_agent_id, other_state, distance):
        """Handle when another agent is in proximity"""
        # Default behavior: maintain safe distance
        pass

class BalanceAgent(HumanoidAgentNode):
    """Agent responsible for balance control"""

    def __init__(self):
        super().__init__('balance_agent')

        # Balance-specific parameters
        self.com_reference_height = 0.85  # Desired CoM height
        self.zmp_tolerance = 0.05         # ZMP position tolerance
        self.balance_kp = 50.0            # Balance proportional gain
        self.balance_kd = 10.0            # Balance derivative gain

        self.get_logger().info('Balance Agent initialized')

    def execute_agent_behavior(self):
        """Execute balance control behavior"""
        # Calculate balance correction based on IMU data
        balance_correction = self.calculate_balance_correction()

        # If robot is unstable, prioritize balance
        if self.agent_state['balance_state'] == 'unstable':
            self.apply_balance_correction(balance_correction)
        else:
            # Otherwise, just maintain balance
            self.maintain_balance(balance_correction)

    def calculate_balance_correction(self):
        """Calculate balance correction torques"""
        # Calculate CoM position from IMU and joint data
        com_pos = self.estimate_com_position()
        com_vel = self.estimate_com_velocity()

        # Calculate ZMP from IMU data
        zmp_pos = self.calculate_zmp()

        # Calculate balance error
        zmp_error = zmp_pos[:2]  # Only consider X,Y for balance
        com_error = com_pos[:2] - np.array([0.0, 0.0])  # Deviation from center

        # Calculate correction torques using PD control
        correction = np.zeros(len(self.agent_state.get('joint_positions', [])))

        # Apply balance correction to appropriate joints
        # In practice, this would use inverse dynamics
        if len(correction) >= 6:  # At least 6 joints
            # Apply correction to hip and ankle joints
            correction[0] = self.balance_kp * zmp_error[0] + self.balance_kd * com_vel[0]  # Hip roll
            correction[1] = self.balance_kp * zmp_error[1] + self.balance_kd * com_vel[1]  # Hip pitch
            correction[4] = -correction[0] * 0.5  # Ankle roll correction
            correction[5] = -correction[1] * 0.5  # Ankle pitch correction

        return correction

    def estimate_com_position(self):
        """Estimate center of mass position"""
        # Simplified CoM estimation
        # In practice, use full kinematic model
        return np.array([0.0, 0.0, self.com_reference_height])

    def calculate_zmp(self):
        """Calculate Zero Moment Point from IMU data"""
        # ZMP = CoM - (CoM_height / gravity) * [linear_acc_x, linear_acc_y, 0]
        gravity = 9.81
        linear_acc = np.array(self.agent_state['linear_acceleration'])
        com_height = self.com_reference_height

        zmp = np.array([
            0.0 - (com_height / gravity) * linear_acc[0],  # X
            0.0 - (com_height / gravity) * linear_acc[1],  # Y
            0.0  # Z (on ground)
        ])

        return zmp

    def apply_balance_correction(self, correction_torques):
        """Apply balance correction torques to joints"""
        if 'joint_positions' in self.agent_state:
            # Create joint command message
            cmd_msg = JointState()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.name = [f'joint_{i}' for i in range(len(correction_torques))]
            cmd_msg.effort = correction_torques.tolist()

            self.joint_cmd_pub.publish(cmd_msg)

    def maintain_balance(self, correction_torques):
        """Maintain balance with minimal corrections"""
        # Apply smaller corrections for maintenance
        scaled_correction = correction_torques * 0.3
        self.apply_balance_correction(scaled_correction)

class LocomotionAgent(HumanoidAgentNode):
    """Agent responsible for walking and locomotion"""

    def __init__(self):
        super().__init__('locomotion_agent')

        # Locomotion parameters
        self.step_height = 0.05
        self.step_length = 0.3
        self.step_duration = 0.8
        self.nominal_com_height = 0.85
        self.walk_frequency = 1.25

        # Walking state
        self.gait_phase = 0.0
        self.left_foot_support = True
        self.step_counter = 0
        self.desired_velocity = np.array([0.0, 0.0, 0.0])

        # Subscribe to velocity commands
        self.vel_cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_command_callback, 10)

        self.get_logger().info('Locomotion Agent initialized')

    def velocity_command_callback(self, msg):
        """Receive velocity commands"""
        self.desired_velocity = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.angular.z
        ])

    def execute_agent_behavior(self):
        """Execute locomotion behavior"""
        if np.linalg.norm(self.desired_velocity) > 0.01:  # If moving
            # Generate walking gait based on desired velocity
            walking_commands = self.generate_walking_gait()

            # Publish walking commands
            self.publish_walking_commands(walking_commands)
        else:
            # Stop walking if no velocity command
            self.stop_walking()

    def generate_walking_gait(self):
        """Generate walking gait based on desired velocity"""
        # Update gait phase
        dt = 0.02  # 50 Hz control loop
        step_dt = 1.0 / self.walk_frequency
        self.gait_phase += dt / step_dt

        if self.gait_phase >= 1.0:
            self.gait_phase = 0.0
            self.step_counter += 1
            self.left_foot_support = not self.left_foot_support  # Alternate support foot

        # Calculate gait commands based on phase and desired velocity
        commands = self.calculate_gait_commands()

        return commands

    def calculate_gait_commands(self):
        """Calculate joint commands for walking gait"""
        # Calculate desired joint positions for walking
        commands = np.zeros(len(self.agent_state.get('joint_positions', [])))

        # Simplified walking gait calculation
        # In practice, use inverse kinematics or predefined gait patterns
        phase = self.gait_phase

        # Calculate swing foot trajectory
        if self.left_foot_support:  # Right foot swinging
            # Right leg gait
            if len(commands) > 5:
                # Calculate right leg joint angles based on gait phase
                commands[3] = self.step_length * 0.5 * np.sin(2 * np.pi * phase)  # Right hip pitch
                commands[4] = self.step_height * 0.5 * (1 - np.cos(2 * np.pi * phase))  # Right knee
                commands[5] = -self.step_height * 0.2 * np.sin(2 * np.pi * phase)  # Right ankle
        else:  # Left foot swinging
            # Left leg gait
            if len(commands) > 2:
                # Calculate left leg joint angles based on gait phase
                commands[0] = self.step_length * 0.5 * np.sin(2 * np.pi * phase)  # Left hip pitch
                commands[1] = self.step_height * 0.5 * (1 - np.cos(2 * np.pi * phase))  # Left knee
                commands[2] = -self.step_height * 0.2 * np.sin(2 * np.pi * phase)  # Left ankle

        # Apply desired forward velocity
        if self.desired_velocity[0] > 0.01:  # Moving forward
            # Scale gait based on desired speed
            speed_factor = min(2.0, self.desired_velocity[0] * 5.0)  # Scale factor
            commands *= speed_factor

        return commands

    def publish_walking_commands(self, commands):
        """Publish walking joint commands"""
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = [f'joint_{i}' for i in range(len(commands))]
        cmd_msg.effort = commands.tolist()

        self.joint_cmd_pub.publish(cmd_msg)

    def stop_walking(self):
        """Stop walking motion"""
        # Publish zero commands to stop movement
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = [f'joint_{i}' for i in range(10)]  # Assuming 10 joints
        cmd_msg.effort = [0.0] * 10

        self.joint_cmd_pub.publish(cmd_msg)

class NavigationAgent(HumanoidAgentNode):
    """Agent responsible for path planning and navigation"""

    def __init__(self):
        super().__init__('navigation_agent')

        # Navigation parameters
        self.path = []
        self.path_index = 0
        self.waypoint_tolerance = 0.3
        self.max_linear_velocity = 0.3
        self.max_angular_velocity = 0.5

        # Subscribe to navigation goals
        self.goal_sub = self.create_subscription(
            Pose, '/navigation_goal', self.navigation_goal_callback, 10)

        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.get_logger().info('Navigation Agent initialized')

    def navigation_goal_callback(self, msg):
        """Receive navigation goal"""
        self.target_position = np.array([
            msg.position.x,
            msg.position.y,
            msg.position.z
        ])

        # Plan path to goal
        self.plan_path_to_goal()

    def plan_path_to_goal(self):
        """Plan path to goal using A* or other algorithm"""
        # In practice, implement path planning algorithm
        # For now, create a simple straight-line path
        current_pos = self.agent_state['position']
        direction = self.target_position - current_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            steps = int(distance / 0.5)  # 0.5m waypoints
            self.path = []

            for i in range(steps + 1):
                fraction = i / steps if steps > 0 else 0
                waypoint = current_pos + direction * fraction
                self.path.append(waypoint)

        self.path_index = 0

    def execute_agent_behavior(self):
        """Execute navigation behavior"""
        if self.path and self.path_index < len(self.path):
            # Follow current path
            self.follow_path()
        else:
            # Stop if no path or path completed
            self.stop_navigation()

    def follow_path(self):
        """Follow planned path"""
        if self.path_index >= len(self.path):
            self.stop_navigation()
            return

        target_waypoint = self.path[self.path_index]
        current_pos = self.agent_state['position']

        # Calculate distance to waypoint
        distance = np.linalg.norm(target_waypoint[:2] - current_pos[:2])

        if distance < self.waypoint_tolerance:
            # Reached waypoint, move to next
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.get_logger().info('Reached destination')
                return

        # Calculate desired velocity towards next waypoint
        direction = target_waypoint - current_pos
        direction[2] = 0  # Ignore height difference for navigation
        direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([0, 0, 0])

        # Calculate desired velocity
        desired_vel = direction * min(self.max_linear_velocity, distance * 0.5)

        # Calculate angular velocity for heading adjustment
        current_yaw = self.get_yaw_from_orientation()
        desired_yaw = np.arctan2(direction[1], direction[0])
        yaw_error = self.normalize_angle(desired_yaw - current_yaw)

        angular_vel = yaw_error * 1.5  # Proportional control

        # Publish velocity command
        cmd_msg = Twist()
        cmd_msg.linear.x = float(desired_vel[0])
        cmd_msg.angular.z = float(angular_vel)

        self.cmd_vel_pub.publish(cmd_msg)

    def get_yaw_from_orientation(self):
        """Extract yaw from orientation quaternion"""
        quat = self.agent_state['orientation']
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion(quat)
        return euler[2]  # yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def stop_navigation(self):
        """Stop navigation"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create agents
    balance_agent = BalanceAgent()
    locomotion_agent = LocomotionAgent()
    navigation_agent = NavigationAgent()

    # Create executor to run all agents
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(balance_agent)
    executor.add_node(locomotion_agent)
    executor.add_node(navigation_agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        balance_agent.destroy_node()
        locomotion_agent.destroy_node()
        navigation_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Decision-Making Framework for Humanoid Robots

### Hierarchical Decision-Making System

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import JointState, Imu
import numpy as np
import json
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class TaskPriority(Enum):
    EMERGENCY = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    name: str
    priority: TaskPriority
    requirements: List[str]  # Required resources/skills
    dependencies: List[str]  # Other tasks that must complete first
    duration: float  # Expected duration in seconds
    success_criteria: List[str]  # Conditions for success
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class HumanoidDecisionMaker(Node):
    """Hierarchical decision-making system for humanoid robots"""

    def __init__(self):
        super().__init__('humanoid_decision_maker')

        # Task management
        self.task_queue = []  # Priority queue of tasks
        self.active_tasks = {}  # Currently executing tasks
        self.completed_tasks = []  # History of completed tasks
        self.failed_tasks = []  # History of failed tasks

        # Robot state
        self.robot_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'joint_positions': [],
            'joint_velocities': [],
            'balance_state': 'stable',
            'battery_level': 100.0,
            'health': 100.0,
            'current_task': None,
            'capabilities': ['balance', 'walk', 'communicate', 'perceive']
        }

        # Decision-making parameters
        self.decision_frequency = 10.0  # Hz
        self.max_concurrent_tasks = 3
        self.task_timeout = 30.0  # seconds

        # Publishers and subscribers
        self.state_sub = self.create_subscription(
            Float64MultiArray, '/robot_state', self.state_callback, 10)
        self.task_request_sub = self.create_subscription(
            String, '/task_requests', self.task_request_callback, 10)
        self.task_status_pub = self.create_publisher(
            String, '/task_status', 10)
        self.decision_pub = self.create_publisher(
            String, '/high_level_commands', 10)

        # Decision timer
        self.decision_timer = self.create_timer(
            1.0/self.decision_frequency, self.decision_cycle)

        self.get_logger().info('Humanoid Decision Maker initialized')

    def state_callback(self, msg):
        """Update robot state from sensor data"""
        try:
            data = msg.data
            if len(data) >= 10:  # Minimum expected data points
                self.robot_state['position'] = np.array(data[0:3])
                self.robot_state['orientation'] = np.array(data[3:7])
                self.robot_state['velocity'] = np.array(data[7:10])

                # Update other state variables as needed
                if len(data) >= 11:
                    self.robot_state['balance_state'] = 'stable' if data[10] > 0.5 else 'unstable'
                if len(data) >= 12:
                    self.robot_state['battery_level'] = data[11]
                if len(data) >= 13:
                    self.robot_state['health'] = data[12]
        except Exception as e:
            self.get_logger().error(f'Error parsing state message: {e}')

    def task_request_callback(self, msg):
        """Handle incoming task requests"""
        try:
            task_data = json.loads(msg.data)
            task = Task(
                id=task_data['id'],
                name=task_data['name'],
                priority=TaskPriority[task_data['priority']],
                requirements=task_data.get('requirements', []),
                dependencies=task_data.get('dependencies', []),
                duration=task_data.get('duration', 10.0),
                success_criteria=task_data.get('success_criteria', [])
            )

            self.add_task(task)
            self.get_logger().info(f'Added task: {task.name} (Priority: {task.priority})')

        except Exception as e:
            self.get_logger().error(f'Error parsing task request: {e}')

    def add_task(self, task):
        """Add a task to the queue"""
        # Insert task based on priority
        inserted = False
        for i, queued_task in enumerate(self.task_queue):
            if task.priority.value < queued_task.priority.value:
                self.task_queue.insert(i, task)
                inserted = True
                break

        if not inserted:
            self.task_queue.append(task)

    def decision_cycle(self):
        """Main decision-making cycle"""
        try:
            # Check for completed/failed tasks
            self.monitor_active_tasks()

            # Schedule new tasks if capacity allows
            self.schedule_tasks()

            # Update task statuses
            self.publish_task_statuses()

        except Exception as e:
            self.get_logger().error(f'Decision cycle error: {e}')

    def monitor_active_tasks(self):
        """Monitor active tasks and update their status"""
        current_time = time.time()

        for task_id, task in list(self.active_tasks.items()):
            # Check if task has timed out
            if task.start_time and (current_time - task.start_time) > self.task_timeout:
                task.status = TaskStatus.FAILED
                self.failed_tasks.append(task)
                del self.active_tasks[task_id]
                self.get_logger().warn(f'Task {task.name} timed out')

            # Check success/failure conditions (simplified)
            elif self.check_task_success(task):
                task.status = TaskStatus.COMPLETED
                task.end_time = current_time
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
                self.get_logger().info(f'Task {task.name} completed successfully')

            elif self.check_task_failure(task):
                task.status = TaskStatus.FAILED
                task.end_time = current_time
                self.failed_tasks.append(task)
                del self.active_tasks[task_id]
                self.get_logger().warn(f'Task {task.name} failed')

    def check_task_success(self, task):
        """Check if task has been successfully completed"""
        # In a real implementation, this would check specific success conditions
        # For now, we'll use a simple simulation
        if task.name == "stand_up":
            # Check if robot is upright and stable
            return (self.robot_state['balance_state'] == 'stable' and
                   abs(self.robot_state['orientation'][1]) < 0.1)  # Small pitch angle
        elif task.name == "walk_forward":
            # Check if robot moved forward
            return self.robot_state['velocity'][0] > 0.1
        elif task.name == "reach_target":
            # Check if robot reached target position
            return np.linalg.norm(self.robot_state['position'] - task.target_position) < 0.5
        else:
            # Default: no specific check
            return False

    def check_task_failure(self, task):
        """Check if task has failed"""
        # Check for failure conditions
        if self.robot_state['balance_state'] == 'unstable':
            return True  # Robot fell
        if self.robot_state['health'] < 10.0:
            return True  # Robot health too low
        if self.robot_state['battery_level'] < 5.0:
            return True  # Battery critically low

        return False

    def schedule_tasks(self):
        """Schedule tasks based on priorities and resource availability"""
        # Check if we can start more tasks
        available_slots = self.max_concurrent_tasks - len(self.active_tasks)

        if available_slots <= 0:
            return

        # Check task dependencies and requirements
        scheduled_count = 0
        for i, task in enumerate(self.task_queue[:]):  # Create copy to iterate safely
            if scheduled_count >= available_slots:
                break

            # Check if all dependencies are satisfied
            if not self.are_dependencies_met(task):
                continue

            # Check if robot has required capabilities
            if not self.has_required_capabilities(task):
                continue

            # Check if resources are available
            if not self.can_schedule_task(task):
                continue

            # Schedule the task
            self.start_task(task)
            self.task_queue.pop(i)  # Remove from queue
            scheduled_count += 1

    def are_dependencies_met(self, task):
        """Check if all task dependencies are satisfied"""
        for dep_id in task.dependencies:
            # Check if dependency is completed
            if not any(t.id == dep_id and t.status == TaskStatus.COMPLETED for t in self.completed_tasks):
                return False
        return True

    def has_required_capabilities(self, task):
        """Check if robot has required capabilities for task"""
        for req in task.requirements:
            if req not in self.robot_state['capabilities']:
                return False
        return True

    def can_schedule_task(self, task):
        """Check if task can be scheduled given current state"""
        # Check resource conflicts
        for active_task in self.active_tasks.values():
            # Check for conflicting resource usage
            if set(task.requirements) & set(active_task.requirements):
                # If both tasks use the same resource, check if they're compatible
                if not self.are_tasks_compatible(task, active_task):
                    return False

        # Check state constraints
        if task.name == "stand_up" and self.robot_state['balance_state'] == 'stable':
            return False  # Already standing

        return True

    def are_tasks_compatible(self, task1, task2):
        """Check if two tasks can run simultaneously"""
        # In a real implementation, this would check for resource conflicts
        # For now, assume most tasks are incompatible
        return False

    def start_task(self, task):
        """Start executing a task"""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        self.active_tasks[task.id] = task

        # Publish command to execute task
        command_msg = String()
        command_msg.data = json.dumps({
            'command': 'execute_task',
            'task_id': task.id,
            'task_name': task.name,
            'priority': task.priority.name
        })
        self.decision_pub.publish(command_msg)

        self.get_logger().info(f'Started task: {task.name}')

    def publish_task_statuses(self):
        """Publish current task statuses"""
        status_msg = String()
        status_data = {
            'active_tasks': [
                {
                    'id': task.id,
                    'name': task.name,
                    'status': task.status.value,
                    'assigned_agent': task.assigned_agent,
                    'progress': self.estimate_task_progress(task)
                }
                for task in self.active_tasks.values()
            ],
            'queue_length': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }
        status_msg.data = json.dumps(status_data)
        self.task_status_pub.publish(status_msg)

    def estimate_task_progress(self, task):
        """Estimate task completion progress (0-1)"""
        if not task.start_time:
            return 0.0

        elapsed = time.time() - task.start_time
        expected_duration = task.duration

        if expected_duration <= 0:
            return 0.0

        progress = min(1.0, elapsed / expected_duration)
        return progress

    def emergency_stop(self):
        """Emergency stop - cancel all tasks"""
        self.get_logger().warn('EMERGENCY STOP: Cancelling all tasks')

        # Cancel all active tasks
        for task_id, task in self.active_tasks.items():
            task.status = TaskStatus.CANCELLED
            self.failed_tasks.append(task)

        self.active_tasks.clear()

        # Clear task queue
        self.task_queue.clear()

        # Publish emergency stop command
        emergency_msg = String()
        emergency_msg.data = json.dumps({'command': 'emergency_stop'})
        self.decision_pub.publish(emergency_msg)

class MissionPlanner(Node):
    """Higher-level mission planning for humanoid robots"""

    def __init__(self):
        super().__init__('mission_planner')

        # Mission parameters
        self.current_mission = None
        self.mission_goals = []
        self.mission_tasks = []
        self.mission_state = 'idle'  # idle, planning, executing, completed, failed

        # Publishers and subscribers
        self.mission_request_sub = self.create_subscription(
            String, '/mission_requests', self.mission_request_callback, 10)
        self.mission_status_pub = self.create_publisher(
            String, '/mission_status', 10)
        self.task_publisher = self.create_publisher(
            String, '/task_requests', 10)

        self.get_logger().info('Mission Planner initialized')

    def mission_request_callback(self, msg):
        """Handle mission requests"""
        try:
            mission_data = json.loads(msg.data)
            mission_type = mission_data['type']
            mission_params = mission_data.get('parameters', {})

            if mission_type == 'navigation':
                self.plan_navigation_mission(mission_params)
            elif mission_type == 'exploration':
                self.plan_exploration_mission(mission_params)
            elif mission_type == 'delivery':
                self.plan_delivery_mission(mission_params)
            else:
                self.get_logger().error(f'Unknown mission type: {mission_type}')
                return

            # Execute the mission
            self.execute_mission()

        except Exception as e:
            self.get_logger().error(f'Error processing mission request: {e}')

    def plan_navigation_mission(self, params):
        """Plan navigation mission"""
        start_pos = params.get('start', [0, 0, 0])
        goal_pos = params['goal']
        waypoints = params.get('waypoints', [])

        # Create navigation tasks
        self.mission_tasks = []

        # Move to first waypoint or directly to goal
        if waypoints:
            for waypoint in waypoints:
                task = {
                    'id': f'nav_to_{waypoint[0]:.1f}_{waypoint[1]:.1f}',
                    'name': f'navigate_to_waypoint_{len(self.mission_tasks)}',
                    'priority': 'MEDIUM',
                    'requirements': ['balance', 'walk', 'perceive'],
                    'dependencies': [self.mission_tasks[-1]['id']] if self.mission_tasks else [],
                    'duration': 30.0,
                    'success_criteria': [f'reach_position_{waypoint}']
                }
                self.mission_tasks.append(task)

        # Final goal navigation
        goal_task = {
            'id': f'nav_to_goal_{goal_pos[0]:.1f}_{goal_pos[1]:.1f}',
            'name': 'navigate_to_goal',
            'priority': 'HIGH',
            'requirements': ['balance', 'walk', 'perceive'],
            'dependencies': [self.mission_tasks[-1]['id']] if self.mission_tasks else [],
            'duration': 60.0,
            'success_criteria': [f'reach_position_{goal_pos}']
        }
        self.mission_tasks.append(goal_task)

        self.mission_state = 'planning'
        self.get_logger().info(f'Navigation mission planned with {len(self.mission_tasks)} tasks')

    def plan_exploration_mission(self, params):
        """Plan exploration mission"""
        area_center = params.get('center', [0, 0, 0])
        area_radius = params.get('radius', 5.0)
        resolution = params.get('resolution', 1.0)

        # Calculate exploration waypoints in a spiral pattern
        waypoints = self.calculate_spiral_waypoints(area_center, area_radius, resolution)

        self.mission_tasks = []
        for i, waypoint in enumerate(waypoints):
            task = {
                'id': f'explore_{i:03d}',
                'name': f'explore_waypoint_{i}',
                'priority': 'MEDIUM',
                'requirements': ['balance', 'walk', 'perceive'],
                'dependencies': [self.mission_tasks[-1]['id']] if self.mission_tasks else [],
                'duration': 20.0,
                'success_criteria': [f'reach_position_{waypoint}', f'collect_sensor_data_at_{waypoint}']
            }
            self.mission_tasks.append(task)

        self.mission_state = 'planning'
        self.get_logger().info(f'Exploration mission planned with {len(self.mission_tasks)} tasks')

    def calculate_spiral_waypoints(self, center, radius, resolution):
        """Calculate waypoints in a spiral pattern"""
        waypoints = []
        angle = 0
        r = resolution

        while r <= radius:
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            waypoints.append([x, y, center[2]])

            # Increment angle and radius
            angle += 0.5  # radians
            r += resolution * 0.5

        return waypoints

    def plan_delivery_mission(self, params):
        """Plan delivery mission"""
        pickup_location = params['pickup']
        delivery_location = params['delivery']
        item_description = params.get('item', 'unknown')

        self.mission_tasks = [
            {
                'id': 'navigate_to_pickup',
                'name': 'navigate_to_pickup_location',
                'priority': 'HIGH',
                'requirements': ['balance', 'walk', 'perceive'],
                'dependencies': [],
                'duration': 60.0,
                'success_criteria': [f'reach_position_{pickup_location}']
            },
            {
                'id': 'pickup_item',
                'name': 'pickup_item',
                'priority': 'HIGH',
                'requirements': ['balance', 'manipulate'],
                'dependencies': ['navigate_to_pickup'],
                'duration': 30.0,
                'success_criteria': [f'pickup_{item_description}']
            },
            {
                'id': 'navigate_to_delivery',
                'name': 'navigate_to_delivery_location',
                'priority': 'HIGH',
                'requirements': ['balance', 'walk', 'perceive'],
                'dependencies': ['pickup_item'],
                'duration': 60.0,
                'success_criteria': [f'reach_position_{delivery_location}']
            },
            {
                'id': 'deliver_item',
                'name': 'deliver_item',
                'priority': 'HIGH',
                'requirements': ['balance', 'manipulate'],
                'dependencies': ['navigate_to_delivery'],
                'duration': 30.0,
                'success_criteria': [f'deliver_{item_description}']
            }
        ]

        self.mission_state = 'planning'
        self.get_logger().info(f'Delivery mission planned with {len(self.mission_tasks)} tasks')

    def execute_mission(self):
        """Execute the planned mission"""
        if not self.mission_tasks:
            self.get_logger().warn('No tasks to execute')
            return

        self.mission_state = 'executing'
        self.get_logger().info(f'Executing mission with {len(self.mission_tasks)} tasks')

        # Publish tasks to decision maker
        for task_data in self.mission_tasks:
            task_msg = String()
            task_msg.data = json.dumps(task_data)
            self.task_publisher.publish(task_msg)

    def publish_mission_status(self):
        """Publish mission status"""
        status_msg = String()
        status_data = {
            'mission_state': self.mission_state,
            'total_tasks': len(self.mission_tasks),
            'completed_tasks': 0,  # Would be updated based on actual task completion
            'remaining_tasks': len(self.mission_tasks)
        }
        status_msg.data = json.dumps(status_data)
        self.mission_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create decision-making nodes
    decision_maker = HumanoidDecisionMaker()
    mission_planner = MissionPlanner()

    # Create executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(decision_maker)
    executor.add_node(mission_planner)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        decision_maker.destroy_node()
        mission_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with ROS 2 Control

### AI Agent Integration with ROS 2 Control

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray, String
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import torch
import torch.nn as nn
import threading
import time
from collections import deque

class AIControlIntegration(Node):
    """Integration node for AI agents with ROS 2 Control"""

    def __init__(self):
        super().__init__('ai_control_integration')

        # AI model parameters
        self.model_path = self.declare_parameter('model_path', '').value
        self.use_gpu = self.declare_parameter('use_gpu', True).value
        self.control_frequency = self.declare_parameter('control_frequency', 50.0).value

        # Robot state buffers
        self.joint_state_buffer = deque(maxlen=10)
        self.imu_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)

        # Current robot state
        self.current_joint_positions = []
        self.current_joint_velocities = []
        self.current_joint_efforts = []
        self.current_imu_data = None
        self.current_pose = None
        self.current_twist = None

        # Control publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.velocity_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        # State subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        # AI command subscribers
        self.ai_command_sub = self.create_subscription(
            String, '/ai_commands', self.ai_command_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(
            1.0/self.control_frequency, self.ai_control_loop)

        # Initialize AI model
        self.ai_model = self.initialize_ai_model()
        self.model_initialized = self.ai_model is not None

        self.get_logger().info('AI Control Integration initialized')

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.current_joint_positions = list(msg.position)
        self.current_joint_velocities = list(msg.velocity)
        self.current_joint_efforts = list(msg.effort)

        # Add to buffer
        state_data = {
            'position': list(msg.position),
            'velocity': list(msg.velocity),
            'effort': list(msg.effort),
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }
        self.joint_state_buffer.append(state_data)

    def imu_callback(self, msg):
        """Update IMU data"""
        self.current_imu_data = {
            'linear_acceleration': [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ],
            'angular_velocity': [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ],
            'orientation': [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ]
        }

        # Add to buffer
        imu_data = {
            'linear_acceleration': self.current_imu_data['linear_acceleration'],
            'angular_velocity': self.current_imu_data['angular_velocity'],
            'orientation': self.current_imu_data['orientation'],
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }
        self.imu_buffer.append(imu_data)

    def ai_command_callback(self, msg):
        """Handle AI commands"""
        try:
            command_data = msg.data
            # In a real implementation, this would trigger specific AI behaviors
            self.get_logger().info(f'AI command received: {command_data}')
        except Exception as e:
            self.get_logger().error(f'Error processing AI command: {e}')

    def initialize_ai_model(self):
        """Initialize AI model for control"""
        try:
            # In a real implementation, this would load a trained model
            # For now, we'll create a placeholder
            self.get_logger().info('Initializing AI control model')
            return HumanoidControlModel()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize AI model: {e}')
            return None

    def ai_control_loop(self):
        """Main AI control loop"""
        if not self.model_initialized:
            return

        try:
            # Prepare state for AI model
            state_vector = self.prepare_state_vector()

            if state_vector is not None and len(state_vector) > 0:
                # Get action from AI model
                action = self.ai_model.get_action(state_vector)

                # Convert action to robot commands
                joint_commands, velocity_commands = self.convert_action_to_commands(action)

                # Publish commands
                if joint_commands is not None:
                    self.publish_joint_trajectory(joint_commands)

                if velocity_commands is not None:
                    cmd_vel = Twist()
                    cmd_vel.linear.x = float(velocity_commands[0])
                    cmd_vel.angular.z = float(velocity_commands[1])
                    self.velocity_cmd_pub.publish(cmd_vel)

        except Exception as e:
            self.get_logger().error(f'AI control loop error: {e}')

    def prepare_state_vector(self):
        """Prepare state vector for AI model"""
        if (not self.current_joint_positions or
            not self.current_joint_velocities or
            self.current_imu_data is None):
            return None

        state = []

        # Add joint positions (normalized)
        for pos in self.current_joint_positions:
            state.append(pos)

        # Add joint velocities (normalized)
        for vel in self.current_joint_velocities:
            state.append(vel)

        # Add IMU data
        state.extend(self.current_imu_data['linear_acceleration'])
        state.extend(self.current_imu_data['angular_velocity'])
        state.extend(self.current_imu_data['orientation'])

        # Add derived features
        # Balance features
        roll, pitch, yaw = self.quaternion_to_euler(*self.current_imu_data['orientation'])
        state.extend([roll, pitch, yaw])

        # Velocity magnitude
        vel_mag = np.linalg.norm(self.current_joint_velocities[:3]) if len(self.current_joint_velocities) >= 3 else 0.0
        state.append(vel_mag)

        return np.array(state, dtype=np.float32)

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def convert_action_to_commands(self, action):
        """Convert AI action to robot commands"""
        if len(action) < 6:  # Need at least 6 values
            return None, None

        # Split action into joint commands and velocity commands
        num_joints = min(len(self.current_joint_positions), len(action) - 2)
        joint_commands = action[:num_joints]
        velocity_commands = action[num_joints:num_joints+2]  # linear_x, angular_z

        return joint_commands, velocity_commands

    def publish_joint_trajectory(self, joint_commands):
        """Publish joint trajectory commands"""
        if not joint_commands or len(joint_commands) == 0:
            return

        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = 'base_link'

        # Set joint names (assuming standard humanoid joint names)
        traj_msg.joint_names = [f'joint_{i}' for i in range(len(joint_commands))]

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [float(cmd) for cmd in joint_commands]
        point.velocities = [0.0] * len(joint_commands)  # Zero velocities for now
        point.accelerations = [0.0] * len(joint_commands)  # Zero accelerations
        point.time_from_start = Duration(sec=0, nanosec=20000000)  # 20ms

        traj_msg.points = [point]
        self.joint_trajectory_pub.publish(traj_msg)

class HumanoidControlModel:
    """Placeholder for humanoid control AI model"""

    def __init__(self):
        # In a real implementation, this would load a trained neural network
        self.input_dim = 30  # Example input dimension
        self.output_dim = 20  # Example output dimension (joints + velocities)
        self.model = self.build_model()

    def build_model(self):
        """Build the control model"""
        # This is a placeholder - in practice, you'd load a trained model
        return None

    def get_action(self, state):
        """Get action from model given state"""
        # Placeholder implementation - in practice, this would run inference
        # For now, return a random action scaled to reasonable ranges
        action = np.random.randn(self.output_dim).astype(np.float32)

        # Scale to reasonable ranges
        # Joint positions: limit to reasonable ranges
        action[:10] = np.tanh(action[:10]) * 0.5  # Joint position changes
        # Velocities: limit to reasonable ranges
        action[10:12] = np.tanh(action[10:12]) * 0.3  # Linear and angular velocities

        return action

def main(args=None):
    rclpy.init(args=args)
    ai_integration = AIControlIntegration()

    try:
        rclpy.spin(ai_integration)
    except KeyboardInterrupt:
        pass
    finally:
        ai_integration.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization and Safety

### Safety Monitor for AI-Controlled Humanoid

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64MultiArray
from builtin_interfaces.msg import Time
import numpy as np
import threading
import time

class HumanoidSafetyMonitor(Node):
    """Safety monitor for AI-controlled humanoid robot"""

    def __init__(self):
        super().__init__('humanoid_safety_monitor')

        # Safety parameters
        self.declare_parameter('max_joint_velocity', 5.0)
        self.declare_parameter('max_joint_torque', 100.0)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 0.5)
        self.declare_parameter('max_lean_angle', 0.5)
        self.declare_parameter('min_com_height', 0.3)
        self.declare_parameter('max_zmp_deviation', 0.3)
        self.declare_parameter('safety_check_frequency', 100.0)

        self.max_joint_velocity = self.get_parameter('max_joint_velocity').value
        self.max_joint_torque = self.get_parameter('max_joint_torque').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.max_lean_angle = self.get_parameter('max_lean_angle').value
        self.min_com_height = self.get_parameter('min_com_height').value
        self.max_zmp_deviation = self.get_parameter('max_zmp_deviation').value
        self.safety_check_frequency = self.get_parameter('safety_check_frequency').value

        # Robot state
        self.current_joint_state = None
        self.current_imu_data = None
        self.current_cmd_vel = None
        self.emergency_stop_active = False

        # Safety statistics
        self.safety_violations = 0
        self.last_safety_check = self.get_clock().now()

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(Float64MultiArray, '/safety_status', 10)
        self.safety_violation_pub = self.create_publisher(String, '/safety_violations', 10)

        # Safety check timer
        self.safety_timer = self.create_timer(
            1.0/self.safety_check_frequency, self.safety_check)

        self.get_logger().info('Humanoid Safety Monitor initialized')

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.current_joint_state = msg

    def imu_callback(self, msg):
        """Update IMU data"""
        self.current_imu_data = msg

    def cmd_vel_callback(self, msg):
        """Update commanded velocity"""
        self.current_cmd_vel = msg

    def safety_check(self):
        """Perform safety check"""
        if self.emergency_stop_active:
            return

        violations = []
        warnings = []

        # Check joint safety
        if self.current_joint_state:
            violations.extend(self.check_joint_safety())

        # Check balance safety
        if self.current_imu_data:
            violations.extend(self.check_balance_safety())

        # Check velocity safety
        if self.current_cmd_vel:
            violations.extend(self.check_velocity_safety())

        # Publish safety status
        self.publish_safety_status(violations, warnings)

        # Handle violations
        if violations:
            self.safety_violations += len(violations)
            self.handle_safety_violations(violations)

    def check_joint_safety(self):
        """Check joint safety violations"""
        violations = []

        # Check joint velocities
        for i, velocity in enumerate(self.current_joint_state.velocity):
            if abs(velocity) > self.max_joint_velocity:
                violations.append(f'Joint {i} velocity exceeded: {velocity:.2f} > {self.max_joint_velocity:.2f}')

        # Check joint efforts (if available)
        if len(self.current_joint_state.effort) > 0:
            for i, effort in enumerate(self.current_joint_state.effort):
                if abs(effort) > self.max_joint_torque:
                    violations.append(f'Joint {i} torque exceeded: {effort:.2f} > {self.max_joint_torque:.2f}')

        # Check joint position limits (if available)
        # This would require joint limits to be defined

        return violations

    def check_balance_safety(self):
        """Check balance safety violations"""
        violations = []

        # Extract orientation from IMU
        orientation = self.current_imu_data.orientation
        # Convert quaternion to Euler angles to check lean angles
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        roll, pitch, yaw = euler

        # Check lean angles
        if abs(roll) > self.max_lean_angle:
            violations.append(f'Roll angle exceeded: {abs(roll):.2f} > {self.max_lean_angle:.2f}')

        if abs(pitch) > self.max_lean_angle:
            violations.append(f'Pitch angle exceeded: {abs(pitch):.2f} > {self.max_lean_angle:.2f}')

        # Calculate ZMP (simplified)
        linear_acc = self.current_imu_data.linear_acceleration
        gravity = 9.81
        com_height = 0.85  # Assumed CoM height

        zmp_x = -com_height * linear_acc.x / gravity
        zmp_y = -com_height * linear_acc.y / gravity

        zmp_magnitude = np.sqrt(zmp_x**2 + zmp_y**2)

        if zmp_magnitude > self.max_zmp_deviation:
            violations.append(f'ZMP deviation exceeded: {zmp_magnitude:.2f} > {self.max_zmp_deviation:.2f}')

        return violations

    def check_velocity_safety(self):
        """Check velocity safety violations"""
        violations = []

        # Check linear velocity
        if abs(self.current_cmd_vel.linear.x) > self.max_linear_velocity:
            violations.append(f'Linear velocity exceeded: {abs(self.current_cmd_vel.linear.x):.2f} > {self.max_linear_velocity:.2f}')

        # Check angular velocity
        if abs(self.current_cmd_vel.angular.z) > self.max_angular_velocity:
            violations.append(f'Angular velocity exceeded: {abs(self.current_cmd_vel.angular.z):.2f} > {self.max_angular_velocity:.2f}')

        return violations

    def handle_safety_violations(self, violations):
        """Handle safety violations"""
        for violation in violations:
            self.get_logger().error(f'Safety violation: {violation}')

        # If severe violations, trigger emergency stop
        severe_violations = [v for v in violations if 'fall' in v.lower() or 'angle' in v.lower()]

        if len(severe_violations) > 2 or len(violations) > 5:
            self.trigger_emergency_stop()
            self.get_logger().fatal('EMERGENCY STOP TRIGGERED due to multiple safety violations')
        elif severe_violations:
            self.get_logger().warn('Safety violations detected - reducing control authority')

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True

        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)

        self.get_logger().warn('Emergency stop activated')

    def publish_safety_status(self, violations, warnings):
        """Publish safety status"""
        status_msg = Float64MultiArray()
        status_msg.data = [
            0.0 if not self.emergency_stop_active else 1.0,  # Emergency stop status
            len(violations),  # Number of violations
            len(warnings),    # Number of warnings
            self.safety_violations  # Total violations since start
        ]
        self.safety_status_pub.publish(status_msg)

        # Publish violation details
        if violations:
            violation_msg = String()
            violation_msg.data = '; '.join(violations)
            self.safety_violation_pub.publish(violation_msg)

def main(args=None):
    rclpy.init(args=args)
    safety_monitor = HumanoidSafetyMonitor()

    try:
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        safety_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered intelligent agents for humanoid control, including:

- Deep reinforcement learning architectures (DDPG, SAC) for humanoid locomotion
- Custom humanoid environment implementation with PyBullet
- Multi-agent systems for coordinated humanoid behavior
- Hierarchical decision-making frameworks
- AI control integration with ROS 2 Control
- Safety monitoring for AI-controlled robots

The implementation provides a comprehensive framework for developing AI agents that can control humanoid robots safely and effectively, with proper safety measures and coordination between different control aspects.

## Exercises

1. Implement a custom reward function for humanoid walking that encourages energy-efficient gait patterns.
2. Design a multi-agent system where one agent handles balance, another handles navigation, and a third handles obstacle avoidance.
3. How would you modify the safety monitor to handle collaborative tasks with humans?

## Code Example: Complete Humanoid AI Control System

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray, Bool, String
import numpy as np
import threading
import time

class CompleteHumanoidAISystem(Node):
    def __init__(self):
        super().__init__('complete_humanoid_ai_system')

        # System state
        self.system_state = {
            'current_joint_positions': [],
            'current_joint_velocities': [],
            'current_imu_data': None,
            'current_pose': None,
            'current_velocity': None,
            'ai_control_active': True,
            'balance_state': 'stable',
            'navigation_goal': None,
            'task_queue': [],
            'safety_violations': 0
        }

        # AI control parameters
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('safety_frequency', 100.0)
        self.declare_parameter('learning_enabled', True)

        self.control_frequency = self.get_parameter('control_frequency').value
        self.safety_frequency = self.get_parameter('safety_frequency').value
        self.learning_enabled = self.get_parameter('learning_enabled').value

        # Initialize AI components
        self.balance_agent = BalanceAgent()
        self.locomotion_agent = LocomotionAgent()
        self.navigation_agent = NavigationAgent()
        self.decision_maker = HumanoidDecisionMaker()
        self.safety_monitor = HumanoidSafetyMonitor()

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

        # Control timer
        self.control_timer = self.create_timer(
            1.0/self.control_frequency, self.ai_control_loop)

        # System status publisher
        self.system_status_pub = self.create_publisher(
            Float64MultiArray, '/system_status', 10)

        self.get_logger().info('Complete Humanoid AI Control System initialized')

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.system_state['current_joint_positions'] = list(msg.position)
        self.system_state['current_joint_velocities'] = list(msg.velocity)

    def imu_callback(self, msg):
        """Update IMU data"""
        self.system_state['current_imu_data'] = msg

    def ai_control_loop(self):
        """Main AI control loop"""
        if not self.system_state['ai_control_active']:
            return

        try:
            # Get AI control commands from various agents
            balance_commands = self.balance_agent.get_balance_commands(
                self.system_state)
            locomotion_commands = self.locomotion_agent.get_locomotion_commands(
                self.system_state)
            navigation_commands = self.navigation_agent.get_navigation_commands(
                self.system_state)

            # Fuse commands from different agents
            fused_commands = self.fuse_agent_commands(
                balance_commands, locomotion_commands, navigation_commands)

            # Apply safety constraints
            safe_commands = self.apply_safety_constraints(fused_commands)

            # Publish commands
            self.publish_commands(safe_commands)

            # Update system status
            self.publish_system_status()

        except Exception as e:
            self.get_logger().error(f'AI control loop error: {e}')
            self.system_state['safety_violations'] += 1

    def fuse_agent_commands(self, balance_cmd, loco_cmd, nav_cmd):
        """Fuse commands from different agents"""
        # Simple weighted fusion - in practice, use more sophisticated methods
        fused_cmd = {
            'joint_positions': balance_cmd.get('joint_positions', [])[:5] +
                              loco_cmd.get('joint_positions', [])[5:],
            'linear_velocity': 0.7 * loco_cmd.get('linear_velocity', 0.0) +
                              0.3 * nav_cmd.get('linear_velocity', 0.0),
            'angular_velocity': 0.5 * loco_cmd.get('angular_velocity', 0.0) +
                               0.5 * nav_cmd.get('angular_velocity', 0.0)
        }

        return fused_cmd

    def apply_safety_constraints(self, commands):
        """Apply safety constraints to commands"""
        # Limit velocities
        commands['linear_velocity'] = max(-0.5, min(0.5, commands['linear_velocity']))
        commands['angular_velocity'] = max(-0.5, min(0.5, commands['angular_velocity']))

        # Limit joint positions based on safety monitor
        if 'joint_positions' in commands:
            for i, pos in enumerate(commands['joint_positions']):
                commands['joint_positions'][i] = max(-1.5, min(1.5, pos))  # Example limits

        return commands

    def publish_commands(self, commands):
        """Publish AI-generated commands"""
        if 'linear_velocity' in commands and 'angular_velocity' in commands:
            # Publish velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = float(commands['linear_velocity'])
            cmd_vel.angular.z = float(commands['angular_velocity'])
            self.cmd_vel_pub.publish(cmd_vel)

        if 'joint_positions' in commands:
            # Publish joint commands
            joint_cmd = JointState()
            joint_cmd.header.stamp = self.get_clock().now().to_msg()
            joint_cmd.name = [f'joint_{i}' for i in range(len(commands['joint_positions']))]
            joint_cmd.position = [float(pos) for pos in commands['joint_positions']]
            self.joint_cmd_pub.publish(joint_cmd)

    def publish_system_status(self):
        """Publish system status"""
        status_msg = Float64MultiArray()
        status_msg.data = [
            1.0 if self.system_state['ai_control_active'] else 0.0,
            1.0 if self.system_state['balance_state'] == 'stable' else 0.0,
            self.system_state['safety_violations'],
            len(self.system_state['task_queue']),
            self.system_state['current_joint_positions'][0] if self.system_state['current_joint_positions'] else 0.0,
            self.system_state['current_joint_positions'][1] if len(self.system_state['current_joint_positions']) > 1 else 0.0
        ]
        self.system_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    ai_system = CompleteHumanoidAISystem()

    try:
        rclpy.spin(ai_system)
    except KeyboardInterrupt:
        pass
    finally:
        ai_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```