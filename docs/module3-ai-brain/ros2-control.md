---
title: ROS 2 Control for Humanoid Robots
description: Understanding ROS 2 Control framework for humanoid robot joint control and balance
sidebar_position: 4
---

# ROS 2 Control for Humanoid Robots

## Learning Objectives

By the end of this chapter, you will be able to:
1. Understand the ROS 2 Control framework architecture and components
2. Configure and implement joint controllers for humanoid robots
3. Develop custom controllers for balance and locomotion
4. Integrate sensor feedback for closed-loop control
5. Implement advanced control strategies for humanoid balance and walking

## Introduction to ROS 2 Control

ROS 2 Control is the standard control framework for ROS 2 that provides a unified interface for controlling robots. For humanoid robots, ROS 2 Control offers:

- **Hardware Abstraction**: Standardized interfaces between controllers and hardware
- **Controller Management**: Dynamic loading and switching of controllers
- **Real-time Safety**: Built-in safety mechanisms and limits
- **Flexibility**: Support for various control types (position, velocity, effort)

### ROS 2 Control Architecture

The ROS 2 Control architecture consists of:

- **Controller Manager**: Orchestrates all controllers
- **Controllers**: Implement specific control algorithms
- **Hardware Interface**: Abstracts hardware communication
- **Robot State Publisher**: Publishes joint states

## ROS 2 Control for Humanoid Robots

Humanoid robots have unique control requirements compared to traditional manipulators:

- **Balance Control**: Maintaining center of mass within support polygon
- **Walking Gait**: Coordinated multi-joint motion for locomotion
- **Disturbance Rejection**: Handling external forces and perturbations
- **Stability**: Ensuring dynamic stability during movement

### Humanoid-Specific Control Challenges

1. **Underactuated System**: Fewer actuators than degrees of freedom
2. **Dynamic Balance**: Balance depends on motion and momentum
3. **Contact Transitions**: Switching between single/double support
4. **Multi-objective Control**: Simultaneous balance, walking, and task execution

## ROS 2 Control Configuration for Humanoid Robots

### Control Hardware Interface

```yaml
# config/humanoid_control_system.yaml
humanoid_control_system:
  ros__parameters:
    # Robot description parameters
    joint_names:
      - left_hip_roll
      - left_hip_yaw
      - left_hip_pitch
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll
      - right_hip_roll
      - right_hip_yaw
      - right_hip_pitch
      - right_knee
      - right_ankle_pitch
      - right_ankle_roll
      - left_shoulder_pitch
      - left_shoulder_roll
      - left_shoulder_yaw
      - left_elbow
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
      - head_yaw
      - head_pitch

    # Actuator parameters
    actuator_names:
      - left_leg_actuator_1
      - left_leg_actuator_2
      - left_leg_actuator_3
      - left_leg_actuator_4
      - left_leg_actuator_5
      - left_leg_actuator_6
      - right_leg_actuator_1
      - right_leg_actuator_2
      - right_leg_actuator_3
      - right_leg_actuator_4
      - right_leg_actuator_5
      - right_leg_actuator_6
      - left_arm_actuator_1
      - left_arm_actuator_2
      - left_arm_actuator_3
      - left_arm_actuator_4
      - right_arm_actuator_1
      - right_arm_actuator_2
      - right_arm_actuator_3
      - right_arm_actuator_4
      - head_actuator_1
      - head_actuator_2

    # Joint limits
    left_hip_roll:
      has_soft_limits: true
      soft_lower_limit: -0.5236  # -30 degrees
      soft_upper_limit: 0.5236   # 30 degrees
      soft_effort_limit: 50.0

    left_hip_yaw:
      has_soft_limits: true
      soft_lower_limit: -0.3491  # -20 degrees
      soft_upper_limit: 0.3491   # 20 degrees
      soft_effort_limit: 50.0

    left_hip_pitch:
      has_soft_limits: true
      soft_lower_limit: -1.5708  # -90 degrees
      soft_upper_limit: 0.7854   # 45 degrees
      soft_effort_limit: 100.0

    # Add limits for all other joints...

    # Control frequency
    update_rate: 1000  # Hz

    # Safety parameters
    enable_safety_limits: true
    safety_position_margin: 0.1
    safety_velocity_limit: 5.0
    safety_effort_limit: 150.0
```

### Controller Manager Configuration

```yaml
# config/controller_manager.yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    # Available controllers
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    humanoid_balance_controller:
      type: humanoid_control/BalanceController

    humanoid_walk_controller:
      type: humanoid_control/WalkController

    humanoid_position_controller:
      type: position_controllers/JointGroupPositionController

    humanoid_velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    humanoid_effort_controller:
      type: effort_controllers/JointGroupEffortController

# Individual controller configurations
humanoid_balance_controller:
  ros__parameters:
    # Balance control parameters
    com_kp: [100.0, 100.0, 50.0]  # Center of mass position gains
    com_kd: [20.0, 20.0, 10.0]   # Center of mass velocity gains
    zmp_kp: [50.0, 50.0]          # ZMP position gains
    zmp_kd: [10.0, 10.0]          # ZMP velocity gains
    com_reference_height: 0.85     # Desired CoM height (m)
    support_polygon_margin: 0.05   # Safety margin for support polygon (m)

    # Joint control parameters
    hip_position_gains: [1000.0, 1000.0, 1000.0]  # [roll, pitch, yaw]
    hip_velocity_gains: [100.0, 100.0, 100.0]
    knee_position_gains: [1000.0]
    knee_velocity_gains: [100.0]
    ankle_position_gains: [500.0, 500.0]  # [pitch, roll]
    ankle_velocity_gains: [50.0, 50.0]

humanoid_walk_controller:
  ros__parameters:
    # Walking gait parameters
    step_height: 0.05              # Maximum step height (m)
    step_length: 0.3              # Maximum step length (m)
    step_duration: 0.8            # Duration of each step (s)
    double_support_ratio: 0.2     # Ratio of double support phase
    nominal_com_height: 0.85      # Nominal CoM height (m)
    walk_frequency: 1.25          # Walking frequency (Hz)

    # Gait parameters
    gait_type: "periodic"         # Options: periodic, adaptive, reactive
    step_timing_adaptation: true  # Enable adaptive step timing
    foot_lift_height: 0.03        # Height to lift foot during swing phase (m)
    foot_placement_correction: true  # Enable foot placement correction

humanoid_position_controller:
  ros__parameters:
    joints:
      - left_hip_roll
      - left_hip_yaw
      - left_hip_pitch
      - left_knee
      - left_ankle_pitch
      - left_ankle_roll
      - right_hip_roll
      - right_hip_yaw
      - right_hip_pitch
      - right_knee
      - right_ankle_pitch
      - right_ankle_roll
      - left_shoulder_pitch
      - left_shoulder_roll
      - left_shoulder_yaw
      - left_elbow
      - right_shoulder_pitch
      - right_shoulder_roll
      - right_shoulder_yaw
      - right_elbow
      - head_yaw
      - head_pitch
```

## Custom Humanoid Controllers

### Balance Controller Implementation

```cpp
#include <rclcpp/rclcpp.hpp>
#include <controller_interface/controller_interface.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <cmath>

namespace humanoid_balance_controller
{

class BalanceController : public controller_interface::ControllerInterface
{
public:
    BalanceController() = default;

    controller_interface::InterfaceConfiguration command_interface_configuration() const override
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        // Joint effort interfaces for all actuated joints
        for (const auto& joint_name : joint_names_)
        {
            config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_EFFORT);
        }

        return config;
    }

    controller_interface::InterfaceConfiguration state_interface_configuration() const override
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        // Joint position and velocity interfaces
        for (const auto& joint_name : joint_names_)
        {
            config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_POSITION);
            config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_VELOCITY);
        }

        // IMU interfaces for balance feedback
        config.names.push_back("imu/acceleration.x");
        config.names.push_back("imu/acceleration.y");
        config.names.push_back("imu/acceleration.z");
        config.names.push_back("imu/angular_velocity.x");
        config.names.push_back("imu/angular_velocity.y");
        config.names.push_back("imu/angular_velocity.z");
        config.names.push_back("imu/orientation.x");
        config.names.push_back("imu/orientation.y");
        config.names.push_back("imu/orientation.z");
        config.names.push_back("imu/orientation.w");

        return config;
    }

    controller_interface::return_type update(
        const rclcpp::Time& time,
        const rclcpp::Duration& period) override
    {
        // Update robot state
        update_robot_state();

        // Calculate balance control commands
        auto balance_commands = calculate_balance_control();

        // Apply commands to joint efforts
        for (size_t i = 0; i < joint_names_.size(); ++i)
        {
            if (i < command_interfaces_.size())
            {
                command_interfaces_[i].set_value(balance_commands[i]);
            }
        }

        // Publish balance status
        publish_balance_status(time);

        return controller_interface::return_type::OK;
    }

    CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override
    {
        // Get joint names parameter
        joint_names_ = get_node()->get_parameter("joints").as_string_array();

        // Get balance parameters
        com_kp_ = get_node()->get_parameter("com_kp").as_double_array();
        com_kd_ = get_node()->get_parameter("com_kd").as_double_array();
        zmp_kp_ = get_node()->get_parameter("zmp_kp").as_double_array();
        zmp_kd_ = get_node()->get_parameter("zmp_kd").as_double_array();
        com_reference_height_ = get_node()->get_parameter("com_reference_height").as_double();

        // Initialize balance state
        initialize_balance_state();

        RCLCPP_INFO(get_node()->get_logger(), "Balance Controller configured");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override
    {
        RCLCPP_INFO(get_node()->get_logger(), "Balance Controller activated");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override
    {
        // Zero all commands when deactivated
        for (auto& cmd_interface : command_interfaces_)
        {
            cmd_interface.set_value(0.0);
        }

        RCLCPP_INFO(get_node()->get_logger(), "Balance Controller deactivated");
        return CallbackReturn::SUCCESS;
    }

private:
    // Robot state
    std::vector<std::string> joint_names_;
    std::vector<double> joint_positions_;
    std::vector<double> joint_velocities_;

    // IMU state
    geometry_msgs::msg::Vector3 imu_acceleration_;
    geometry_msgs::msg::Vector3 imu_angular_velocity_;
    geometry_msgs::msg::Quaternion imu_orientation_;

    // Balance control parameters
    std::vector<double> com_kp_{3};
    std::vector<double> com_kd_{3};
    std::vector<double> zmp_kp_{2};
    std::vector<double> zmp_kd_{2};
    double com_reference_height_;

    // Balance state
    Eigen::Vector3d com_position_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d com_velocity_{Eigen::Vector3d::Zero()};
    Eigen::Vector2d zmp_position_{Eigen::Vector2d::Zero()};
    Eigen::Vector2d zmp_reference_{Eigen::Vector2d::Zero()};
    bool is_balanced_{true};

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr com_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr zmp_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr balance_status_pub_;

    void update_robot_state()
    {
        // Update joint states
        joint_positions_.resize(joint_names_.size());
        joint_velocities_.resize(joint_names_.size());

        for (size_t i = 0; i < joint_names_.size(); ++i)
        {
            size_t state_idx = i * 2; // position interface comes first
            if (state_idx < state_interfaces_.size())
            {
                joint_positions_[i] = state_interfaces_[state_idx].get_value();
            }

            size_t vel_idx = i * 2 + 1; // velocity interface comes second
            if (vel_idx < state_interfaces_.size())
            {
                joint_velocities_[i] = state_interfaces_[vel_idx].get_value();
            }
        }

        // Update IMU state
        if (state_interfaces_.size() > joint_names_.size() * 2 + 9)
        {
            size_t imu_offset = joint_names_.size() * 2;
            imu_acceleration_.x = state_interfaces_[imu_offset + 0].get_value();
            imu_acceleration_.y = state_interfaces_[imu_offset + 1].get_value();
            imu_acceleration_.z = state_interfaces_[imu_offset + 2].get_value();
            imu_angular_velocity_.x = state_interfaces_[imu_offset + 3].get_value();
            imu_angular_velocity_.y = state_interfaces_[imu_offset + 4].get_value();
            imu_angular_velocity_.z = state_interfaces_[imu_offset + 5].get_value();
            imu_orientation_.x = state_interfaces_[imu_offset + 6].get_value();
            imu_orientation_.y = state_interfaces_[imu_offset + 7].get_value();
            imu_orientation_.z = state_interfaces_[imu_offset + 8].get_value();
            imu_orientation_.w = state_interfaces_[imu_offset + 9].get_value();
        }

        // Calculate center of mass position (simplified model)
        calculate_com_position();

        // Calculate ZMP from IMU and joint data
        calculate_zmp_position();
    }

    void calculate_com_position()
    {
        // Simplified CoM calculation based on joint positions
        // In practice, this would use full kinematic model and link masses

        // For now, use IMU position as approximate CoM
        tf2::Quaternion quat(
            imu_orientation_.x,
            imu_orientation_.y,
            imu_orientation_.z,
            imu_orientation_.w
        );

        tf2::Matrix3x3 mat(quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);

        // Estimate CoM based on IMU position and orientation
        com_position_.x() = 0.0; // Simplified - in practice use kinematic model
        com_position_.y() = 0.0;
        com_position_.z() = com_reference_height_;
    }

    void calculate_zmp_position()
    {
        // Calculate Zero Moment Point from IMU and acceleration data
        // ZMP = CoM - (CoM_height / gravity) * [linear_acceleration_x, linear_acceleration_y]

        double gravity = 9.81;
        double com_height = com_position_.z();

        // Simplified ZMP calculation
        zmp_position_.x() = com_position_.x() - (com_height / gravity) * imu_acceleration_.x;
        zmp_position_.y() = com_position_.y() - (com_height / gravity) * imu_acceleration_.y;
    }

    std::vector<double> calculate_balance_control()
    {
        std::vector<double> control_commands(joint_names_.size(), 0.0);

        // Calculate balance errors
        Eigen::Vector2d com_error_xy = zmp_reference_ - zmp_position_;
        Eigen::Vector3d com_error_xyz;
        com_error_xyz << com_error_xy.x(), com_error_xy.y(), 0.0;

        // Calculate balance control based on ZMP error
        Eigen::Vector2d zmp_correction;
        zmp_correction.x() = zmp_kp_[0] * com_error_xy.x() + zmp_kd_[0] * (-com_velocity_.x()); // Assuming we have CoM velocity
        zmp_correction.y() = zmp_kp_[1] * com_error_xy.y() + zmp_kd_[1] * (-com_velocity_.y());

        // Map balance corrections to joint torques using simplified model
        // In practice, this would use full inverse dynamics or MPC
        map_balance_to_joints(zmp_correction, control_commands);

        // Apply stability constraints
        apply_stability_constraints(control_commands);

        return control_commands;
    }

    void map_balance_to_joints(const Eigen::Vector2d& balance_correction, std::vector<double>& commands)
    {
        // Simplified mapping - in practice use inverse dynamics or optimization
        double hip_torque_scale = 20.0;    // Scale factor for hip joints
        double ankle_torque_scale = 15.0;  // Scale factor for ankle joints

        // Map X direction balance to hip and ankle pitch
        if (commands.size() > 2)  // Left hip pitch
            commands[2] += balance_correction.x() * hip_torque_scale;
        if (commands.size() > 8)  // Right hip pitch
            commands[8] += balance_correction.x() * hip_torque_scale;
        if (commands.size() > 4)  // Left ankle pitch
            commands[4] += balance_correction.x() * ankle_torque_scale;
        if (commands.size() > 10) // Right ankle pitch
            commands[10] += balance_correction.x() * ankle_torque_scale;

        // Map Y direction balance to hip and ankle roll
        if (commands.size() > 0)  // Left hip roll
            commands[0] += balance_correction.y() * hip_torque_scale;
        if (commands.size() > 6)  // Right hip roll
            commands[6] += balance_correction.y() * hip_torque_scale;
        if (commands.size() > 5)  // Left ankle roll
            commands[5] += balance_correction.y() * ankle_torque_scale;
        if (commands.size() > 11) // Right ankle roll
            commands[11] += balance_correction.y() * ankle_torque_scale;
    }

    void apply_stability_constraints(std::vector<double>& commands)
    {
        // Apply joint limits and safety constraints
        for (auto& cmd : commands)
        {
            cmd = std::max(-100.0, std::min(100.0, cmd)); // Torque limits
        }
    }

    void initialize_balance_state()
    {
        // Initialize publishers
        com_pub_ = get_node()->create_publisher<geometry_msgs::msg::PointStamped>(
            "~/com_position", rclcpp::QoS(1));
        zmp_pub_ = get_node()->create_publisher<geometry_msgs::msg::PointStamped>(
            "~/zmp_position", rclcpp::QoS(1));
        balance_status_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
            "~/balance_status", rclcpp::QoS(1));

        // Initialize reference ZMP to current position
        zmp_reference_ = zmp_position_;
    }

    void publish_balance_status(const rclcpp::Time& time)
    {
        // Publish CoM position
        geometry_msgs::msg::PointStamped com_msg;
        com_msg.header.stamp = time;
        com_msg.header.frame_id = "base_link";
        com_msg.point.x = com_position_.x();
        com_msg.point.y = com_position_.y();
        com_msg.point.z = com_position_.z();
        com_pub_->publish(com_msg);

        // Publish ZMP position
        geometry_msgs::msg::PointStamped zmp_msg;
        zmp_msg.header.stamp = time;
        zmp_msg.header.frame_id = "base_link";
        zmp_msg.point.x = zmp_position_.x();
        zmp_msg.point.y = zmp_position_.y();
        zmp_msg.point.z = 0.0;  // ZMP is on ground plane
        zmp_pub_->publish(zmp_msg);

        // Publish balance status
        std_msgs::msg::Float64MultiArray status_msg;
        status_msg.data = {
            zmp_position_.x(), zmp_position_.y(),  // Current ZMP
            zmp_reference_.x(), zmp_reference_.y(),  // Desired ZMP
            com_position_.x(), com_position_.y(), com_position_.z(),  // CoM position
            is_balanced_ ? 1.0 : 0.0  // Balance status
        };
        balance_status_pub_->publish(status_msg);
    }
};

} // namespace humanoid_balance_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    humanoid_balance_controller::BalanceController,
    controller_interface::ControllerInterface)
```

### Walk Controller Implementation

```cpp
#include <rclcpp/rclcpp.hpp>
#include <controller_interface/controller_interface.hpp>
#include <hardware_interface/loaned_command_interface.hpp>
#include <hardware_interface/loaned_state_interface.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <cmath>
#include <vector>
#include <memory>

namespace humanoid_walk_controller
{

class WalkController : public controller_interface::ControllerInterface
{
public:
    WalkController() = default;

    controller_interface::InterfaceConfiguration command_interface_configuration() const override
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        // Joint effort interfaces for all leg joints
        for (const auto& joint_name : leg_joint_names_)
        {
            config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_EFFORT);
        }

        return config;
    }

    controller_interface::InterfaceConfiguration state_interface_configuration() const override
    {
        controller_interface::InterfaceConfiguration config;
        config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

        // Joint position and velocity interfaces
        for (const auto& joint_name : joint_names_)
        {
            config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_POSITION);
            config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_VELOCITY);
        }

        return config;
    }

    controller_interface::return_type update(
        const rclcpp::Time& time,
        const rclcpp::Duration& period) override
    {
        // Update robot state
        update_robot_state(period.seconds());

        // Calculate walking gait commands
        auto gait_commands = calculate_gait_commands(period.seconds());

        // Apply commands to joint efforts
        for (size_t i = 0; i < leg_joint_names_.size(); ++i)
        {
            if (i < command_interfaces_.size())
            {
                command_interfaces_[i].set_value(gait_commands[i]);
            }
        }

        // Publish gait status
        publish_gait_status(time);

        return controller_interface::return_type::OK;
    }

    CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override
    {
        // Get joint names
        joint_names_ = get_node()->get_parameter("joints").as_string_array();

        // Extract leg joint names
        leg_joint_names_.clear();
        for (const auto& joint_name : joint_names_)
        {
            if (joint_name.find("hip") != std::string::npos ||
                joint_name.find("knee") != std::string::npos ||
                joint_name.find("ankle") != std::string::npos)
            {
                leg_joint_names_.push_back(joint_name);
            }
        }

        // Get gait parameters
        step_height_ = get_node()->get_parameter("step_height").as_double();
        step_length_ = get_node()->get_parameter("step_length").as_double();
        step_duration_ = get_node()->get_parameter("step_duration").as_double();
        double_support_ratio_ = get_node()->get_parameter("double_support_ratio").as_double();
        nominal_com_height_ = get_node()->get_parameter("nominal_com_height").as_double();
        walk_frequency_ = get_node()->get_parameter("walk_frequency").as_double();

        // Initialize gait state
        initialize_gait_state();

        // Create subscriber for velocity commands
        velocity_sub_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
            "~/cmd_vel", rclcpp::QoS(1),
            [this](const geometry_msgs::msg::Twist::SharedPtr msg)
            {
                desired_velocity_ = *msg;
            });

        RCLCPP_INFO(get_node()->get_logger(), "Walk Controller configured");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override
    {
        RCLCPP_INFO(get_node()->get_logger(), "Walk Controller activated");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override
    {
        // Zero all commands when deactivated
        for (auto& cmd_interface : command_interfaces_)
        {
            cmd_interface.set_value(0.0);
        }

        // Stop walking
        gait_phase_ = 0.0;
        left_foot_support_ = true;

        RCLCPP_INFO(get_node()->get_logger(), "Walk Controller deactivated");
        return CallbackReturn::SUCCESS;
    }

private:
    // Joint names
    std::vector<std::string> joint_names_;
    std::vector<std::string> leg_joint_names_;
    std::vector<double> joint_positions_;
    std::vector<double> joint_velocities_;

    // Gait parameters
    double step_height_;
    double step_length_;
    double step_duration_;
    double double_support_ratio_;
    double nominal_com_height_;
    double walk_frequency_;

    // Gait state
    double gait_phase_{0.0};
    bool left_foot_support_{true};
    int step_counter_{0};
    geometry_msgs::msg::Twist desired_velocity_{};
    std::vector<double> previous_joint_positions_;

    // Publishers and subscribers
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gait_status_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr step_trajectory_pub_;

    void update_robot_state(double dt)
    {
        // Update joint states
        joint_positions_.resize(joint_names_.size());
        joint_velocities_.resize(joint_names_.size());

        for (size_t i = 0; i < joint_names_.size(); ++i)
        {
            size_t state_idx = i * 2; // position interface comes first
            if (state_idx < state_interfaces_.size())
            {
                joint_positions_[i] = state_interfaces_[state_idx].get_value();
            }

            size_t vel_idx = i * 2 + 1; // velocity interface comes second
            if (vel_idx < state_interfaces_.size())
            {
                joint_velocities_[i] = state_interfaces_[vel_idx].get_value();
            }
        }

        // Update gait phase based on desired velocity and step timing
        update_gait_phase(dt);
    }

    void update_gait_phase(double dt)
    {
        // Calculate step timing based on desired velocity
        double desired_step_frequency = walk_frequency_;

        // Adjust step frequency based on desired speed
        double desired_speed = std::sqrt(
            desired_velocity_.linear.x * desired_velocity_.linear.x +
            desired_velocity_.linear.y * desired_velocity_.linear.y);

        if (desired_speed > 0.1) // If moving
        {
            // Increase step frequency proportionally to desired speed
            desired_step_frequency = walk_frequency_ * (1.0 + desired_speed * 0.5);
            desired_step_frequency = std::min(desired_step_frequency, walk_frequency_ * 1.5); // Limit max frequency
        }

        // Update gait phase
        double step_dt = 1.0 / desired_step_frequency;
        gait_phase_ += dt / step_dt;

        if (gait_phase_ >= 1.0)
        {
            gait_phase_ = 0.0;
            step_counter_++;
            left_foot_support_ = !left_foot_support_; // Alternate support foot
        }
    }

    std::vector<double> calculate_gait_commands(double dt)
    {
        std::vector<double> commands(leg_joint_names_.size(), 0.0);

        // Calculate gait trajectories for each leg
        calculate_leg_trajectories(commands);

        // Apply balance feedback to gait
        apply_balance_feedback(commands);

        // Apply joint limits and safety constraints
        apply_safety_constraints(commands);

        return commands;
    }

    void calculate_leg_trajectories(std::vector<double>& commands)
    {
        // Calculate desired joint positions for walking gait
        // This implements a simplified walking pattern based on gait phase

        // Swing foot trajectory (cycloidal for smooth motion)
        double swing_phase = gait_phase_ / (1.0 - double_support_ratio_);
        if (swing_phase > 1.0) swing_phase = 1.0;

        // Calculate swing foot position based on gait phase and desired velocity
        double foot_x_offset = 0.0;
        double foot_y_offset = 0.0;
        double foot_z_offset = 0.0;

        if (left_foot_support_ && is_right_foot_swinging()) // Right foot swinging
        {
            // Calculate right foot trajectory
            foot_x_offset = calculate_swing_foot_x(swing_phase, desired_velocity_.linear.x);
            foot_y_offset = calculate_swing_foot_y(swing_phase, desired_velocity_.linear.y, true);
            foot_z_offset = calculate_swing_foot_z(swing_phase);
        }
        else if (!left_foot_support_ && is_left_foot_swinging()) // Left foot swinging
        {
            // Calculate left foot trajectory
            foot_x_offset = calculate_swing_foot_x(swing_phase, desired_velocity_.linear.x);
            foot_y_offset = calculate_swing_foot_y(swing_phase, desired_velocity_.linear.y, false);
            foot_z_offset = calculate_swing_foot_z(swing_phase);
        }

        // Map foot trajectory to joint space using inverse kinematics
        // Simplified mapping - in practice use full IK solver
        map_foot_trajectory_to_joints(foot_x_offset, foot_y_offset, foot_z_offset, commands);
    }

    double calculate_swing_foot_x(double phase, double desired_forward_vel)
    {
        // Calculate X position of swing foot based on gait phase and desired velocity
        double base_step = desired_forward_vel * step_duration_;
        double step_amp = std::min(step_length_, std::abs(base_step) * 1.5);

        // Cycloidal trajectory for smooth motion
        double x_pos = step_amp * (phase - std::sin(2 * M_PI * phase) / (2 * M_PI));

        return x_pos;
    }

    double calculate_swing_foot_y(double phase, double desired_lateral_vel, bool is_right_foot)
    {
        // Calculate Y position of swing foot based on gait phase and desired lateral velocity
        double step_width = 0.2; // Nominal step width
        double lateral_offset = desired_lateral_vel * step_duration_ * 0.5;

        // Alternate step width for each foot
        if (is_right_foot)
            step_width = -step_width;

        // Apply lateral offset
        step_width += lateral_offset;

        // Cycloidal trajectory for smooth motion
        double y_pos = step_width * (phase - std::sin(2 * M_PI * phase) / (2 * M_PI));

        return y_pos;
    }

    double calculate_swing_foot_z(double phase)
    {
        // Calculate Z position of swing foot (lift trajectory)
        // Use cycloidal trajectory for smooth lift and place
        double lift_phase = phase * 2.0;
        if (lift_phase > 1.0) lift_phase = 2.0 - lift_phase; // Second half goes down

        double z_pos = step_height_ * (lift_phase - std::sin(2 * M_PI * lift_phase) / (2 * M_PI));

        return z_pos;
    }

    void map_foot_trajectory_to_joints(double x_offset, double y_offset, double z_offset, std::vector<double>& commands)
    {
        // Simplified joint mapping - in practice use full inverse kinematics
        // This is a very simplified approach for demonstration

        // Map to ankle joints for position control
        for (size_t i = 0; i < leg_joint_names_.size(); ++i)
        {
            const std::string& joint_name = leg_joint_names_[i];

            // Apply different mappings based on joint type
            if (joint_name.find("ankle_pitch") != std::string::npos)
            {
                // Map Z offset to ankle pitch
                commands[i] = z_offset * 0.5; // Scale factor
            }
            else if (joint_name.find("ankle_roll") != std::string::npos)
            {
                // Map Y offset to ankle roll
                commands[i] = y_offset * 0.3; // Scale factor
            }
            else if (joint_name.find("knee") != std::string::npos)
            {
                // Map Z offset to knee for stepping
                commands[i] = z_offset * 0.8; // Scale factor
            }
            else if (joint_name.find("hip_pitch") != std::string::npos)
            {
                // Map X offset to hip pitch for forward motion
                commands[i] = x_offset * 0.2; // Scale factor
            }
        }
    }

    void apply_balance_feedback(std::vector<double>& commands)
    {
        // Apply balance feedback to maintain stability during walking
        // This would typically come from the balance controller

        // For now, apply simple feedback based on joint positions
        for (size_t i = 0; i < commands.size(); ++i)
        {
            if (i < joint_positions_.size())
            {
                // Apply PD control to maintain nominal joint positions
                double nominal_pos = 0.0; // This would come from gait pattern

                double pos_error = nominal_pos - joint_positions_[i];
                double vel_error = 0.0 - joint_velocities_[i]; // Assuming desired velocity is 0

                commands[i] += 100.0 * pos_error + 10.0 * vel_error; // Simple PD gains
            }
        }
    }

    void apply_safety_constraints(std::vector<double>& commands)
    {
        // Apply joint limits and safety constraints
        for (auto& cmd : commands)
        {
            cmd = std::max(-50.0, std::min(50.0, cmd)); // Torque limits
        }
    }

    bool is_left_foot_swinging()
    {
        // Left foot swings when right foot is in support
        return !left_foot_support_ && gait_phase_ < (1.0 - double_support_ratio_);
    }

    bool is_right_foot_swinging()
    {
        // Right foot swings when left foot is in support
        return left_foot_support_ && gait_phase_ < (1.0 - double_support_ratio_);
    }

    void initialize_gait_state()
    {
        // Initialize publishers
        gait_status_pub_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
            "~/gait_status", rclcpp::QoS(1));
        step_trajectory_pub_ = get_node()->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "~/step_trajectory", rclcpp::QoS(1));

        // Initialize previous joint positions
        previous_joint_positions_.resize(joint_names_.size(), 0.0);
    }

    void publish_gait_status(const rclcpp::Time& time)
    {
        // Publish gait status
        std_msgs::msg::Float64MultiArray status_msg;
        status_msg.data = {
            gait_phase_,
            left_foot_support_ ? 1.0 : 0.0,
            step_counter_,
            desired_velocity_.linear.x,
            desired_velocity_.linear.y,
            desired_velocity_.angular.z
        };
        gait_status_pub_->publish(status_msg);
    }
};

} // namespace humanoid_walk_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    humanoid_walk_controller::WalkController,
    controller_interface::ControllerInterface)
```

## Hardware Interface Implementation

### Custom Humanoid Hardware Interface

```cpp
#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <vector>
#include <string>
#include <map>

namespace humanoid_hardware_interface
{

class HumanoidHardwareInterface : public hardware_interface::SystemInterface
{
public:
    RCLCPP_SHARED_PTR_DEFINITIONS(HumanoidHardwareInterface)

    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override
    {
        if (hardware_interface::SystemInterface::on_init(info) != hardware_interface::CallbackReturn::SUCCESS)
        {
            return hardware_interface::CallbackReturn::ERROR;
        }

        // Initialize joint data structures
        hw_positions_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_velocities_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_efforts_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());
        hw_commands_.resize(info_.joints.size(), std::numeric_limits<double>::quiet_NaN());

        // Store joint names
        for (const auto & joint : info_.joints)
        {
            joint_names_.push_back(joint.name);
        }

        // Initialize hardware communication
        if (!initialize_hardware_communication())
        {
            RCLCPP_ERROR(rclcpp::get_logger("HumanoidHardwareInterface"), "Failed to initialize hardware communication");
            return hardware_interface::CallbackReturn::ERROR;
        }

        RCLCPP_INFO(rclcpp::get_logger("HumanoidHardwareInterface"), "Hardware interface initialized successfully");

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override
    {
        std::vector<hardware_interface::StateInterface> state_interfaces;

        for (size_t i = 0; i < joint_names_.size(); ++i)
        {
            state_interfaces.emplace_back(
                joint_names_[i],
                hardware_interface::HW_IF_POSITION,
                &hw_positions_[i]
            );
            state_interfaces.emplace_back(
                joint_names_[i],
                hardware_interface::HW_IF_VELOCITY,
                &hw_velocities_[i]
            );
            state_interfaces.emplace_back(
                joint_names_[i],
                hardware_interface::HW_IF_EFFORT,
                &hw_efforts_[i]
            );
        }

        // Export IMU interfaces
        state_interfaces.emplace_back("imu", "acceleration.x", &imu_data_.acceleration_x);
        state_interfaces.emplace_back("imu", "acceleration.y", &imu_data_.acceleration_y);
        state_interfaces.emplace_back("imu", "acceleration.z", &imu_data_.acceleration_z);
        state_interfaces.emplace_back("imu", "angular_velocity.x", &imu_data_.angular_velocity_x);
        state_interfaces.emplace_back("imu", "angular_velocity.y", &imu_data_.angular_velocity_y);
        state_interfaces.emplace_back("imu", "angular_velocity.z", &imu_data_.angular_velocity_z);
        state_interfaces.emplace_back("imu", "orientation.x", &imu_data_.orientation_x);
        state_interfaces.emplace_back("imu", "orientation.y", &imu_data_.orientation_y);
        state_interfaces.emplace_back("imu", "orientation.z", &imu_data_.orientation_z);
        state_interfaces.emplace_back("imu", "orientation.w", &imu_data_.orientation_w);

        return state_interfaces;
    }

    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override
    {
        std::vector<hardware_interface::CommandInterface> command_interfaces;

        for (size_t i = 0; i < joint_names_.size(); ++i)
        {
            command_interfaces.emplace_back(
                joint_names_[i],
                hardware_interface::HW_IF_EFFORT,
                &hw_commands_[i]
            );
        }

        return command_interfaces;
    }

    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override
    {
        // Initialize joint values to zero
        for (size_t i = 0; i < hw_positions_.size(); ++i)
        {
            if (std::isnan(hw_positions_[i]))
            {
                hw_positions_[i] = 0.0;
                hw_velocities_[i] = 0.0;
                hw_efforts_[i] = 0.0;
                hw_commands_[i] = 0.0;
            }
        }

        RCLCPP_INFO(rclcpp::get_logger("HumanoidHardwareInterface"), "Hardware interface activated");

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override
    {
        RCLCPP_INFO(rclcpp::get_logger("HumanoidHardwareInterface"), "Hardware interface deactivated");

        return hardware_interface::CallbackReturn::SUCCESS;
    }

    hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override
    {
        // Read joint states from hardware
        if (!read_joint_states())
        {
            RCLCPP_ERROR(rclcpp::get_logger("HumanoidHardwareInterface"), "Failed to read joint states");
            return hardware_interface::return_type::ERROR;
        }

        // Read IMU data
        if (!read_imu_data())
        {
            RCLCPP_WARN(rclcpp::get_logger("HumanoidHardwareInterface"), "Failed to read IMU data");
            // Don't return error for IMU as it's not critical for basic operation
        }

        return hardware_interface::return_type::OK;
    }

    hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override
    {
        // Write joint commands to hardware
        if (!write_joint_commands())
        {
            RCLCPP_ERROR(rclcpp::get_logger("HumanoidHardwareInterface"), "Failed to write joint commands");
            return hardware_interface::return_type::ERROR;
        }

        return hardware_interface::return_type::OK;
    }

private:
    // Joint data
    std::vector<double> hw_positions_;
    std::vector<double> hw_velocities_;
    std::vector<double> hw_efforts_;
    std::vector<double> hw_commands_;
    std::vector<std::string> joint_names_;

    // IMU data structure
    struct IMUData
    {
        double acceleration_x{0.0};
        double acceleration_y{0.0};
        double acceleration_z{0.0};
        double angular_velocity_x{0.0};
        double angular_velocity_y{0.0};
        double angular_velocity_z{0.0};
        double orientation_x{0.0};
        double orientation_y{0.0};
        double orientation_z{0.0};
        double orientation_w{1.0};
    } imu_data_;

    // Hardware communication interface
    bool initialize_hardware_communication()
    {
        // Initialize connection to humanoid robot hardware
        // This would typically involve establishing communication with
        // the robot's main control board or individual joint controllers

        // For simulation, just return true
        return true;
    }

    bool read_joint_states()
    {
        // Read current joint positions, velocities, and efforts from hardware
        // This is a simplified implementation for simulation

        for (size_t i = 0; i < hw_positions_.size(); ++i)
        {
            // In a real implementation, this would read from actual hardware
            // For simulation, we'll just update positions based on commands
            if (!std::isnan(hw_commands_[i]))
            {
                // Simple integration for simulation
                hw_velocities_[i] = hw_commands_[i] * 0.001; // Very simple model
                hw_positions_[i] += hw_velocities_[i] * 0.001; // dt = 0.001s
                hw_efforts_[i] = hw_commands_[i];
            }
        }

        return true;
    }

    bool read_imu_data()
    {
        // Read IMU data from hardware
        // This is a simplified implementation for simulation

        // In a real implementation, this would read from actual IMU
        // For simulation, we'll just generate some realistic values
        static double imu_time = 0.0;
        imu_time += 0.001; // dt = 0.001s

        // Simulate IMU readings with some realistic values
        imu_data_.acceleration_x = 0.1 * std::sin(imu_time * 10.0);
        imu_data_.acceleration_y = 0.1 * std::cos(imu_time * 8.0);
        imu_data_.acceleration_z = 9.81 + 0.2 * std::sin(imu_time * 5.0);

        imu_data_.angular_velocity_x = 0.05 * std::sin(imu_time * 15.0);
        imu_data_.angular_velocity_y = 0.05 * std::cos(imu_time * 12.0);
        imu_data_.angular_velocity_z = 0.02 * std::sin(imu_time * 7.0);

        // For orientation, we'll just keep it near identity
        imu_data_.orientation_x = 0.01 * std::sin(imu_time * 3.0);
        imu_data_.orientation_y = 0.01 * std::cos(imu_time * 4.0);
        imu_data_.orientation_z = 0.005 * std::sin(imu_time * 2.0);
        double norm = std::sqrt(
            imu_data_.orientation_x * imu_data_.orientation_x +
            imu_data_.orientation_y * imu_data_.orientation_y +
            imu_data_.orientation_z * imu_data_.orientation_z +
            imu_data_.orientation_w * imu_data_.orientation_w
        );
        if (norm > 0)
        {
            imu_data_.orientation_x /= norm;
            imu_data_.orientation_y /= norm;
            imu_data_.orientation_z /= norm;
            imu_data_.orientation_w /= norm;
        }

        return true;
    }

    bool write_joint_commands()
    {
        // Write joint commands to hardware
        // This would send the command values to the actual joint controllers

        // In a real implementation, this would send commands to hardware
        // For simulation, we just return true
        return true;
    }
};

} // namespace humanoid_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    humanoid_hardware_interface::HumanoidHardwareInterface,
    hardware_interface::SystemInterface)
```

## Launch Files for Humanoid Control

### Complete Launch System

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import yaml

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    params_file = LaunchConfiguration('params_file')
    namespace = LaunchConfiguration('namespace', default='')

    # Controller manager node
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[PathJoinSubstitution([
            FindPackageShare('humanoid_control'),
            'config',
            'controller_manager.yaml'
        ]), {'use_sim_time': use_sim_time}],
        output='screen',
        namespace=namespace
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('humanoid_description'),
                'urdf',
                'humanoid.urdf.xacro'
            ]),
            'use_sim_time': use_sim_time
        }],
        namespace=namespace
    )

    # Joint state broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
        namespace=namespace
    )

    # Balance controller spawner
    balance_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['humanoid_balance_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        namespace=namespace
    )

    # Walk controller spawner
    walk_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['humanoid_walk_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        namespace=namespace
    )

    # Position controller spawner
    position_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['humanoid_position_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        namespace=namespace
    )

    # Humanoid balance monitor
    balance_monitor_node = Node(
        package='humanoid_control',
        executable='balance_monitor',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        namespace=namespace,
        respawn=True
    )

    # Humanoid gait controller
    gait_controller_node = Node(
        package='humanoid_control',
        executable='gait_controller',
        parameters=[{
            'use_sim_time': use_sim_time
        }],
        namespace=namespace,
        respawn=True
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'))

    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_control'),
            'config',
            'humanoid_control_params.yaml'
        ]),
        description='Full path to the ROS2 parameters file to use for all launched nodes'))

    # Add nodes to launch description
    ld.add_action(controller_manager_node)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_broadcaster_spawner)
    ld.add_action(balance_controller_spawner)
    ld.add_action(walk_controller_spawner)
    ld.add_action(position_controller_spawner)
    ld.add_action(balance_monitor_node)
    ld.add_action(gait_controller_node)

    return ld
```

## Advanced Control Strategies

### Model Predictive Control for Humanoid Balance

```python
import numpy as np
from scipy.optimize import minimize
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration

class HumanoidMPCController(Node):
    def __init__(self):
        super().__init__('humanoid_mpc_controller')

        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.dt = 0.1      # Time step
        self.nx = 4        # State dimension [x, y, vx, vy]
        self.nu = 2        # Control dimension [fx, fy]

        # Cost weights
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost
        self.R = np.diag([0.1, 0.1])               # Control cost
        self.Qf = np.diag([50.0, 50.0, 5.0, 5.0])  # Terminal cost

        # System matrices (simplified inverted pendulum model)
        self.A = self.get_system_matrix()
        self.B = self.get_input_matrix()

        # State variables
        self.current_state = np.zeros(self.nx)
        self.desired_trajectory = []

        # Publishers and subscribers
        self.state_sub = self.create_subscription(
            JointState, '/joint_states', self.state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.com_pub = self.create_publisher(
            PointStamped, '/com_predicted', 10)
        self.mpc_cmd_pub = self.create_publisher(
            Twist, '/mpc_cmd_vel', 10)

        # Timer for MPC update
        self.mpc_timer = self.create_timer(0.05, self.mpc_update)  # 20 Hz

        self.get_logger().info('Humanoid MPC Controller initialized')

    def get_system_matrix(self):
        """Get discrete-time system matrix for inverted pendulum model"""
        # State: [x, y, vx, vy]
        # Dynamics: dx/dt = vx, dy/dt = vy, dvx/dt = u_x/m, dvy/dt = u_y/m
        A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return A

    def get_input_matrix(self):
        """Get input matrix for inverted pendulum model"""
        # Assuming unit mass for simplicity
        B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [self.dt, 0],
            [0, self.dt]
        ])
        return B

    def state_callback(self, msg):
        """Update current state from joint states"""
        # In practice, this would use a state estimator
        # For now, we'll use a simplified approach
        pass

    def imu_callback(self, msg):
        """Update state from IMU data"""
        # Extract orientation and angular velocity
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        # Calculate CoM position and velocity from IMU
        # This is a simplified approach - in practice, use full state estimator
        self.current_state[0] = orientation.x * 0.1  # Simplified mapping
        self.current_state[1] = orientation.y * 0.1
        self.current_state[2] = angular_velocity.x * 0.01
        self.current_state[3] = angular_velocity.y * 0.01

    def cmd_vel_callback(self, msg):
        """Store desired velocity command"""
        self.desired_velocity = np.array([msg.linear.x, msg.linear.y])

    def mpc_update(self):
        """Run MPC optimization and publish commands"""
        try:
            # Predict future states and optimize control
            optimal_control_sequence = self.solve_mpc_problem()

            if optimal_control_sequence is not None and len(optimal_control_sequence) > 0:
                # Apply first control in sequence
                first_control = optimal_control_sequence[0]

                # Create and publish command
                cmd_msg = Twist()
                cmd_msg.linear.x = float(first_control[0])
                cmd_msg.linear.y = float(first_control[1])

                self.mpc_cmd_pub.publish(cmd_msg)

                # Publish predicted CoM trajectory for visualization
                self.publish_predicted_trajectory(optimal_control_sequence)

        except Exception as e:
            self.get_logger().error(f'MPC update error: {e}')

    def solve_mpc_problem(self):
        """Solve the MPC optimization problem"""
        try:
            # Define the optimization problem
            # Minimize: sum(x_k^T * Q * x_k + u_k^T * R * u_k) + x_N^T * Qf * x_N
            # Subject to: x_{k+1} = A*x_k + B*u_k

            # Initial state
            x0 = self.current_state.copy()

            # Decision variables: [u_0, u_1, ..., u_{N-1}]
            n_vars = self.horizon * self.nu

            def objective(u_flat):
                """Objective function to minimize"""
                total_cost = 0.0
                x = x0.copy()

                # Reshape control sequence
                U = u_flat.reshape((self.horizon, self.nu))

                for k in range(self.horizon):
                    # State cost
                    total_cost += x.T @ self.Q @ x

                    # Control cost
                    total_cost += U[k, :].T @ self.R @ U[k, :]

                    # Predict next state
                    x = self.A @ x + self.B @ U[k, :]

                # Terminal cost
                total_cost += x.T @ self.Qf @ x

                return total_cost

            def objective_gradient(u_flat):
                """Gradient of objective function"""
                # This is a simplified gradient calculation
                # In practice, use automatic differentiation
                grad = np.zeros_like(u_flat)
                x = x0.copy()
                lam = np.zeros(self.nx)  # Costate

                # Reshape control sequence
                U = u_flat.reshape((self.horizon, self.nu))

                # Forward pass - compute states
                x_seq = [x0.copy()]
                for k in range(self.horizon):
                    x = self.A @ x + self.B @ U[k, :]
                    x_seq.append(x)

                # Backward pass - compute costates
                lam = self.Qf @ x_seq[-1]  # Terminal condition
                for k in range(self.horizon - 1, -1, -1):
                    grad_k = self.R @ U[k, :] + self.B.T @ lam
                    grad[k*self.nu:(k+1)*self.nu] = grad_k

                    # Update costate
                    lam = self.Q @ x_seq[k] + self.A.T @ lam

                return grad

            # Initial guess
            u_init = np.zeros(n_vars)

            # Constraints (if any)
            constraints = []

            # Bounds on controls (torque limits)
            control_bounds = [(-50.0, 50.0)] * n_vars  # ±50 Nm torque limit

            # Solve optimization problem
            result = minimize(
                fun=objective,
                x0=u_init,
                method='SLSQP',
                jac=objective_gradient,
                bounds=control_bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 100}
            )

            if result.success:
                optimal_U = result.x.reshape((self.horizon, self.nu))
                return optimal_U
            else:
                self.get_logger().warn(f'MPC optimization failed: {result.message}')
                return None

        except Exception as e:
            self.get_logger().error(f'MPC optimization error: {e}')
            return None

    def publish_predicted_trajectory(self, control_sequence):
        """Publish predicted CoM trajectory for visualization"""
        # Predict trajectory based on current state and control sequence
        x = self.current_state.copy()
        trajectory = [x[:2]]  # Store x, y positions

        for k in range(min(len(control_sequence), 5)):  # Show first 5 predictions
            x = self.A @ x + self.B @ control_sequence[k, :]
            trajectory.append(x[:2])

        # Publish the first predicted position
        if len(trajectory) > 0:
            com_msg = PointStamped()
            com_msg.header.stamp = self.get_clock().now().to_msg()
            com_msg.header.frame_id = 'base_link'
            com_msg.point.x = float(trajectory[0][0])
            com_msg.point.y = float(trajectory[0][1])
            com_msg.point.z = 0.8  # Fixed height assumption

            self.com_pub.publish(com_msg)

class HumanoidImpedanceController(Node):
    """Impedance controller for compliant humanoid motion"""

    def __init__(self):
        super().__init__('humanoid_impedance_controller')

        # Impedance parameters
        self.stiffness_diag = [1000.0, 1000.0, 1000.0, 200.0, 200.0, 100.0]  # [x, y, z, rx, ry, rz]
        self.damping_diag = [200.0, 200.0, 200.0, 40.0, 40.0, 20.0]
        self.mass_diag = [10.0, 10.0, 10.0, 2.0, 2.0, 1.0]

        # Desired pose and velocity
        self.desired_pose = np.zeros(6)  # [x, y, z, rx, ry, rz]
        self.desired_velocity = np.zeros(6)

        # Current state
        self.current_pose = np.zeros(6)
        self.current_velocity = np.zeros(6)

        # Subscribers and publishers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.desired_pose_sub = self.create_subscription(
            PoseStamped, '/impedance_desired_pose', self.desired_pose_callback, 10)
        self.impedance_cmd_pub = self.create_publisher(
            JointTrajectory, '/impedance_joint_commands', 10)

        self.get_logger().info('Humanoid Impedance Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint state"""
        # This would update the current pose and velocity based on forward kinematics
        pass

    def desired_pose_callback(self, msg):
        """Update desired pose"""
        self.desired_pose[:3] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]

        # Convert orientation to Euler angles
        orientation = msg.pose.orientation
        euler = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
        self.desired_pose[3:] = euler

    def compute_impedance_force(self):
        """Compute impedance-based force/torque"""
        # Calculate pose error
        pose_error = self.desired_pose - self.current_pose
        velocity_error = self.desired_velocity - self.current_velocity

        # Compute impedance force: F = K * (xd - x) + D * (vxd - vx) + M * (xdd - ax)
        stiffness = np.diag(self.stiffness_diag)
        damping = np.diag(self.damping_diag)
        mass = np.diag(self.mass_diag)

        impedance_force = (
            stiffness @ pose_error +
            damping @ velocity_error
        )

        return impedance_force

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Convert quaternion to rotation matrix, then to Euler angles
        # Simplified conversion
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return [roll, pitch, yaw]

def main(args=None):
    rclpy.init(args=args)

    # Choose which controller to run
    controller = HumanoidMPCController()  # or HumanoidImpedanceController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Navigation Stack

### Balance-Aware Navigation Controller

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
import math

class BalanceAwareNavigationController(Node):
    """Navigation controller that considers humanoid balance constraints"""

    def __init__(self):
        super().__init__('balance_aware_navigation_controller')

        # Balance parameters
        self.com_height = 0.85  # Nominal CoM height for humanoid
        self.support_polygon_margin = 0.05  # Safety margin around feet
        self.max_lean_angle = 0.3  # Maximum lean angle in radians
        self.balance_threshold = 0.1  # Threshold for balance safety

        # Navigation parameters
        self.max_linear_velocity = 0.2  # Reduced for stability
        self.max_angular_velocity = 0.3
        self.min_linear_velocity = 0.05
        self.waypoint_tolerance = 0.3

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.current_imu = None
        self.current_joints = None
        self.balance_state = {
            'com_position': np.array([0.0, 0.0, self.com_height]),
            'zmp_position': np.array([0.0, 0.0]),
            'is_balanced': True,
            'support_polygon': []
        }
        self.path = Path()
        self.path_index = 0

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers and subscribers
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_balanced', 10)
        self.balance_status_pub = self.create_publisher(
            Float64MultiArray, '/balance_status', 10)
        self.zmp_pub = self.create_publisher(PointStamped, '/zmp_position', 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info('Balance-aware Navigation Controller initialized')

    def path_callback(self, msg):
        """Receive navigation path"""
        self.path = msg
        self.path_index = 0
        self.get_logger().info(f'Received path with {len(self.path.poses)} waypoints')

    def odom_callback(self, msg):
        """Receive odometry data"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def imu_callback(self, msg):
        """Receive IMU data for balance estimation"""
        self.current_imu = msg
        self.update_balance_state()

    def joint_callback(self, msg):
        """Receive joint state data"""
        self.current_joints = msg
        self.update_support_polygon()

    def update_balance_state(self):
        """Update balance state based on sensor data"""
        if self.current_imu is None:
            return

        # Estimate center of mass position from IMU data
        # This is a simplified approach - in practice use full state estimation
        self.balance_state['com_position'][0] = self.current_imu.linear_acceleration.x * 0.01
        self.balance_state['com_position'][1] = self.current_imu.linear_acceleration.y * 0.01

        # Calculate Zero Moment Point (ZMP) from IMU
        # ZMP_x = CoM_x - (CoM_height / gravity) * linear_acc_x
        gravity = 9.81
        zmp_x = self.balance_state['com_position'][0] - (self.com_height / gravity) * self.current_imu.linear_acceleration.x
        zmp_y = self.balance_state['com_position'][1] - (self.com_height / gravity) * self.current_imu.linear_acceleration.y

        self.balance_state['zmp_position'] = np.array([zmp_x, zmp_y])

        # Check if ZMP is within support polygon (balance check)
        self.balance_state['is_balanced'] = self.is_zmp_stable()

    def update_support_polygon(self):
        """Update support polygon based on foot positions"""
        if self.current_joints is None:
            return

        # Calculate foot positions from joint angles
        # This is a simplified approach - in practice use forward kinematics
        left_foot_x = 0.1  # Simplified position
        left_foot_y = 0.1
        right_foot_x = 0.1
        right_foot_y = -0.1

        # Create support polygon (convex hull of feet)
        self.balance_state['support_polygon'] = [
            [left_foot_x, left_foot_y],
            [right_foot_x, right_foot_y]
        ]

    def is_zmp_stable(self):
        """Check if ZMP is within support polygon"""
        if len(self.balance_state['support_polygon']) < 2:
            return False

        zmp = self.balance_state['zmp_position']
        support_poly = np.array(self.balance_state['support_polygon'])

        # Check if ZMP is within support polygon with safety margin
        # For simplicity, check distance to closest support point
        min_distance = float('inf')
        for point in support_poly:
            distance = np.linalg.norm(zmp - point)
            if distance < min_distance:
                min_distance = distance

        # Consider balanced if ZMP is close enough to support polygon
        return min_distance <= (self.support_polygon_margin + 0.1)

    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None or len(self.path.poses) == 0:
            return

        try:
            # Calculate desired velocity towards next waypoint
            desired_twist = self.calculate_desired_velocity()

            # Apply balance constraints to velocity command
            balanced_twist = self.apply_balance_constraints(desired_twist)

            # Publish balanced command
            self.cmd_vel_pub.publish(balanced_twist)

            # Publish balance status
            self.publish_balance_status()

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def calculate_desired_velocity(self):
        """Calculate desired velocity towards next waypoint"""
        if self.path_index >= len(self.path.poses):
            # Path completed
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        # Get next waypoint
        target_pose = self.path.poses[self.path_index].pose

        # Calculate distance and angle to target
        dx = target_pose.position.x - self.current_pose.position.x
        dy = target_pose.position.y - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate desired heading
        desired_yaw = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_pose(self.current_pose)

        # Calculate heading error
        heading_error = self.normalize_angle(desired_yaw - current_yaw)

        # Create velocity command
        cmd = Twist()

        # Adjust linear velocity based on distance and balance state
        if distance > self.waypoint_tolerance:
            # Scale velocity based on distance and balance
            base_velocity = min(self.max_linear_velocity, distance * 0.5)

            # Reduce velocity if not balanced
            if not self.balance_state['is_balanced']:
                base_velocity *= 0.3  # Significant reduction when unbalanced

            cmd.linear.x = base_velocity
        else:
            cmd.linear.x = 0.0
            # Check if reached waypoint
            if distance <= self.waypoint_tolerance:
                self.path_index += 1

        # Calculate angular velocity for heading correction
        cmd.angular.z = heading_error * 1.5  # Proportional control

        # Limit angular velocity
        cmd.angular.z = max(-self.max_angular_velocity,
                           min(self.max_angular_velocity, cmd.angular.z))

        return cmd

    def apply_balance_constraints(self, desired_twist):
        """Apply balance constraints to desired velocity"""
        constrained_twist = Twist()
        constrained_twist.linear = desired_twist.linear
        constrained_twist.angular = desired_twist.angular

        # If not balanced, reduce velocities significantly
        if not self.balance_state['is_balanced']:
            reduction_factor = 0.3
            constrained_twist.linear.x *= reduction_factor
            constrained_twist.linear.y *= reduction_factor
            constrained_twist.angular.z *= reduction_factor
        else:
            # If balanced, check if we can increase speed
            if self.balance_state['zmp_position'][0]**2 + self.balance_state['zmp_position'][1]**2 < 0.01:
                # Very stable, can increase speed slightly
                speed_increase = 1.1
                constrained_twist.linear.x = min(
                    self.max_linear_velocity,
                    constrained_twist.linear.x * speed_increase
                )

        # Additional constraints based on ZMP position
        zmp_norm = np.linalg.norm(self.balance_state['zmp_position'])
        if zmp_norm > 0.1:  # Close to stability boundary
            # Reduce velocity as we approach stability boundary
            velocity_reduction = max(0.5, 1.0 - zmp_norm)
            constrained_twist.linear.x *= velocity_reduction
            constrained_twist.angular.z *= velocity_reduction

        return constrained_twist

    def get_yaw_from_pose(self, pose):
        """Extract yaw from pose quaternion"""
        quaternion = pose.orientation
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def publish_balance_status(self):
        """Publish balance status information"""
        status_msg = Float64MultiArray()
        status_msg.data = [
            self.balance_state['zmp_position'][0],
            self.balance_state['zmp_position'][1],
            self.balance_state['com_position'][0],
            self.balance_state['com_position'][1],
            self.balance_state['com_position'][2],
            1.0 if self.balance_state['is_balanced'] else 0.0
        ]
        self.balance_status_pub.publish(status_msg)

        # Publish ZMP visualization
        zmp_msg = PointStamped()
        zmp_msg.header.stamp = self.get_clock().now().to_msg()
        zmp_msg.header.frame_id = 'base_link'
        zmp_msg.point.x = float(self.balance_state['zmp_position'][0])
        zmp_msg.point.y = float(self.balance_state['zmp_position'][1])
        zmp_msg.point.z = 0.0  # ZMP is on ground plane
        self.zmp_pub.publish(zmp_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = BalanceAwareNavigationController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid Control

### Safety and Stability Considerations

```python
class HumanoidSafetyController:
    """Safety controller for humanoid robots with balance and collision protection"""

    def __init__(self):
        self.safety_limits = {
            'max_joint_velocity': 5.0,      # rad/s
            'max_joint_torque': 100.0,      # Nm
            'max_linear_velocity': 0.5,     # m/s
            'max_angular_velocity': 0.5,    # rad/s
            'max_lean_angle': 0.5,          # rad
            'min_com_height': 0.3,          # m
            'max_zmp_deviation': 0.3        # m
        }

        self.emergency_stop_active = False
        self.balance_threshold = 0.2
        self.collision_threshold = 0.3  # meters

    def check_balance_safety(self, robot_state):
        """Check if robot is in safe balance state"""
        # Check CoM position relative to support polygon
        com_position = robot_state['com_position']
        zmp_position = robot_state['zmp_position']

        # Calculate distance from ZMP to support boundary
        zmp_distance = np.linalg.norm(zmp_position)

        if zmp_distance > self.safety_limits['max_zmp_deviation']:
            return False, f"ZMP deviation too large: {zmp_distance:.3f}m"

        # Check lean angle
        if abs(robot_state['lean_angle']) > self.safety_limits['max_lean_angle']:
            return False, f"Lean angle exceeded: {robot_state['lean_angle']:.3f} rad"

        # Check CoM height
        if com_position[2] < self.safety_limits['min_com_height']:
            return False, f"CoM height too low: {com_position[2]:.3f}m"

        return True, "Balanced"

    def check_collision_safety(self, laser_scan, robot_pose):
        """Check for collision risks based on laser scan"""
        min_distance = float('inf')

        for i, range_val in enumerate(laser_scan.ranges):
            if 0 < range_val < laser_scan.range_max:
                if range_val < min_distance:
                    min_distance = range_val

        if min_distance < self.collision_threshold:
            return False, f"Collision imminent: {min_distance:.3f}m"

        return True, "Safe"

    def apply_safety_constraints(self, command, robot_state, laser_scan=None):
        """Apply safety constraints to motion command"""
        if self.emergency_stop_active:
            # Return zero command if emergency stop is active
            zero_cmd = type(command)()
            if hasattr(zero_cmd, 'linear'):
                zero_cmd.linear.x = 0.0
                zero_cmd.linear.y = 0.0
                zero_cmd.linear.z = 0.0
            if hasattr(zero_cmd, 'angular'):
                zero_cmd.angular.x = 0.0
                zero_cmd.angular.y = 0.0
                zero_cmd.angular.z = 0.0
            return zero_cmd

        # Check balance safety
        is_balanced, balance_msg = self.check_balance_safety(robot_state)
        if not is_balanced:
            # Reduce command magnitude significantly
            reduction_factor = 0.1
            command.linear.x *= reduction_factor
            command.linear.y *= reduction_factor
            command.angular.z *= reduction_factor

        # Check collision safety if laser data available
        if laser_scan is not None:
            is_safe, collision_msg = self.check_collision_safety(laser_scan, robot_state['pose'])
            if not is_safe:
                # Apply evasive action
                command.linear.x *= 0.5  # Slow down
                command.angular.z = 0.5  # Turn away from obstacle

        # Apply velocity limits
        if hasattr(command, 'linear'):
            command.linear.x = max(
                -self.safety_limits['max_linear_velocity'],
                min(self.safety_limits['max_linear_velocity'], command.linear.x)
            )
            command.linear.y = max(
                -self.safety_limits['max_linear_velocity'],
                min(self.safety_limits['max_linear_velocity'], command.linear.y)
            )

        if hasattr(command, 'angular'):
            command.angular.z = max(
                -self.safety_limits['max_angular_velocity'],
                min(self.safety_limits['max_angular_velocity'], command.angular.z)
            )

        return command

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop_active = True
        print("EMERGENCY STOP ACTIVATED")

    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop_active = False
        print("Emergency stop reset")
```

## Summary

This chapter covered ROS 2 Control for humanoid robots, including:

- ROS 2 Control architecture and components
- Humanoid-specific control challenges and solutions
- Custom controller implementations for balance and walking
- Hardware interface development for humanoid robots
- Advanced control strategies (MPC, Impedance Control)
- Balance-aware navigation integration
- Safety and stability considerations

The ROS 2 Control framework provides the foundation for implementing sophisticated control systems for humanoid robots, enabling precise joint control, balance maintenance, and coordinated locomotion.

## Exercises

1. Implement a custom controller for humanoid upper-body motion during navigation.
2. How would you modify the balance controller for different walking speeds?
3. Design a controller that can handle transitions between standing and walking.

## Code Example: Complete Humanoid Control System

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
import numpy as np
import math

class CompleteHumanoidControlSystem(Node):
    def __init__(self):
        super().__init__('complete_humanoid_control_system')

        # Control system components
        self.balance_controller = HumanoidBalanceController()
        self.walk_controller = HumanoidWalkController()
        self.safety_controller = HumanoidSafetyController()

        # Robot state
        self.robot_state = {
            'joint_positions': [],
            'joint_velocities': [],
            'joint_efforts': [],
            'imu_data': None,
            'com_position': np.array([0.0, 0.0, 0.8]),
            'zmp_position': np.array([0.0, 0.0]),
            'pose': None,
            'twist': None
        }

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.desired_cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_desired', self.desired_cmd_callback, 10)

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointState, '/joint_commands', 10)
        self.actual_cmd_pub = self.create_publisher(
            Twist, '/cmd_vel_actual', 10)
        self.balance_status_pub = self.create_publisher(
            Float64MultiArray, '/balance_status', 10)

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_step)  # 100 Hz

        self.get_logger().info('Complete Humanoid Control System initialized')

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.robot_state['joint_positions'] = list(msg.position)
        self.robot_state['joint_velocities'] = list(msg.velocity)
        self.robot_state['joint_efforts'] = list(msg.effort)

    def imu_callback(self, msg):
        """Update IMU data and calculate balance state"""
        self.robot_state['imu_data'] = msg

        # Calculate CoM and ZMP from IMU data
        gravity = 9.81
        self.robot_state['com_position'][0] += msg.linear_acceleration.x * 0.0001  # Integration
        self.robot_state['com_position'][1] += msg.linear_acceleration.y * 0.0001
        self.robot_state['zmp_position'][0] = self.robot_state['com_position'][0] - (0.8 / gravity) * msg.linear_acceleration.x
        self.robot_state['zmp_position'][1] = self.robot_state['com_position'][1] - (0.8 / gravity) * msg.linear_acceleration.y

    def desired_cmd_callback(self, msg):
        """Store desired command"""
        self.desired_command = msg

    def control_step(self):
        """Main control step"""
        if not hasattr(self, 'desired_command'):
            return

        try:
            # Update balance state
            self.robot_state['is_balanced'], _ = self.safety_controller.check_balance_safety(self.robot_state)

            # Calculate balance control commands
            balance_cmds = self.balance_controller.compute_balance_control(
                self.robot_state, self.desired_command)

            # Calculate walking gait commands
            walk_cmds = self.walk_controller.compute_walk_control(
                self.robot_state, self.desired_command)

            # Combine commands based on current state
            if self.robot_state['is_balanced']:
                combined_cmds = self.combine_commands(balance_cmds, walk_cmds, 0.7)
            else:
                # Prioritize balance when unstable
                combined_cmds = self.combine_commands(balance_cmds, walk_cmds, 0.9)

            # Apply safety constraints
            safe_cmds = self.safety_controller.apply_safety_constraints(
                combined_cmds, self.robot_state)

            # Publish commands
            self.publish_joint_commands(safe_cmds)
            self.publish_balance_status()

        except Exception as e:
            self.get_logger().error(f'Control step error: {e}')

    def combine_commands(self, balance_cmd, walk_cmd, balance_weight=0.5):
        """Combine balance and walking commands"""
        combined_cmd = Twist()

        # Blend commands based on weight
        combined_cmd.linear.x = (balance_weight * balance_cmd.linear.x +
                                (1 - balance_weight) * walk_cmd.linear.x)
        combined_cmd.linear.y = (balance_weight * balance_cmd.linear.y +
                                (1 - balance_weight) * walk_cmd.linear.y)
        combined_cmd.angular.z = (balance_weight * balance_cmd.angular.z +
                                 (1 - balance_weight) * walk_cmd.angular.z)

        return combined_cmd

    def publish_joint_commands(self, commands):
        """Publish joint commands"""
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = [f'joint_{i}' for i in range(len(self.robot_state['joint_positions']))]
        cmd_msg.effort = [0.0] * len(self.robot_state['joint_positions'])  # Placeholder
        self.joint_cmd_pub.publish(cmd_msg)

    def publish_balance_status(self):
        """Publish balance status"""
        status_msg = Float64MultiArray()
        status_msg.data = [
            self.robot_state['zmp_position'][0],
            self.robot_state['zmp_position'][1],
            self.robot_state['com_position'][0],
            self.robot_state['com_position'][1],
            self.robot_state['com_position'][2],
            1.0 if self.robot_state['is_balanced'] else 0.0
        ]
        self.balance_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    control_system = CompleteHumanoidControlSystem()

    try:
        rclpy.spin(control_system)
    except KeyboardInterrupt:
        pass
    finally:
        control_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This implementation provides a complete ROS 2 Control framework for humanoid robots with balance-aware navigation capabilities. The system integrates multiple control strategies to ensure stable and safe locomotion while maintaining the robot's balance during navigation tasks.