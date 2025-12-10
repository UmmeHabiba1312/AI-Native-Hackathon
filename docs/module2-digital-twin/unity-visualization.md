---
title: Unity Visualization & Human-Robot Interaction
description: Understanding Unity for robotics simulation and human-robot interaction
sidebar_position: 3
---

# Unity Visualization & Human-Robot Interaction

## Learning Objectives

By the end of this chapter, you will be able to:
1. Create realistic 3D visualization environments in Unity for robotics
2. Implement human-robot interaction mechanisms in Unity
3. Integrate Unity with ROS 2 for real-time simulation
4. Design intuitive interfaces for robot teleoperation and monitoring
5. Implement multi-user collaboration in shared virtual environments

## Introduction to Unity for Robotics

Unity has emerged as a powerful platform for robotics simulation and visualization due to its advanced graphics capabilities, physics engine, and flexible scripting environment. For robotics applications, Unity offers:

- **High-fidelity graphics**: Realistic rendering for synthetic data generation
- **Physics simulation**: Accurate physics for robot dynamics
- **Flexible interaction**: Intuitive interfaces for human-robot interaction
- **Cross-platform deployment**: Runs on various devices and platforms
- **Asset ecosystem**: Extensive library of models and tools

## Setting Up Unity for Robotics

### Unity Robotics Hub Installation

To set up Unity for robotics applications, install the Unity Robotics Hub which includes:

- **Unity Robotics Package**: ROS-TCP-Connector and tutorials
- **Unity Computer Vision Package**: Tools for synthetic data generation
- **Unity Simulation Package**: Large-scale simulation capabilities

### Basic Robot Setup in Unity

```csharp
using UnityEngine;
using System.Collections;
using RosSharp.RosBridgeClient;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "my_robot";
    public float maxVelocity = 1.0f;
    public float maxAngularVelocity = 1.0f;

    [Header("ROS Connection")]
    public string rosBridgeServerUrl = "ws://192.168.1.100:9090";

    [Header("Joint Configuration")]
    public Transform[] joints;
    public string[] jointNames;

    // ROS Communication
    private RosSocket rosSocket;
    private string rosTopic = "/cmd_vel";
    private string jointStatesTopic = "/joint_states";

    // Robot state
    private float linearVelocity = 0f;
    private float angularVelocity = 0f;
    private float[] jointPositions;

    void Start()
    {
        // Initialize ROS connection
        InitializeRosConnection();

        // Initialize joint positions
        jointPositions = new float[joints.Length];

        // Subscribe to command topics
        SubscribeToTopics();
    }

    void InitializeRosConnection()
    {
        // Create ROS socket
        WebSocketProtocols webSocketProtocol = new WebSocketProtocols();
        rosSocket = new RosSocket(webSocketProtocol, rosBridgeServerUrl);

        Debug.Log($"Connected to ROS Bridge at {rosBridgeServerUrl}");
    }

    void SubscribeToTopics()
    {
        // Subscribe to velocity commands
        rosSocket.Subscribe<Messages.Geometry.Twist>(rosTopic, ProcessVelocityCommand);

        // Subscribe to joint state commands
        rosSocket.Subscribe<Messages.Sensor.JointState>(jointStatesTopic, ProcessJointCommands);
    }

    void ProcessVelocityCommand(Messages.Geometry.Twist message)
    {
        // Update robot velocities from ROS message
        linearVelocity = (float)message.linear.x;
        angularVelocity = (float)message.angular.z;

        // Apply differential drive kinematics
        ApplyDifferentialDrive(linearVelocity, angularVelocity);
    }

    void ProcessJointCommands(Messages.Sensor.JointState message)
    {
        // Process joint state commands
        for (int i = 0; i < message.name.Count; i++)
        {
            string jointName = message.name[i];
            float position = (float)message.position[i];

            // Find corresponding joint in our array
            for (int j = 0; j < jointNames.Length; j++)
            {
                if (jointNames[j] == jointName)
                {
                    jointPositions[j] = position;
                    break;
                }
            }
        }
    }

    void ApplyDifferentialDrive(float linearVel, float angularVel)
    {
        // Simple differential drive model
        float leftWheelVel = linearVel - angularVel * 0.5f; // 0.5m track width
        float rightWheelVel = linearVel + angularVel * 0.5f;

        // Apply velocities to wheel joints
        if (joints.Length >= 2)
        {
            // Update wheel rotations based on velocity
            joints[0].Rotate(Vector3.right, leftWheelVel * Time.deltaTime * Mathf.Rad2Deg);
            joints[1].Rotate(Vector3.right, rightWheelVel * Time.deltaTime * Mathf.Rad2Deg);
        }

        // Move the robot body
        transform.Translate(Vector3.forward * linearVel * Time.deltaTime);
        transform.Rotate(Vector3.up, angularVel * Time.deltaTime * Mathf.Rad2Deg);
    }

    void Update()
    {
        // Apply joint positions to transforms
        for (int i = 0; i < joints.Length && i < jointPositions.Length; i++)
        {
            joints[i].localRotation = Quaternion.Euler(0, 0, jointPositions[i] * Mathf.Rad2Deg);
        }
    }

    void OnDestroy()
    {
        // Clean up ROS connection
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

## Human-Robot Interaction Interfaces

### VR/AR Interaction Systems

```csharp
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit;

public class VRHumanRobotInterface : MonoBehaviour
{
    [Header("VR Interaction")]
    public XRNode controllerNode;
    public InputDeviceCharacteristics controllerCharacteristics;

    [Header("Robot Interaction")]
    public UnityRobotController robotController;
    public Transform interactionPoint;

    [Header("UI Elements")]
    public GameObject commandPanel;
    public GameObject statusDisplay;

    private InputDevice targetDevice;
    private Vector2 primaryAxis;
    private bool triggerPressed = false;
    private bool gripPressed = false;

    void Start()
    {
        // Find controller
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(controllerNode, devices);

        if (devices.Count > 0)
        {
            targetDevice = devices[0];
        }
    }

    void Update()
    {
        // Update controller input
        UpdateControllerInput();

        // Handle robot interaction
        HandleRobotInteraction();

        // Update UI
        UpdateStatusDisplay();
    }

    void UpdateControllerInput()
    {
        // Get controller input
        if (targetDevice.isValid)
        {
            // Get thumbstick input for movement
            if (targetDevice.TryGetFeatureValue(CommonUsages.primary2DAxis, out primaryAxis))
            {
                // Map thumbstick to robot velocity
                float linearVel = primaryAxis.y * robotController.maxVelocity;
                float angularVel = primaryAxis.x * robotController.maxAngularVelocity;

                // Send command to robot
                SendVelocityCommand(linearVel, angularVel);
            }

            // Get trigger press for special actions
            if (targetDevice.TryGetFeatureValue(CommonUsages.triggerButton, out triggerPressed) && triggerPressed)
            {
                // Trigger pressed - perform action
                PerformAction();
            }

            // Get grip press for grasping
            if (targetDevice.TryGetFeatureValue(CommonUsages.gripButton, out gripPressed) && gripPressed)
            {
                // Grip pressed - control gripper
                ControlGripper(true);
            }
            else
            {
                ControlGripper(false);
            }
        }
    }

    void HandleRobotInteraction()
    {
        // Handle direct interaction with robot
        if (Vector3.Distance(interactionPoint.position, robotController.transform.position) < 3.0f)
        {
            commandPanel.SetActive(true);

            // Highlight robot when in range
            Renderer robotRenderer = robotController.GetComponent<Renderer>();
            if (robotRenderer != null)
            {
                robotRenderer.material.color = Color.yellow;
            }
        }
        else
        {
            commandPanel.SetActive(false);

            // Reset robot color
            Renderer robotRenderer = robotController.GetComponent<Renderer>();
            if (robotRenderer != null)
            {
                robotRenderer.material.color = Color.white;
            }
        }
    }

    void SendVelocityCommand(float linearVelocity, float angularVelocity)
    {
        // Send velocity command to robot via ROS
        if (robotController != null)
        {
            // This would typically send a ROS message
            Debug.Log($"Sending velocity command: linear={linearVelocity}, angular={angularVelocity}");
        }
    }

    void PerformAction()
    {
        // Perform special action when trigger is pressed
        Debug.Log("Performing action via VR interface");
    }

    void ControlGripper(bool gripActive)
    {
        // Control robot gripper
        Debug.Log($"Controlling gripper: {gripActive}");
    }

    void UpdateStatusDisplay()
    {
        // Update robot status display
        if (statusDisplay != null)
        {
            // This would update with real robot status
            statusDisplay.GetComponent<UnityEngine.UI.Text>().text =
                $"Robot: {robotController.robotName}\n" +
                $"Position: {robotController.transform.position}";
        }
    }
}
```

## Teleoperation Interface

### Unity-based Teleoperation System

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

public class UnityTeleoperationInterface : MonoBehaviour
{
    [Header("Teleoperation Panels")]
    public GameObject mainControlPanel;
    public GameObject cameraFeedPanel;
    public GameObject joystickPanel;
    public GameObject commandHistoryPanel;

    [Header("Camera Feeds")]
    public RawImage mainCameraFeed;
    public RawImage overheadCameraFeed;
    public RawImage armCameraFeed;

    [Header("Joystick Controls")]
    public RectTransform leftJoystick;
    public RectTransform rightJoystick;
    public float joystickDeadZone = 0.1f;

    [Header("Command Interface")]
    public Button[] commandButtons;
    public InputField commandInputField;
    public Text commandHistoryText;

    [Header("Robot Connection")]
    public UnityRobotController robotController;

    // Joystick state
    private Vector2 leftJoystickInput;
    private Vector2 rightJoystickInput;
    private bool leftJoystickActive;
    private bool rightJoystickActive;

    // Command history
    private List<string> commandHistory = new List<string>();
    private const int maxCommandHistory = 20;

    void Start()
    {
        InitializeTeleoperationInterface();
        SetupEventHandlers();
    }

    void InitializeTeleoperationInterface()
    {
        // Initialize camera feeds
        SetupCameraFeeds();

        // Initialize joystick controls
        SetupJoysticks();

        // Initialize command buttons
        SetupCommandButtons();

        // Load saved preferences
        LoadPreferences();
    }

    void SetupCameraFeeds()
    {
        // In a real implementation, these would connect to robot cameras
        // For simulation, we might show pre-recorded footage or render textures
        Debug.Log("Setting up camera feeds...");
    }

    void SetupJoysticks()
    {
        // Set up joystick drag handlers
        leftJoystick.GetComponent<EventTrigger>().triggers.Add(CreateDragHandler(true));
        rightJoystick.GetComponent<EventTrigger>().triggers.Add(CreateDragHandler(false));
    }

    EventTrigger.Entry CreateDragHandler(bool isLeftJoystick)
    {
        EventTrigger.Entry entry = new EventTrigger.Entry();
        entry.eventID = EventTriggerType.Drag;

        if (isLeftJoystick)
        {
            entry.callback.AddListener((data) => {
                PointerEventData pointerData = (PointerEventData)data;
                leftJoystickInput = CalculateJoystickInput(pointerData, leftJoystick);
                leftJoystickActive = true;
            });
        }
        else
        {
            entry.callback.AddListener((data) => {
                PointerEventData pointerData = (PointerEventData)data;
                rightJoystickInput = CalculateJoystickInput(pointerData, rightJoystick);
                rightJoystickActive = true;
            });
        }

        return entry;
    }

    Vector2 CalculateJoystickInput(PointerEventData data, RectTransform joystick)
    {
        // Calculate normalized joystick input
        Vector2 localPoint;
        RectTransformUtility.ScreenPointToLocalPointInRectangle(joystick, data.position, data.pressEventCamera, out localPoint);

        // Normalize to -1 to 1 range
        Vector2 normalized = new Vector2(
            Mathf.Clamp(localPoint.x / (joystick.rect.width / 2), -1, 1),
            Mathf.Clamp(localPoint.y / (joystick.rect.height / 2), -1, 1)
        );

        // Apply dead zone
        if (normalized.magnitude < joystickDeadZone)
        {
            return Vector2.zero;
        }

        return normalized;
    }

    void SetupCommandButtons()
    {
        foreach (Button button in commandButtons)
        {
            button.onClick.AddListener(() => {
                string command = button.GetComponentInChildren<Text>().text;
                ExecuteCommand(command);
            });
        }

        // Setup command input field
        commandInputField.onEndEdit.AddListener((command) => {
            ExecuteCommand(command);
        });
    }

    void Update()
    {
        // Process joystick input
        ProcessJoystickInput();

        // Update UI
        UpdateUI();
    }

    void ProcessJoystickInput()
    {
        // Process left joystick for movement
        if (leftJoystickActive)
        {
            float linearVel = leftJoystickInput.y * robotController.maxVelocity;
            float angularVel = leftJoystickInput.x * robotController.maxAngularVelocity;

            SendVelocityCommand(linearVel, angularVel);

            // Reset joystick position
            if (leftJoystickInput.magnitude < joystickDeadZone)
            {
                leftJoystickActive = false;
            }
        }

        // Process right joystick for arm control (if available)
        if (rightJoystickActive && robotController.joints.Length > 2)
        {
            // Use right joystick for arm control
            float joint1Vel = rightJoystickInput.y * 0.5f; // Scale appropriately
            float joint2Vel = rightJoystickInput.x * 0.5f;

            // Apply to arm joints
            if (robotController.joints.Length > 2)
            {
                robotController.joints[2].Rotate(Vector3.up, joint1Vel * Time.deltaTime);
            }
            if (robotController.joints.Length > 3)
            {
                robotController.joints[3].Rotate(Vector3.forward, joint2Vel * Time.deltaTime);
            }

            // Reset joystick position
            if (rightJoystickInput.magnitude < joystickDeadZone)
            {
                rightJoystickActive = false;
            }
        }
    }

    void SendVelocityCommand(float linearVel, float angularVel)
    {
        // Send command to robot
        if (robotController != null)
        {
            robotController.linearVelocity = linearVel;
            robotController.angularVelocity = angularVel;

            // Add to command history
            string command = $"Move: linear={linearVel:F2}, angular={angularVel:F2}";
            AddToCommandHistory(command);
        }
    }

    void ExecuteCommand(string command)
    {
        Debug.Log($"Executing command: {command}");

        switch (command.ToLower())
        {
            case "stop":
                SendVelocityCommand(0, 0);
                break;
            case "home":
                // Send robot to home position
                SendHomeCommand();
                break;
            case "dock":
                // Send robot to dock
                SendDockCommand();
                break;
            case "emergency stop":
                SendEmergencyStop();
                break;
            default:
                // Handle custom commands
                SendCustomCommand(command);
                break;
        }

        AddToCommandHistory(command);
    }

    void SendHomeCommand()
    {
        // Send robot to home position
        Debug.Log("Sending robot to home position");
    }

    void SendDockCommand()
    {
        // Send robot to docking station
        Debug.Log("Sending robot to docking station");
    }

    void SendEmergencyStop()
    {
        // Emergency stop - halt all movement
        if (robotController != null)
        {
            robotController.linearVelocity = 0;
            robotController.angularVelocity = 0;
        }
        Debug.Log("EMERGENCY STOP ACTIVATED");
    }

    void SendCustomCommand(string command)
    {
        // Handle custom commands
        Debug.Log($"Sending custom command: {command}");
    }

    void AddToCommandHistory(string command)
    {
        commandHistory.Insert(0, $"{System.DateTime.Now:HH:mm:ss} - {command}");

        // Limit history size
        if (commandHistory.Count > maxCommandHistory)
        {
            commandHistory.RemoveAt(maxCommandHistory);
        }

        UpdateCommandHistoryDisplay();
    }

    void UpdateCommandHistoryDisplay()
    {
        if (commandHistoryText != null)
        {
            commandHistoryText.text = string.Join("\n", commandHistory.ToArray());
        }
    }

    void UpdateUI()
    {
        // Update robot status display
        if (robotController != null)
        {
            // Update with current robot state
        }
    }

    void SetupEventHandlers()
    {
        // Add event handlers for various UI elements
    }

    void LoadPreferences()
    {
        // Load saved interface preferences
        Debug.Log("Loading interface preferences...");
    }

    public void ToggleControlPanel()
    {
        mainControlPanel.SetActive(!mainControlPanel.activeSelf);
    }

    public void ToggleCameraFeed()
    {
        cameraFeedPanel.SetActive(!cameraFeedPanel.activeSelf);
    }

    public void ToggleJoystick()
    {
        joystickPanel.SetActive(!joystickPanel.activeSelf);
    }
}
```

## Multi-User Collaboration System

### Shared Virtual Environment

```csharp
using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class MultiUserCollaborationSystem : MonoBehaviour
{
    [Header("User Management")]
    public string currentUserId;
    public string currentUserName;
    public Color userColor = Color.blue;

    [Header("Network Configuration")]
    public string serverAddress = "ws://localhost:8080";
    public int maxUsers = 10;

    [Header("Avatar Prefabs")]
    public GameObject userAvatarPrefab;
    public GameObject cursorPrefab;

    [Header("Interaction Objects")]
    public List<GameObject> sharedObjects;

    // User tracking
    private Dictionary<string, GameObject> userAvatars = new Dictionary<string, GameObject>();
    private Dictionary<string, Vector3> userPositions = new Dictionary<string, Vector3>();
    private Dictionary<string, Quaternion> userRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, string> userNames = new Dictionary<string, string>();

    // Network messaging
    private bool isConnected = false;

    void Start()
    {
        InitializeCollaborationSystem();
        ConnectToServer();
    }

    void InitializeCollaborationSystem()
    {
        // Generate unique user ID if not set
        if (string.IsNullOrEmpty(currentUserId))
        {
            currentUserId = System.Guid.NewGuid().ToString();
        }

        // Set up local avatar
        SetupLocalAvatar();
    }

    void SetupLocalAvatar()
    {
        // Create local user avatar
        GameObject localAvatar = Instantiate(userAvatarPrefab);
        localAvatar.name = $"User_{currentUserId}";
        localAvatar.GetComponent<Renderer>().material.color = userColor;

        // Add user identification
        var userIdComponent = localAvatar.AddComponent<UserIdentity>();
        userIdComponent.userId = currentUserId;
        userIdComponent.userName = currentUserName;
        userIdComponent.userColor = userColor;
    }

    void ConnectToServer()
    {
        // In a real implementation, this would connect to a WebSocket server
        // For now, we'll simulate connection
        isConnected = true;
        Debug.Log($"Connected to collaboration server: {serverAddress}");

        // Start periodic updates
        InvokeRepeating("SendPositionUpdate", 0.1f, 0.1f);
    }

    void Update()
    {
        // Update remote user positions
        UpdateRemoteUsers();

        // Handle local user input
        HandleLocalInput();
    }

    void HandleLocalInput()
    {
        // Handle input that should be shared with other users
        if (Input.GetKeyDown(KeyCode.Space))
        {
            // Share interaction with other users
            ShareInteraction("space_pressed", transform.position);
        }

        // Handle object selection
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit))
            {
                if (sharedObjects.Contains(hit.collider.gameObject))
                {
                    // Share object interaction
                    ShareInteraction("object_selected", hit.collider.gameObject.name);
                }
            }
        }
    }

    void SendPositionUpdate()
    {
        if (!isConnected) return;

        // Send current position to server
        var positionData = new
        {
            userId = currentUserId,
            userName = currentUserName,
            position = transform.position,
            rotation = transform.rotation,
            timestamp = System.DateTime.UtcNow.Ticks
        };

        // In a real implementation, send this via WebSocket
        string jsonData = JsonUtility.ToJson(positionData);
        Debug.Log($"Sending position update: {jsonData}");

        // Simulate sending to server
        SimulateServerMessage(jsonData);
    }

    void ReceiveUserUpdate(string jsonData)
    {
        // Parse received user update
        var userData = JsonUtility.FromJson<UserData>(jsonData);

        // Update remote user
        UpdateRemoteUser(userData.userId, userData.position, userData.rotation, userData.userName);
    }

    void UpdateRemoteUser(string userId, Vector3 position, Quaternion rotation, string userName)
    {
        if (userId == currentUserId) return; // Don't update local user

        // Create or update avatar
        if (!userAvatars.ContainsKey(userId))
        {
            CreateRemoteUserAvatar(userId, userName);
        }

        // Update position and rotation
        userPositions[userId] = position;
        userRotations[userId] = rotation;
        userNames[userId] = userName;

        // Update avatar transform
        userAvatars[userId].transform.position = position;
        userAvatars[userId].transform.rotation = rotation;

        // Update user name display
        var nameDisplay = userAvatars[userId].GetComponentInChildren<TextMesh>();
        if (nameDisplay != null)
        {
            nameDisplay.text = userName;
        }
    }

    void CreateRemoteUserAvatar(string userId, string userName)
    {
        GameObject avatar = Instantiate(userAvatarPrefab);
        avatar.name = $"User_{userId}";

        // Set user-specific color
        Color userColor = GetUserColor(userId);
        avatar.GetComponent<Renderer>().material.color = userColor;

        // Add name display
        GameObject nameObject = new GameObject("UserName");
        nameObject.transform.SetParent(avatar.transform);
        nameObject.transform.localPosition = new Vector3(0, 2f, 0);
        TextMesh nameText = nameObject.AddComponent<TextMesh>();
        nameText.text = userName;
        nameText.fontSize = 24;
        nameText.alignment = TextAlignment.Center;

        // Store reference
        userAvatars[userId] = avatar;
        userNames[userId] = userName;
    }

    Color GetUserColor(string userId)
    {
        // Generate consistent color based on user ID
        int hash = userId.GetHashCode();
        float r = (hash & 0xFF) / 255.0f;
        float g = ((hash >> 8) & 0xFF) / 255.0f;
        float b = ((hash >> 16) & 0xFF) / 255.0f;
        return new Color(r, g, b);
    }

    void UpdateRemoteUsers()
    {
        // Smoothly interpolate remote user positions
        foreach (string userId in userPositions.Keys.ToList())
        {
            if (userAvatars.ContainsKey(userId))
            {
                // Interpolate to target position
                userAvatars[userId].transform.position = Vector3.Lerp(
                    userAvatars[userId].transform.position,
                    userPositions[userId],
                    Time.deltaTime * 10f // Smooth interpolation
                );

                // Interpolate rotation
                userAvatars[userId].transform.rotation = Quaternion.Slerp(
                    userAvatars[userId].transform.rotation,
                    userRotations[userId],
                    Time.deltaTime * 10f
                );
            }
        }
    }

    void ShareInteraction(string interactionType, object data)
    {
        var interactionData = new
        {
            userId = currentUserId,
            interactionType = interactionType,
            data = data.ToString(),
            timestamp = System.DateTime.UtcNow.Ticks
        };

        string jsonData = JsonUtility.ToJson(interactionData);
        Debug.Log($"Sharing interaction: {jsonData}");

        // In real implementation, send to server
        SimulateServerMessage(jsonData);
    }

    void ReceiveInteraction(string jsonData)
    {
        var interaction = JsonUtility.FromJson<InteractionData>(jsonData);

        // Handle interaction based on type
        switch (interaction.interactionType)
        {
            case "space_pressed":
                HandleSpacePress(interaction.userId, JsonUtility.FromJson<Vector3>(interaction.data));
                break;
            case "object_selected":
                HandleObjectSelection(interaction.userId, interaction.data);
                break;
            default:
                Debug.Log($"Unknown interaction: {interaction.interactionType}");
                break;
        }
    }

    void HandleSpacePress(string userId, Vector3 position)
    {
        Debug.Log($"{userNames[userId]} pressed space at {position}");

        // Visual effect at position
        if (userAvatars.ContainsKey(userId))
        {
            GameObject effect = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            effect.transform.position = position + Vector3.up * 0.5f;
            effect.transform.localScale = Vector3.one * 0.2f;
            Destroy(effect, 1.0f);
        }
    }

    void HandleObjectSelection(string userId, string objectName)
    {
        Debug.Log($"{userNames[userId]} selected object: {objectName}");

        // Highlight selected object
        GameObject selectedObject = sharedObjects.FirstOrDefault(obj => obj.name == objectName);
        if (selectedObject != null)
        {
            // Add temporary highlight
            Renderer renderer = selectedObject.GetComponent<Renderer>();
            if (renderer != null)
            {
                Material originalMaterial = renderer.material;
                renderer.material.color = Color.yellow;

                // Reset after delay
                StartCoroutine(ResetMaterial(renderer, originalMaterial, 2.0f));
            }
        }
    }

    IEnumerator ResetMaterial(Renderer renderer, Material originalMaterial, float delay)
    {
        yield return new WaitForSeconds(delay);
        renderer.material = originalMaterial;
    }

    void SimulateServerMessage(string jsonData)
    {
        // Simulate receiving a message from the server
        // In real implementation, this would be triggered by actual network events
        if (jsonData.Contains("position")) // It's a position update
        {
            ReceiveUserUpdate(jsonData);
        }
        else if (jsonData.Contains("interactionType")) // It's an interaction
        {
            ReceiveInteraction(jsonData);
        }
    }

    void OnDestroy()
    {
        // Clean up network connection
        if (isConnected)
        {
            // Send disconnect message
            CancelInvoke("SendPositionUpdate");
        }
    }
}

[System.Serializable]
public class UserData
{
    public string userId;
    public string userName;
    public Vector3 position;
    public Quaternion rotation;
}

[System.Serializable]
public class InteractionData
{
    public string userId;
    public string interactionType;
    public string data;
}

// Component to identify user avatars
public class UserIdentity : MonoBehaviour
{
    public string userId;
    public string userName;
    public Color userColor;
}
```

## Physics Simulation and Dynamics

### Advanced Physics for Robot Simulation

```csharp
using UnityEngine;
using System.Collections;

public class AdvancedRobotPhysics : MonoBehaviour
{
    [Header("Robot Physical Properties")]
    public float robotMass = 50.0f;
    public float wheelRadius = 0.1f;
    public float trackWidth = 0.5f; // Distance between wheels
    public float maxTorque = 100.0f;

    [Header("Friction and Terrain")]
    public float groundFriction = 0.8f;
    public float wheelFriction = 0.9f;
    public PhysicMaterial[] terrainMaterials;

    [Header("Sensors")]
    public bool enableCollisionSensors = true;
    public bool enableProximitySensors = true;
    public float proximityRange = 1.0f;

    // Robot components
    private Rigidbody robotBody;
    private WheelCollider[] wheelColliders;
    private Transform[] wheelVisuals;

    // Control inputs
    private float motorTorque = 0f;
    private float steeringAngle = 0f;

    // Sensor data
    private bool[] collisionSensors;
    private float[] proximitySensors;

    void Start()
    {
        InitializePhysicsComponents();
        SetupSensors();
    }

    void InitializePhysicsComponents()
    {
        // Get robot rigidbody
        robotBody = GetComponent<Rigidbody>();
        if (robotBody == null)
        {
            robotBody = gameObject.AddComponent<Rigidbody>();
        }

        // Configure rigidbody
        robotBody.mass = robotMass;
        robotBody.drag = 0.1f; // Air resistance
        robotBody.angularDrag = 0.05f; // Rotational resistance
        robotBody.interpolation = RigidbodyInterpolation.Interpolate;

        // Find wheel colliders
        wheelColliders = GetComponentsInChildren<WheelCollider>();
        wheelVisuals = new Transform[wheelColliders.Length];

        for (int i = 0; i < wheelColliders.Length; i++)
        {
            // Configure wheel collider
            ConfigureWheelCollider(wheelColliders[i]);

            // Find corresponding visual wheel
            Transform visualWheel = FindVisualWheel(wheelColliders[i].transform);
            if (visualWheel != null)
            {
                wheelVisuals[i] = visualWheel;
            }
        }
    }

    void ConfigureWheelCollider(WheelCollider collider)
    {
        // Configure suspension
        JointSpring suspension = collider.suspensionSpring;
        suspension.spring = 40000f;
        suspension.damper = 4000f;
        suspension.targetPosition = 0.5f;
        collider.suspensionSpring = suspension;

        // Configure friction
        WheelFrictionCurve forwardFriction = collider.forwardFriction;
        forwardFriction.extremumSlip = 0.4f;
        forwardFriction.extremumValue = 1f;
        forwardFriction.asymptoteSlip = 0.8f;
        forwardFriction.asymptoteValue = 0.5f;
        collider.forwardFriction = forwardFriction;

        WheelFrictionCurve sidewaysFriction = collider.sidewaysFriction;
        sidewaysFriction.extremumSlip = 0.2f;
        sidewaysFriction.extremumValue = 1f;
        sidewaysFriction.asymptoteSlip = 0.5f;
        sidewaysFriction.asymptoteValue = 0.75f;
        collider.sidewaysFriction = sidewaysFriction;

        // Set wheel properties
        collider.radius = wheelRadius;
        collider.mass = 5f;
        collider.wheelDampingRate = 1.5f;
    }

    Transform FindVisualWheel(Transform wheelColliderTransform)
    {
        // Look for visual wheel in children
        foreach (Transform child in wheelColliderTransform.parent)
        {
            if (child != wheelColliderTransform && child.GetComponent<MeshFilter>() != null)
            {
                return child;
            }
        }
        return null;
    }

    void SetupSensors()
    {
        if (enableCollisionSensors)
        {
            collisionSensors = new bool[wheelColliders.Length];
        }

        if (enableProximitySensors)
        {
            proximitySensors = new float[8]; // 8 proximity sensors around the robot
        }
    }

    void FixedUpdate()
    {
        // Apply wheel forces
        ApplyWheelForces();

        // Update wheel visuals
        UpdateWheelVisuals();

        // Update sensors
        UpdateSensors();
    }

    void ApplyWheelForces()
    {
        for (int i = 0; i < wheelColliders.Length; i++)
        {
            WheelCollider wheel = wheelColliders[i];

            // Apply motor torque
            wheel.motorTorque = motorTorque;

            // Apply steering (for front wheels in typical setup)
            if (i < 2) // Assuming first 2 wheels are steerable
            {
                wheel.steerAngle = steeringAngle;
            }

            // Apply brake if needed
            if (motorTorque < 0)
            {
                wheel.brakeTorque = Mathf.Abs(motorTorque) * 0.5f;
            }
        }
    }

    void UpdateWheelVisuals()
    {
        for (int i = 0; i < wheelColliders.Length && i < wheelVisuals.Length; i++)
        {
            if (wheelVisuals[i] != null)
            {
                WheelCollider wheel = wheelColliders[i];
                Vector3 position;
                Quaternion rotation;

                // Get wheel position and rotation from collider
                wheel.GetWorldPose(out position, out rotation);

                // Update visual wheel
                wheelVisuals[i].position = position;
                wheelVisuals[i].rotation = rotation;
            }
        }
    }

    void UpdateSensors()
    {
        // Update collision sensors
        if (enableCollisionSensors)
        {
            for (int i = 0; i < wheelColliders.Length; i++)
            {
                collisionSensors[i] = wheelColliders[i].isGrounded;
            }
        }

        // Update proximity sensors
        if (enableProximitySensors)
        {
            UpdateProximitySensors();
        }
    }

    void UpdateProximitySensors()
    {
        float sensorAngleStep = 360f / proximitySensors.Length;

        for (int i = 0; i < proximitySensors.Length; i++)
        {
            float angle = i * sensorAngleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, proximityRange))
            {
                proximitySensors[i] = hit.distance;
            }
            else
            {
                proximitySensors[i] = proximityRange; // No obstacle detected
            }
        }
    }

    // Public methods for controlling the robot
    public void SetMotorTorque(float torque)
    {
        motorTorque = Mathf.Clamp(torque, -maxTorque, maxTorque);
    }

    public void SetSteeringAngle(float angle)
    {
        steeringAngle = Mathf.Clamp(angle, -45f, 45f); // Max 45 degree steering
    }

    public void SetVelocity(float linearVelocity, float angularVelocity)
    {
        // Convert linear/angular velocities to wheel torques
        float leftWheelVel = linearVelocity - angularVelocity * trackWidth / 2;
        float rightWheelVel = linearVelocity + angularVelocity * trackWidth / 2;

        // Convert to torques (simplified)
        float leftTorque = leftWheelVel * 10f; // Scale factor
        float rightTorque = rightWheelVel * 10f;

        // Apply to wheels (assuming differential drive)
        if (wheelColliders.Length >= 2)
        {
            // This is a simplification - in practice you'd use proper kinematics
            SetMotorTorque((leftTorque + rightTorque) / 2); // Average for both wheels
        }
    }

    // Sensor access methods
    public bool[] GetCollisionSensors()
    {
        return collisionSensors;
    }

    public float[] GetProximitySensors()
    {
        return proximitySensors;
    }

    public Vector3 GetRobotVelocity()
    {
        return robotBody.velocity;
    }

    public float GetRobotAngularVelocity()
    {
        return robotBody.angularVelocity.y; // Around Y-axis
    }

    // Collision detection
    void OnCollisionEnter(Collision collision)
    {
        Debug.Log($"Robot collided with {collision.gameObject.name}");

        // Send collision event to ROS or other systems
        if (enableCollisionSensors)
        {
            // Update collision state and potentially send ROS message
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Obstacle"))
        {
            Debug.Log($"Robot proximity alert: {other.name}");
        }
    }
}
```

## Best Practices for Unity Robotics

1. **Performance Optimization**: Use appropriate LODs, occlusion culling, and object pooling
2. **Physics Accuracy**: Balance realistic physics with performance requirements
3. **Network Latency**: Implement prediction and interpolation for smooth remote operation
4. **User Experience**: Design intuitive interfaces that reduce cognitive load
5. **Safety**: Implement safety features like emergency stops and collision avoidance
6. **Scalability**: Design systems that can handle multiple robots and users

## Integration with ROS 2

Unity can be integrated with ROS 2 through several approaches:

### ROS TCP Connector Implementation

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Geometry;
using RosSharp.Messages.Sensor;

public class UnityROSIntegration : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeUrl = "ws://localhost:9090";
    public float connectionTimeout = 5.0f;

    [Header("Topics")]
    public string cmdVelTopic = "/cmd_vel";
    public string jointStatesTopic = "/joint_states";
    public string laserScanTopic = "/scan";
    public string odomTopic = "/odom";

    // ROS Components
    private RosSocket rosSocket;
    private TwistSubscriber cmdVelSubscriber;
    private JointStatePublisher jointStatePublisher;

    // Robot state
    private UnityRobotController robotController;
    private AdvancedRobotPhysics physicsController;

    void Start()
    {
        robotController = GetComponent<UnityRobotController>();
        physicsController = GetComponent<AdvancedRobotPhysics>();

        ConnectToROS();
    }

    void ConnectToROS()
    {
        try
        {
            WebSocketProtocols webSocketProtocol = new WebSocketProtocols();
            rosSocket = new RosSocket(webSocketProtocol, rosBridgeUrl);

            // Subscribe to ROS topics
            SubscribeToTopics();

            // Publish to ROS topics
            SetupPublishers();

            Debug.Log("Connected to ROS Bridge");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to ROS: {e.Message}");
        }
    }

    void SubscribeToTopics()
    {
        // Subscribe to velocity commands
        rosSocket.Subscribe<Twist>(cmdVelTopic, ProcessTwistCommand);

        // Subscribe to joint state commands
        rosSocket.Subscribe<JointState>(jointStatesTopic, ProcessJointStateCommand);
    }

    void SetupPublishers()
    {
        // Initialize publishers
        jointStatePublisher = new JointStatePublisher(rosSocket, jointStatesTopic);
    }

    void ProcessTwistCommand(Twist message)
    {
        // Convert ROS Twist message to Unity robot control
        float linearX = (float)message.linear.x;
        float angularZ = (float)message.angular.z;

        // Apply to robot physics
        if (physicsController != null)
        {
            physicsController.SetVelocity(linearX, angularZ);
        }
        else if (robotController != null)
        {
            robotController.linearVelocity = linearX;
            robotController.angularVelocity = angularZ;
        }
    }

    void ProcessJointStateCommand(JointState message)
    {
        // Process joint state commands
        if (robotController != null)
        {
            for (int i = 0; i < message.name.Count; i++)
            {
                string jointName = message.name[i];
                float position = (float)message.position[i];

                // Apply joint position
                robotController.SetJointPosition(jointName, position);
            }
        }
    }

    void Update()
    {
        // Publish robot state to ROS
        PublishRobotState();
    }

    void PublishRobotState()
    {
        if (rosSocket == null || rosSocket.State != WebSocketState.Open) return;

        // Publish joint states
        if (jointStatePublisher != null && robotController != null)
        {
            jointStatePublisher.Publish(robotController.GetJointStates());
        }

        // Publish odometry (if available)
        // Publish laser scan (if available)
        // Publish other sensor data
    }

    void OnDestroy()
    {
        // Clean up ROS connection
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}

// Custom publisher for joint states
public class JointStatePublisher
{
    private RosSocket rosSocket;
    private string topicName;

    public JointStatePublisher(RosSocket socket, string topic)
    {
        rosSocket = socket;
        topicName = topic;
    }

    public void Publish(JointState jointState)
    {
        rosSocket.Publish(topicName, jointState);
    }
}
```

## Summary

This chapter covered Unity visualization and human-robot interaction, including:

- Setting up Unity for robotics applications
- Creating realistic 3D visualization environments
- Implementing human-robot interaction mechanisms
- Designing teleoperation interfaces
- Creating multi-user collaboration systems
- Advanced physics simulation for robot dynamics
- Integration with ROS 2 for real-time simulation

Unity provides a powerful platform for creating immersive and realistic robotics simulation environments that can be used for training, testing, and teleoperation of robotic systems.

## Exercises

1. Create a Unity scene with multiple robots that can be controlled simultaneously.
2. Implement gesture recognition for robot control in VR.
3. Design a collaborative workspace where multiple users can control different robots.

## Code Example: Complete Unity Robot Control System

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;

[RequireComponent(typeof(Rigidbody))]
public class CompleteUnityRobotSystem : MonoBehaviour
{
    [Header("System Configuration")]
    public string robotId = "robot_01";
    public bool enableAIControl = false;
    public bool enableManualControl = true;

    [Header("Physical Properties")]
    public float mass = 50f;
    public float maxSpeed = 2f;
    public float maxAngularSpeed = 90f; // degrees per second

    [Header("Control Interface")]
    public Canvas controlCanvas;
    public Slider speedSlider;
    public Slider angularSpeedSlider;
    public Button emergencyStopButton;
    public Text statusText;

    [Header("Sensors")]
    public bool enableLidar = true;
    public bool enableCamera = true;
    public float sensorRange = 10f;

    // Internal components
    private Rigidbody rb;
    private AdvancedRobotPhysics physics;
    private UnityROSIntegration rosIntegration;
    private List<GameObject> detectedObjects = new List<GameObject>();

    // Control state
    private Vector3 targetVelocity = Vector3.zero;
    private float targetAngularVelocity = 0f;
    private bool emergencyStopActive = false;

    void Start()
    {
        InitializeRobotSystem();
        SetupUI();
    }

    void InitializeRobotSystem()
    {
        // Get components
        rb = GetComponent<Rigidbody>();
        physics = GetComponent<AdvancedRobotPhysics>();
        rosIntegration = GetComponent<UnityROSIntegration>();

        // Configure physics
        rb.mass = mass;
        rb.drag = 0.1f;
        rb.angularDrag = 0.05f;

        // Initialize sensors
        if (enableLidar)
        {
            SetupLidarSimulation();
        }

        if (enableCamera)
        {
            SetupCameraSimulation();
        }

        // Setup emergency stop
        if (emergencyStopButton != null)
        {
            emergencyStopButton.onClick.AddListener(EmergencyStop);
        }

        Debug.Log($"Robot {robotId} initialized");
    }

    void SetupUI()
    {
        if (speedSlider != null)
        {
            speedSlider.onValueChanged.AddListener(OnSpeedChanged);
        }

        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.onValueChanged.AddListener(OnAngularSpeedChanged);
        }
    }

    void SetupLidarSimulation()
    {
        // In a real implementation, this would set up LiDAR simulation
        Debug.Log("LiDAR simulation enabled");
    }

    void SetupCameraSimulation()
    {
        // In a real implementation, this would set up camera simulation
        Debug.Log("Camera simulation enabled");
    }

    void Update()
    {
        // Handle manual control
        if (enableManualControl && !emergencyStopActive)
        {
            HandleManualInput();
        }

        // Apply physics
        ApplyPhysics();

        // Update sensors
        UpdateSensors();

        // Update UI
        UpdateStatusDisplay();
    }

    void HandleManualInput()
    {
        // Keyboard input
        float horizontal = Input.GetAxis("Horizontal"); // A/D or arrow keys
        float vertical = Input.GetAxis("Vertical");     // W/S or arrow keys

        // Calculate target velocity
        Vector3 forwardVelocity = transform.forward * vertical * maxSpeed;
        float angularVelocity = horizontal * maxAngularSpeed * Mathf.Deg2Rad;

        targetVelocity = forwardVelocity;
        targetAngularVelocity = angularVelocity;
    }

    void ApplyPhysics()
    {
        if (emergencyStopActive)
        {
            // Apply emergency stop - gradually reduce velocity
            rb.velocity = Vector3.Lerp(rb.velocity, Vector3.zero, Time.deltaTime * 10f);
            rb.angularVelocity = Vector3.Lerp(rb.angularVelocity, Vector3.zero, Time.deltaTime * 10f);
            return;
        }

        // Apply target velocity
        if (rb.velocity.magnitude < maxSpeed)
        {
            rb.AddForce(targetVelocity * 10f, ForceMode.Force);
        }

        // Apply angular velocity
        rb.AddTorque(Vector3.up * targetAngularVelocity * rb.mass, ForceMode.Force);
    }

    void UpdateSensors()
    {
        // Update proximity sensors
        if (enableLidar)
        {
            DetectNearbyObjects();
        }

        // Update camera feed
        if (enableCamera)
        {
            // This would update camera feed in a real implementation
        }
    }

    void DetectNearbyObjects()
    {
        detectedObjects.Clear();

        // Sphere cast to detect nearby objects
        Collider[] nearbyObjects = Physics.OverlapSphere(transform.position, sensorRange);

        foreach (Collider col in nearbyObjects)
        {
            if (col.gameObject != gameObject) // Don't detect self
            {
                detectedObjects.Add(col.gameObject);

                // Check for collisions
                if (Vector3.Distance(transform.position, col.transform.position) < 0.5f)
                {
                    Debug.LogWarning($"Collision detected with {col.name}");

                    // Send collision notification
                    if (rosIntegration != null)
                    {
                        // Send collision message via ROS
                    }
                }
            }
        }
    }

    void UpdateStatusDisplay()
    {
        if (statusText != null)
        {
            statusText.text = $"Robot: {robotId}\n" +
                             $"Status: {(emergencyStopActive ? "EMERGENCY STOP" : "ACTIVE")}\n" +
                             $"Speed: {rb.velocity.magnitude:F2} m/s\n" +
                             $"Angular: {rb.angularVelocity.y:F2} rad/s\n" +
                             $"Detected: {detectedObjects.Count} objects";
        }
    }

    public void SetTargetVelocity(Vector3 velocity)
    {
        targetVelocity = velocity.normalized * Mathf.Min(velocity.magnitude, maxSpeed);
    }

    public void SetTargetAngularVelocity(float angularVel)
    {
        targetAngularVelocity = Mathf.Clamp(angularVel, -maxAngularSpeed * Mathf.Deg2Rad, maxAngularSpeed * Mathf.Deg2Rad);
    }

    public void EmergencyStop()
    {
        emergencyStopActive = true;
        Debug.Log("EMERGENCY STOP ACTIVATED");

        // Send emergency stop to ROS
        if (rosIntegration != null)
        {
            // Send emergency stop message via ROS
        }
    }

    public void ResumeOperation()
    {
        emergencyStopActive = false;
        Debug.Log("Operation resumed");
    }

    void OnSpeedChanged(float value)
    {
        maxSpeed = value;
    }

    void OnAngularSpeedChanged(float value)
    {
        maxAngularSpeed = value;
    }

    // AI Control Methods
    public void EnableAIControl()
    {
        enableAIControl = true;
        enableManualControl = false;
    }

    public void DisableAIControl()
    {
        enableAIControl = false;
        enableManualControl = true;
    }

    public void SetAICommand(Vector3 targetPosition)
    {
        // Simple navigation to target
        Vector3 direction = (targetPosition - transform.position).normalized;
        float distance = Vector3.Distance(transform.position, targetPosition);

        if (distance > 0.5f) // If not at target
        {
            SetTargetVelocity(direction * maxSpeed);
        }
        else
        {
            SetTargetVelocity(Vector3.zero); // Stop when reached
        }
    }

    // Gizmos for visualization
    void OnDrawGizmosSelected()
    {
        if (enableLidar)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(transform.position, sensorRange);
        }
    }
}
```