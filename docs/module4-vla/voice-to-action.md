---
title: Voice-to-Action: Using Whisper for Commands
description: Understanding voice-to-action systems using OpenAI Whisper and command processing
sidebar_position: 1
---

# Voice-to-Action: Using Whisper for Commands

## Learning Objectives

By the end of this chapter, you will be able to:
1. Implement speech-to-text using OpenAI Whisper
2. Process voice commands for robotic action execution
3. Design a command interpretation pipeline
4. Integrate voice commands with robot control systems

## Introduction to Voice-to-Action Systems

Voice-to-action systems bridge the gap between human language and robotic action execution. These systems typically follow this pipeline:

1. **Speech Recognition**: Convert spoken language to text
2. **Natural Language Understanding**: Interpret the meaning of the command
3. **Action Mapping**: Map understood commands to robot actions
4. **Action Execution**: Execute the mapped actions on the robot

OpenAI Whisper has become a popular choice for speech recognition due to its robustness across different accents, languages, and audio conditions.

## OpenAI Whisper Integration

### Basic Whisper Implementation

```python
import openai
import whisper
import torch
import numpy as np
import pyaudio
import wave
import threading
import queue
import time
from typing import Optional, Dict, Any

class WhisperSpeechToText:
    def __init__(self, model_name: str = "base", use_api: bool = False):
        """
        Initialize Whisper-based speech-to-text system
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            use_api: Whether to use OpenAI API instead of local model
        """
        self.use_api = use_api

        if use_api:
            # Using OpenAI API
            self.model_name = "whisper-1"
        else:
            # Using local model
            self.model = whisper.load_model(model_name)

    def transcribe_audio_file(self, audio_file_path: str) -> str:
        """Transcribe an audio file using Whisper"""
        if self.use_api:
            with open(audio_file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model=self.model_name,
                    file=audio_file
                )
            return transcript.text
        else:
            result = self.model.transcribe(audio_file_path)
            return result["text"]

    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe raw audio data using Whisper"""
        if self.use_api:
            # For API, we need to save to file first
            temp_filename = "temp_audio.wav"
            self.save_audio_to_file(audio_data, sample_rate, temp_filename)
            result = self.transcribe_audio_file(temp_filename)
            import os
            os.remove(temp_filename)  # Clean up temp file
            return result
        else:
            # Convert numpy array to appropriate format for local model
            # Whisper expects audio at 16kHz
            if sample_rate != 16000:
                # Resample if needed (simplified - in practice use proper resampling)
                audio_data = self.resample_audio(audio_data, sample_rate, 16000)

            result = self.model.transcribe(audio_data)
            return result["text"]

    def save_audio_to_file(self, audio_data: np.ndarray, sample_rate: int, filename: str):
        """Save audio data to WAV file"""
        import scipy.io.wavfile as wavfile
        wavfile.write(filename, sample_rate, (audio_data * 32767).astype(np.int16))

    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (use librosa or scipy for better quality)"""
        # This is a simplified implementation
        # In practice, use proper resampling libraries
        if orig_sr == target_sr:
            return audio_data

        num_samples = int(len(audio_data) * target_sr / orig_sr)
        resampled = np.interp(
            np.linspace(0, len(audio_data) - 1, num_samples),
            np.arange(len(audio_data)),
            audio_data
        )
        return resampled

# Example usage
def example_transcription():
    stt = WhisperSpeechToText(model_name="base")

    # Example: Transcribe a sample audio file
    # transcript = stt.transcribe_audio_file("sample_command.wav")
    # print(f"Transcribed: {transcript}")

    print("WhisperSpeechToText initialized")
```

### Real-time Audio Capture and Transcription

```python
import pyaudio
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class AudioConfig:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    record_seconds: int = 5

class RealTimeWhisperTranscriber:
    def __init__(self, stt_engine: WhisperSpeechToText, audio_config: AudioConfig = None):
        self.stt_engine = stt_engine
        self.audio_config = audio_config or AudioConfig()
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.result_callback: Optional[Callable[[str], None]] = None

    def start_listening(self, result_callback: Callable[[str], None] = None):
        """Start real-time audio capture and transcription"""
        self.result_callback = result_callback
        self.is_listening = True

        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self._capture_audio)
        self.capture_thread.start()

        print("Started real-time listening...")

    def stop_listening(self):
        """Stop real-time audio capture"""
        self.is_listening = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()

    def _capture_audio(self):
        """Capture audio in a separate thread"""
        stream = self.audio.open(
            format=self.audio_config.format,
            channels=self.audio_config.channels,
            rate=self.audio_config.rate,
            input=True,
            frames_per_buffer=self.audio_config.chunk
        )

        print("Audio stream opened, capturing...")

        while self.is_listening:
            # Read audio data
            frames = []
            for _ in range(0, int(self.audio_config.rate / self.audio_config.chunk * self.audio_config.record_seconds)):
                data = stream.read(self.audio_config.chunk)
                frames.append(data)

            # Convert to numpy array
            audio_data = self._frames_to_numpy(frames)

            # Transcribe in a separate thread to avoid blocking audio capture
            transcribe_thread = threading.Thread(
                target=self._transcribe_and_callback,
                args=(audio_data,)
            )
            transcribe_thread.start()

        stream.stop_stream()
        stream.close()

    def _frames_to_numpy(self, frames):
        """Convert audio frames to numpy array"""
        import struct
        # Concatenate all frames
        audio_string = b''.join(frames)

        # Convert to numpy array
        audio_data = np.frombuffer(audio_string, dtype=np.int16)

        # Normalize to [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32768.0

        return audio_data

    def _transcribe_and_callback(self, audio_data):
        """Transcribe audio and call result callback"""
        try:
            # Save to temp file for Whisper (or use in-memory approach)
            temp_filename = f"temp_{int(time.time())}.wav"
            self._save_audio_data(audio_data, temp_filename)

            # Transcribe
            transcript = self.stt_engine.transcribe_audio_file(temp_filename)

            # Clean up temp file
            import os
            os.remove(temp_filename)

            # Call result callback if provided
            if self.result_callback:
                self.result_callback(transcript)
            else:
                print(f"Transcribed: {transcript}")

        except Exception as e:
            print(f"Error during transcription: {e}")

    def _save_audio_data(self, audio_data, filename):
        """Save audio data to WAV file"""
        import wave

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.audio_config.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_config.format))
            wf.setframerate(self.audio_config.rate)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

# Example usage
def command_handler(transcript: str):
    print(f"Received command: {transcript}")
    # Process the command here

def example_real_time():
    stt = WhisperSpeechToText(model_name="base")
    transcriber = RealTimeWhisperTranscriber(stt)

    # Start listening
    transcriber.start_listening(command_handler)

    # Listen for 30 seconds
    time.sleep(30)

    # Stop listening
    transcriber.stop_listening()
```

## Command Interpretation Pipeline

### Natural Language Understanding for Robot Commands

```python
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    action: str
    target: str
    attributes: Dict[str, str]
    confidence: float

class CommandInterpreter:
    def __init__(self):
        # Define action patterns and their corresponding robot actions
        self.action_patterns = {
            'move_to': [
                r'move to (.+)',
                r'go to (.+)',
                r'navigate to (.+)',
                r'go over to (.+)'
            ],
            'pick_up': [
                r'pick up (.+)',
                r'grab (.+)',
                r'get (.+)',
                r'take (.+)'
            ],
            'place': [
                r'place (.+) on (.+)',
                r'put (.+) on (.+)',
                r'drop (.+) on (.+)'
            ],
            'look_at': [
                r'look at (.+)',
                r'face (.+)',
                r'turn to (.+)'
            ],
            'follow': [
                r'follow (.+)',
                r'go after (.+)',
                r'chase (.+)'
            ],
            'stop': [
                r'stop',
                r'freeze',
                r'hold position'
            ],
            'greet': [
                r'say hello to (.+)',
                r'greet (.+)',
                r'wave to (.+)'
            ]
        }

        # Location/position patterns
        self.location_patterns = {
            'kitchen': [r'kitchen', r'cooking area'],
            'living_room': [r'living room', r'livingroom', r'sitting area'],
            'bedroom': [r'bedroom', r'bed room', r'sleeping area'],
            'bathroom': [r'bathroom', r'bath room'],
            'office': [r'office', r'study'],
            'entrance': [r'entrance', r'front door', r'entry'],
            'dining_room': [r'dining room', r'dining area', r'dinner table']
        }

        # Object patterns
        self.object_patterns = {
            'water': [r'water', r'water bottle', r'water glass'],
            'coffee': [r'coffee', r'coffee cup', r'coffee mug'],
            'book': [r'book', r'novel', r'booklet'],
            'phone': [r'phone', r'mobile', r'cell phone'],
            'keys': [r'keys', r'keychain', r'key'],
            'medicine': [r'medicine', r'pills', r'drugs']
        }

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """
        Parse a natural language command and extract structured information
        """
        text = text.lower().strip()

        # Try to match each action pattern
        for action, patterns in self.action_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # Extract the main target/object
                    if action == 'place':
                        # Special case for place action (has two objects)
                        if len(match.groups()) >= 2:
                            target = match.group(1)  # object to place
                            destination = match.group(2)  # where to place
                            attributes = {'destination': destination}
                        else:
                            continue  # Skip if we don't have both objects
                    else:
                        target = match.group(1) if match.groups() else ''
                        attributes = {}

                    # Try to identify location in the target
                    location = self._identify_location(target)
                    if location:
                        attributes['location'] = location
                        target = self._remove_location_from_target(target, location)

                    # Try to identify object in the target
                    obj = self._identify_object(target)
                    if obj:
                        attributes['object'] = obj

                    # Calculate confidence based on pattern match
                    confidence = self._calculate_confidence(pattern, text)

                    return ParsedCommand(
                        action=action,
                        target=target,
                        attributes=attributes,
                        confidence=confidence
                    )

        # If no pattern matches, return None
        return None

    def _identify_location(self, text: str) -> Optional[str]:
        """Identify if the text contains a known location"""
        for location, patterns in self.location_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return location
        return None

    def _identify_object(self, text: str) -> Optional[str]:
        """Identify if the text contains a known object"""
        for obj, patterns in self.object_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return obj
        return None

    def _remove_location_from_target(self, target: str, location: str) -> str:
        """Remove location from target string"""
        for pattern in self.location_patterns[location]:
            target = re.sub(pattern, '', target, flags=re.IGNORECASE).strip()
        return target

    def _calculate_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence based on pattern match quality"""
        # Simple confidence calculation
        # In practice, this could be more sophisticated
        match_ratio = len(pattern) / len(text) if text else 0
        return min(1.0, match_ratio + 0.5)  # Add base confidence

# Example usage
def example_command_interpretation():
    interpreter = CommandInterpreter()

    test_commands = [
        "Move to the kitchen",
        "Pick up the water bottle",
        "Place the coffee mug on the table",
        "Look at the person by the door",
        "Follow the dog",
        "Say hello to my friend"
    ]

    for command in test_commands:
        parsed = interpreter.parse_command(command)
        if parsed:
            print(f"Command: '{command}'")
            print(f"  Action: {parsed.action}")
            print(f"  Target: {parsed.target}")
            print(f"  Attributes: {parsed.attributes}")
            print(f"  Confidence: {parsed.confidence:.2f}")
            print()
        else:
            print(f"Could not parse: '{command}'")
```

## Voice Command to Robot Action Mapping

### Action Execution System

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import time

class RobotActionExecutor(ABC):
    """Abstract base class for robot action execution"""

    @abstractmethod
    async def execute_move_to(self, location: str) -> bool:
        """Move robot to specified location"""
        pass

    @abstractmethod
    async def execute_pick_up(self, object_name: str) -> bool:
        """Pick up specified object"""
        pass

    @abstractmethod
    async def execute_place(self, object_name: str, location: str) -> bool:
        """Place object at specified location"""
        pass

    @abstractmethod
    async def execute_look_at(self, target: str) -> bool:
        """Turn robot to look at target"""
        pass

    @abstractmethod
    async def execute_follow(self, target: str) -> bool:
        """Follow specified target"""
        pass

    @abstractmethod
    async def execute_stop(self) -> bool:
        """Stop robot movement"""
        pass

    @abstractmethod
    async def execute_greet(self, target: str) -> bool:
        """Greet specified target"""
        pass

class SimulatedRobotActionExecutor(RobotActionExecutor):
    """Simulated robot action executor for testing"""

    def __init__(self):
        self.current_position = "start_position"
        self.holding_object = None
        self.is_moving = False

    async def execute_move_to(self, location: str) -> bool:
        print(f"Moving to {location}...")
        await asyncio.sleep(2)  # Simulate movement time
        self.current_position = location
        print(f"Arrived at {location}")
        return True

    async def execute_pick_up(self, object_name: str) -> bool:
        print(f"Attempting to pick up {object_name}...")
        await asyncio.sleep(1)  # Simulate picking time
        self.holding_object = object_name
        print(f"Picked up {object_name}")
        return True

    async def execute_place(self, object_name: str, location: str) -> bool:
        if self.holding_object != object_name:
            print(f"Cannot place {object_name}, not holding it")
            return False

        print(f"Placing {object_name} at {location}...")
        await asyncio.sleep(1)  # Simulate placing time
        self.holding_object = None
        print(f"Placed {object_name} at {location}")
        return True

    async def execute_look_at(self, target: str) -> bool:
        print(f"Turning to look at {target}...")
        await asyncio.sleep(0.5)  # Simulate turning time
        print(f"Looking at {target}")
        return True

    async def execute_follow(self, target: str) -> bool:
        print(f"Starting to follow {target}...")
        self.is_moving = True
        # In a real system, this would continue until stopped
        return True

    async def execute_stop(self) -> bool:
        print("Stopping robot...")
        self.is_moving = False
        print("Robot stopped")
        return True

    async def execute_greet(self, target: str) -> bool:
        print(f"Greeting {target}...")
        await asyncio.sleep(1)  # Simulate greeting action
        print(f"Greeted {target}")
        return True

class VoiceCommandProcessor:
    def __init__(self, interpreter: CommandInterpreter, executor: RobotActionExecutor):
        self.interpreter = interpreter
        self.executor = executor

    async def process_command(self, text: str) -> bool:
        """
        Process a voice command from text to robot action
        """
        print(f"Processing command: '{text}'")

        # Parse the command
        parsed_command = self.interpreter.parse_command(text)
        if not parsed_command:
            print("Could not understand the command")
            return False

        print(f"Parsed command: {parsed_command.action} with target '{parsed_command.target}'")

        # Execute the corresponding action
        action_method_name = f"execute_{parsed_command.action}"
        if hasattr(self.executor, action_method_name):
            action_method = getattr(self.executor, action_method_name)

            # Prepare arguments based on action type
            if parsed_command.action == 'move_to':
                success = await action_method(parsed_command.target)
            elif parsed_command.action == 'pick_up':
                success = await action_method(parsed_command.target)
            elif parsed_command.action == 'place':
                destination = parsed_command.attributes.get('destination', 'default_location')
                success = await action_method(parsed_command.target, destination)
            elif parsed_command.action == 'look_at':
                success = await action_method(parsed_command.target)
            elif parsed_command.action == 'follow':
                success = await action_method(parsed_command.target)
            elif parsed_command.action == 'greet':
                success = await action_method(parsed_command.target)
            elif parsed_command.action == 'stop':
                success = await action_method()
            else:
                print(f"Unknown action: {parsed_command.action}")
                return False

            if success:
                print(f"Successfully executed {parsed_command.action}")
                return True
            else:
                print(f"Failed to execute {parsed_command.action}")
                return False
        else:
            print(f"Action {parsed_command.action} not implemented")
            return False

# Example usage
async def example_voice_command_system():
    # Initialize components
    interpreter = CommandInterpreter()
    executor = SimulatedRobotActionExecutor()
    processor = VoiceCommandProcessor(interpreter, executor)

    # Test commands
    test_commands = [
        "Move to the kitchen",
        "Pick up the water bottle",
        "Place the water bottle on the table",
        "Look at the person",
        "Stop"
    ]

    for command in test_commands:
        success = await processor.process_command(command)
        print(f"Command '{command}' {'succeeded' if success else 'failed'}\n")
        await asyncio.sleep(0.5)  # Small delay between commands

# Run the example
# asyncio.run(example_voice_command_system())
```

## Integration with ROS 2

### ROS 2 Voice Command Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile
import asyncio
from threading import Thread

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Create publishers for robot control
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create subscriber for voice commands
        self.voice_subscriber = self.create_subscription(
            String,
            'voice_commands',
            self.voice_command_callback,
            10
        )

        # Create publisher for robot responses
        self.response_publisher = self.create_publisher(String, 'robot_responses', 10)

        # Initialize command processor
        self.interpreter = CommandInterpreter()
        self.executor = SimulatedRobotActionExecutor()  # Replace with real robot executor
        self.processor = VoiceCommandProcessor(self.interpreter, self.executor)

        # Store the event loop for async operations
        self.loop = asyncio.new_event_loop()
        self.loop_thread = Thread(target=self._run_event_loop, args=(self.loop,))
        self.loop_thread.start()

        self.get_logger().info("Voice Command Node initialized")

    def _run_event_loop(self, loop):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def voice_command_callback(self, msg):
        """Handle incoming voice commands"""
        command_text = msg.data
        self.get_logger().info(f"Received voice command: {command_text}")

        # Process the command asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self.processor.process_command(command_text),
            self.loop
        )

        # Add callback to handle the result
        future.add_done_callback(self._command_processed_callback)

    def _command_processed_callback(self, future):
        """Handle the result of command processing"""
        try:
            success = future.result()
            if success:
                response = "Command executed successfully"
            else:
                response = "Failed to execute command"

            # Publish response
            response_msg = String()
            response_msg.data = response
            self.response_publisher.publish(response_msg)
            self.get_logger().info(f"Published response: {response}")

        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")

    def destroy_node(self):
        """Clean up the node"""
        # Stop the event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceCommandNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Error Handling and Confidence Thresholds

### Robust Command Processing

```python
from typing import Tuple
import logging

class RobustVoiceCommandProcessor:
    def __init__(self, interpreter: CommandInterpreter, executor: RobotActionExecutor,
                 confidence_threshold: float = 0.7):
        self.interpreter = interpreter
        self.executor = executor
        self.confidence_threshold = confidence_threshold
        self.command_history = []

        # Set up logging
        self.logger = logging.getLogger(__name__)

    async def process_command_with_validation(self, text: str) -> Tuple[bool, str]:
        """
        Process a voice command with validation and error handling
        Returns: (success, response_message)
        """
        self.logger.info(f"Processing command: '{text}'")

        # First, try to parse the command
        parsed_command = self.interpreter.parse_command(text)
        if not parsed_command:
            response = "Sorry, I didn't understand that command."
            self.logger.warning(f"Could not parse command: {text}")
            return False, response

        # Check confidence threshold
        if parsed_command.confidence < self.confidence_threshold:
            response = f"I'm not confident I understood correctly. Did you mean: '{text}'?"
            self.logger.info(f"Low confidence command: {parsed_command.confidence}")
            return False, response

        # Validate the command makes sense
        is_valid, validation_msg = self._validate_command(parsed_command)
        if not is_valid:
            self.logger.warning(f"Invalid command: {validation_msg}")
            return False, validation_msg

        # Execute the command
        try:
            success = await self._execute_command_with_retry(parsed_command)
            if success:
                response = f"Okay, I will {parsed_command.action.replace('_', ' ')} {parsed_command.target or ''}".strip()
                self.command_history.append((text, parsed_command, True))
                self.logger.info(f"Successfully executed command: {parsed_command.action}")
                return True, response
            else:
                response = f"Sorry, I couldn't {parsed_command.action.replace('_', ' ')} {parsed_command.target or ''}".strip()
                self.command_history.append((text, parsed_command, False))
                self.logger.warning(f"Failed to execute command: {parsed_command.action}")
                return False, response
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            response = "Sorry, something went wrong while executing the command."
            return False, response

    def _validate_command(self, command: ParsedCommand) -> Tuple[bool, str]:
        """Validate if the command is reasonable"""
        # Check for empty targets when needed
        if command.action in ['move_to', 'pick_up', 'look_at', 'follow', 'greet'] and not command.target.strip():
            return False, "The command needs a specific target."

        # Check for required attributes
        if command.action == 'place' and 'destination' not in command.attributes:
            return False, "The place command needs both an object and a destination."

        # Add more validation rules as needed
        return True, "Command is valid"

    async def _execute_command_with_retry(self, command: ParsedCommand, max_retries: int = 3) -> bool:
        """Execute command with retry logic"""
        for attempt in range(max_retries):
            try:
                # Map command to execution
                action_method_name = f"execute_{command.action}"
                if hasattr(self.executor, action_method_name):
                    action_method = getattr(self.executor, action_method_name)

                    # Prepare and execute
                    if command.action == 'place':
                        destination = command.attributes.get('destination', 'default_location')
                        success = await action_method(command.target, destination)
                    elif command.action == 'stop':
                        success = await action_method()
                    else:
                        success = await action_method(command.target)

                    if success:
                        return True
                else:
                    self.logger.error(f"Action {command.action} not implemented")
                    return False
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    return False
                await asyncio.sleep(0.5)  # Wait before retry

        return False

# Example usage with error handling
async def example_robust_system():
    interpreter = CommandInterpreter()
    executor = SimulatedRobotActionExecutor()
    processor = RobustVoiceCommandProcessor(interpreter, executor, confidence_threshold=0.5)

    test_commands = [
        "Move to the kitchen",
        "Pick up the water bottle",
        "Invalid command that won't be understood",
        "Place the water bottle on the table",
        "Move to"  # Incomplete command
    ]

    for command in test_commands:
        success, response = await processor.process_command_with_validation(command)
        print(f"Command: '{command}'")
        print(f"Response: {response}")
        print(f"Success: {success}\n")

# Run the example
# asyncio.run(example_robust_system())
```

## Best Practices for Voice-to-Action Systems

1. **Confidence Thresholds**: Always validate command confidence to avoid misinterpretation
2. **Context Awareness**: Consider the robot's current state and environment
3. **Error Recovery**: Implement graceful degradation when commands fail
4. **Feedback**: Provide clear feedback to users about command status
5. **Privacy**: Handle voice data securely and with user consent

## Summary

This chapter covered voice-to-action systems using OpenAI Whisper for speech recognition and command processing. We implemented a complete pipeline from speech recognition to robot action execution, including natural language understanding, command interpretation, and ROS 2 integration. The system is designed to be robust with error handling and confidence thresholds.

## Exercises

1. Implement a wake word detection system that activates the voice command processor only when a specific word is spoken.
2. Add support for multi-step commands like "Go to the kitchen and pick up the water bottle."
3. How would you modify the system to handle multiple languages?

## Code Example: Complete Voice Command System

```python
import asyncio
import threading
import queue
import time
from typing import Callable, Optional

class CompleteVoiceCommandSystem:
    def __init__(self):
        self.stt_engine = WhisperSpeechToText(model_name="base")
        self.interpreter = CommandInterpreter()
        self.executor = SimulatedRobotActionExecutor()
        self.processor = RobustVoiceCommandProcessor(
            self.interpreter,
            self.executor,
            confidence_threshold=0.6
        )

        self.transcriber = RealTimeWhisperTranscriber(
            self.stt_engine,
            AudioConfig(record_seconds=3)
        )

        self.command_queue = queue.Queue()
        self.is_running = False
        self.command_thread = None

    def start_system(self):
        """Start the complete voice command system"""
        self.is_running = True

        # Start the command processing thread
        self.command_thread = threading.Thread(target=self._process_commands)
        self.command_thread.start()

        # Start listening for voice commands
        self.transcriber.start_listening(self._on_transcription)

        print("Voice Command System started")

    def stop_system(self):
        """Stop the complete voice command system"""
        self.is_running = False

        # Stop transcriber
        self.transcriber.stop_listening()

        # Wait for command thread to finish
        if self.command_thread:
            self.command_thread.join()

        print("Voice Command System stopped")

    def _on_transcription(self, transcript: str):
        """Called when a new transcription is available"""
        print(f"Heard: {transcript}")
        self.command_queue.put(transcript)

    def _process_commands(self):
        """Process commands from the queue"""
        while self.is_running:
            try:
                # Get a command from the queue (with timeout)
                command = self.command_queue.get(timeout=1.0)

                # Process the command asynchronously
                result = asyncio.run_coroutine_threadsafe(
                    self.processor.process_command_with_validation(command),
                    asyncio.new_event_loop()
                )

                # Get the result
                success, response = result.result()
                print(f"Command result: {response}")

            except queue.Empty:
                # Queue is empty, continue loop
                continue
            except Exception as e:
                print(f"Error processing command: {e}")

    def add_command_callback(self, callback: Callable[[bool, str], None]):
        """Add a callback for command results"""
        # This would be implemented to notify external systems
        pass

# Example usage
def example_complete_system():
    system = CompleteVoiceCommandSystem()

    try:
        system.start_system()

        # Let it run for a while
        time.sleep(60)  # Run for 1 minute

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        system.stop_system()

# Uncomment to run the complete system
# example_complete_system()
```