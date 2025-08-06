# PhosphoBot: AI Kinematics Example

This example demonstrates how to use AI-powered vision to control a robot arm. The system can track objects in real-time using RGB-D cameras and move the robot to interact with them.

## Prerequisites

- Python 3.6+
- A robot running the PhosphoBot server
- RGB-D camera (Record3D app on iPhone recommended)
- Anthropic API key (for AI vision features)
- Required Python packages (install via `pip install -r requirements.txt`)

## Configuration

The script `ai_kinematics.py` contains several configurable parameters:

```python
# Robot connection settings
ROBOT_ENDPOINT: str = "http://192.168.178.190/move/absolute?robot_id=0"
POSITION_TOLERANCE: float = 0.03  # meters
ORIENTATION_TOLERANCE: float = 0.2  # radians

# Camera settings
ARUCO_MARKER_ID: int = 0
ARUCO_MARKER_SIZE: float = 0.04  # meters

# AI settings
ANTHROPIC_API_KEY: str = "your-api-key-here"  # Set via environment variable
```

Modify these values according to your setup.

## How to Run

1. Make sure your robot is powered on and the PhosphoBot server is running
2. Set up your RGB-D camera (Record3D app on iPhone)
3. Set your Anthropic API key:
   ```
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Update the `ROBOT_ENDPOINT` variable in the script if needed
6. Run the script:
   ```
   python ai_kinematics.py
   ```

## What the Script Does

1. Connects to an RGB-D camera (Record3D)
2. Displays real-time RGB and depth streams
3. Tracks ArUco markers for coordinate system calibration
4. Allows manual target selection by clicking on the RGB image
5. Uses AI vision to automatically detect and track objects
6. Converts 3D positions to robot coordinates
7. Moves the robot to interact with detected objects

## Features

- **RGB-D Processing**: Real-time color and depth image processing
- **ArUco Marker Tracking**: Automatic coordinate system calibration
- **Manual Target Selection**: Click on RGB image to select targets
- **AI Vision**: Automatic object detection and tracking
- **3D Position Calculation**: Converts pixel coordinates to 3D world coordinates
- **Robot Control**: Automatic robot movement to target positions
- **Real-time Visualization**: Live display of RGB, depth, and confidence maps

## Usage Modes

### Manual Mode
- Click on the RGB image to select a target
- The system will calculate the 3D position and move the robot
- Shows depth information and coordinates in real-time

### AI Vision Mode
- Automatically detects objects based on natural language descriptions
- Continuously tracks and follows objects
- Uses AI to understand spatial relationships and object properties

## Controls

- **Mouse Click**: Select target position in manual mode
- **'q'**: Quit the application
- **'m'**: Toggle between manual and AI modes
- **'r'**: Reset robot position

## Customization

You can modify the script to change:

- The robot endpoint and connection settings
- The ArUco marker configuration
- The AI vision task descriptions
- The position and orientation tolerances
- The camera calibration parameters

## Troubleshooting

- **Camera not connecting**: Make sure Record3D app is running and connected to the same network
- **AI not working**: Check that your Anthropic API key is set correctly
- **Robot not moving**: Verify the robot endpoint URL and network connectivity
- **Poor tracking**: Ensure good lighting and clear ArUco markers

## Advanced Features

- **Coordinate System Calibration**: Uses ArUco markers for accurate 3D positioning
- **Depth Filtering**: Filters out invalid depth readings
- **Robot Frame Conversion**: Converts camera coordinates to robot coordinates
- **Confidence Visualization**: Shows AI confidence levels for object detection 