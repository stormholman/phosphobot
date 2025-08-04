# New Features: Kinematics AI & Manual Modes + MuJoCo Simulation

This document describes the new features implemented in the phosphobot system: AI-powered kinematics with vision analysis, manual kinematics control, and MuJoCo simulation with robot mirroring.

## Features Overview

### 1. Kinematics AI Mode
- **Purpose**: AI-powered object detection and robot control using RGBD camera
- **Capabilities**: 
  - Automatic object detection using vision AI
  - 3D position calculation from depth data
  - ArUco marker-based coordinate transformation
  - Direct robot movement commands
- **Use Case**: Automated picking and placing of objects

### 2. Kinematics Manual Mode
- **Purpose**: Manual control for precise positioning and testing
- **Capabilities**:
  - Real-time 3D position display on mouse hover
  - Click-to-move robot functionality
  - ArUco marker coordinate system
  - Visual crosshair on both RGB and depth views
- **Use Case**: Manual testing, calibration, and precise positioning

### 3. MuJoCo Simulation
- **Purpose**: Real-time simulation with physical robot mirroring
- **Capabilities**:
  - 3D robot simulation with MuJoCo viewer
  - Real-time mirroring from physical robot
  - Joint angle calibration and transformation
  - Interactive workspace visualization
- **Use Case**: Development, testing, and visualization without physical robot

## Setup Instructions

### Prerequisites

1. **Hardware Requirements**:
   - iPhone with Record3D app (for RGBD streaming)
   - ArUco marker (ID: 0, 4cm size)
   - Physical robot (SO-100) accessible at 192.168.178.190

2. **Software Dependencies**:
   ```bash
   pip install opencv-python numpy requests anthropic
   pip install mujoco-python  # For simulation
   ```

3. **Environment Variables**:
   ```bash
   export ANTHROPIC_API_KEY="your_api_key_here"  # For AI vision
   ```

### ArUco Marker Setup

1. **Print ArUco Marker**:
   - Generate ArUco marker with ID: 0
   - Size: 4cm x 4cm
   - Use standard ArUco dictionary (DICT_ARUCO_ORIGINAL)

2. **Position Marker**:
   - Place marker underneath robot end-effector
   - Ensure marker is visible to camera
   - Call robot initialization: `http://192.168.178.190/move/init`

3. **Verify Setup**:
   - Marker should be detected in RGB view
   - Coordinate axes should be drawn on marker
   - Status message: "Marker 0 detected and pose estimated"

## Usage Instructions

### Kinematics AI Mode

**Start Command**:
```bash
cd phosphobot/phosphobot/ai_kinematics
python main.py ai "object_description"
```

**Examples**:
```bash
python main.py ai "red cup"
python main.py ai "blue block"
python main.py ai "small object"
```

**Operation**:
1. AI analyzes first frame automatically
2. Green crosshair shows detected target
3. Robot moves to target position automatically
4. Press 'q' to quit

### Kinematics Manual Mode

**Start Command**:
```bash
cd phosphobot/phosphobot/ai_kinematics
python main.py manual
```

**Operation**:
1. Move mouse over depth window to see 3D positions
2. White crosshair appears on both RGB and depth views
3. Click on depth image to move robot to that position
4. Press 'q' to quit

### MuJoCo Simulation

**Start Command**:
```bash
cd phosphobot/phosphobot/mujoco
python main.py
```

**Operation**:
1. MuJoCo viewer opens with 3D robot model
2. Robot mirroring starts automatically
3. Physical robot movements are mirrored in simulation
4. Interactive commands available:
   - `mirror` - Start/restart mirroring
   - `stop` or `s` - Stop mirroring
   - `current` - Show current pose
   - `workspace` - Show workspace limits
   - `q` - Quit

## Coordinate Systems

### ArUco Marker Frame
- **Origin**: Top-left corner of marker
- **X-axis**: Rightward along top edge
- **Y-axis**: Downward along left edge
- **Z-axis**: Out of marker plane toward camera

### Robot Frame
- **X**: Left/right movement (negative ArUco Y)
- **Y**: Forward/backward movement (negative ArUco X)
- **Z**: Up/down movement (ArUco Z + 13cm offset)

### Workspace Limits
- **X**: -0.338 to +0.338 meters (±34cm)
- **Y**: -0.383 to -0.045 meters (4.5-38cm forward)
- **Z**: 0.096 to 0.315 meters (10-32cm high)

## Troubleshooting

### Common Issues

1. **ArUco Marker Not Detected**:
   - Check marker ID is 0
   - Ensure good lighting
   - Verify marker is 4cm size
   - Check camera focus

2. **Robot Connection Failed**:
   - Verify robot IP: 192.168.178.190
   - Check network connectivity
   - Ensure robot is powered on
   - Call initialization endpoint

3. **AI Vision Not Working**:
   - Check ANTHROPIC_API_KEY environment variable
   - Verify internet connection
   - Ensure object description is clear

4. **MuJoCo Simulation Issues**:
   - Check so100_arm.xml exists
   - Verify MuJoCo installation
   - Restart if threading conflicts occur

### Debug Information

- **Camera Intrinsics**: Printed on startup
- **Marker Detection**: Status messages in console
- **Robot Commands**: HTTP status codes displayed
- **3D Positions**: Real-time coordinates shown

## File Structure

```
phosphobot/
├── phosphobot/
│   ├── ai_kinematics/
│   │   ├── main.py          # AI & Manual kinematics
│   │   ├── rgbd_feed.py     # RGBD stream handling
│   │   └── vision_analyzer.py # AI vision analysis
│   └── mujoco/
│       ├── main.py          # MuJoCo simulation
│       └── so100_simulation.py # Robot simulation
```

## API Endpoints

### Robot Control
- **Move Absolute**: `POST http://192.168.178.190/move/absolute?robot_id=0`
- **Initialize**: `POST http://192.168.178.190/move/init`
- **Read Joints**: `POST http://192.168.178.190/joints/read`

### Payload Format
```json
{
  "x": 10.5,
  "y": -15.2,
  "z": 25.0,
  "open": 0,
  "max_trials": 100,
  "position_tolerance": 0.03,
  "orientation_tolerance": 0.2
}
```

## Performance Notes

- **AI Mode**: Processes first frame only, then exits
- **Manual Mode**: Real-time processing with mouse interaction
- **MuJoCo**: 5Hz mirroring frequency to avoid conflicts
- **Coordinate Scaling**: Automatic RGB-to-depth coordinate conversion
- **Error Handling**: Graceful degradation for connection issues 