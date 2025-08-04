# SO-100 Robotic Arm Simulation

A robust MuJoCo simulation of the SO-100 robotic arm with interactive GUI and real robot mirroring capabilities.

## Features

- **Real-time 3D visualization** with MuJoCo viewer
- **Robust threading** - no more viewer conflicts during collisions
- **Interactive banana object** for manipulation testing
- **6-DOF robotic arm** simulation with gripper
- **Real robot mirroring** support with automatic calibration
- **Thread-safe operations** with enhanced error handling
- **Web API integration** for remote control via phosphobot framework

## Requirements

```bash
pip install mujoco numpy requests
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Start

### Direct Launch
Run the integrated simulation with the MuJoCo viewer:

```bash
mjpython main.py
```

This starts the SO-100 arm simulation, opens the viewer, and automatically begins mirroring the physical robot's movements in real-time.

### Web API Control
The simulation can be controlled remotely through the phosphobot web interface:

- **Launch simulation**: `POST /simulation/launch`
- **Stop simulation**: `POST /simulation/stop`
- **Check status**: `GET /simulation/status`

The web API automatically handles process management and provides status feedback.

### Use in Your Code

```python
from so100_simulation import SO100Simulation

# Create simulation
sim = SO100Simulation()

# Set joint angles
sim.set_joint_angles([0.0, -1.0, 1.0, 0.5, -0.5, 0.0])

# Get end effector position
position = sim.get_end_effector_position()
print(f"End effector at: {position}")

# Close simulation
sim.close()
```

## File Structure

```
mujoco/
├── so100_simulation.py      # Main simulation class
├── main.py                  # Simulation bootstrap + robot mirroring
├── so100_arm.xml            # MuJoCo model definition
├── rgbd_feed.py             # Record3D RGB-D camera helper (used by AI-kin)
└── assets/                  # 3D model files, textures, etc.
```

## Web API Integration

The simulation integrates with the phosphobot web framework through REST API endpoints:

### Available Endpoints

- **`POST /simulation/launch`** - Launch MuJoCo simulation
  - Returns: Status confirmation with process PID
  - Background: Launches `mjpython main.py` in mujoco directory

- **`POST /simulation/stop`** - Stop running simulation
  - Returns: Status confirmation
  - Process: Graceful termination with 5-second timeout

- **`GET /simulation/status`** - Check simulation status
  - Returns: Current status (running/stopped) with PID if active

### API Response Format
```json
{
  "status": "ok",
  "message": "Simulation is running (PID: 12345)"
}
```

### Error Handling
- **Already running**: Returns 400 if simulation is already active
- **Process not found**: Returns 500 if main.py script is missing
- **Launch failure**: Returns 500 with error details
- **Stop timeout**: Force kills process after 5 seconds

### Integration with phosphobot
The simulation endpoints are automatically registered with the main phosphobot FastAPI application and can be accessed through the web dashboard.

## Key Components

### SO100Simulation Class
The main simulation class with robust error handling:

- **Thread-safe operations**: Eliminates viewer conflicts during collisions
- **Enhanced error handling**: Gracefully handles known threading issues
- **Command queue system**: Safe communication between threads
- **Automatic recovery**: Continues running even during OpenGL conflicts

### Robot Mirroring System
Real-time synchronization with physical robot:

- **Automatic calibration**: Hardcoded joint offsets and directions
- **Thread-safe mirroring**: 5Hz update rate to avoid conflicts
- **Error handling**: Graceful handling of connection issues
- **Joint mapping**: Direct mapping from physical to simulation joints

### Banana Interaction
The simulation includes a yellow banana object for testing:

- **Realistic physics**: Proper collision detection and response
- **Pickable object**: Sized appropriately for the gripper
- **No threading conflicts**: Smooth interaction without errors

## Robot Specifications

- **6 degrees of freedom** (6-DOF)
- **Joint types**: All revolute joints
- **Workspace**: Approximately 0.8m reach radius
- **End effector**: Red sphere indicating tool center point (TCP)

### Joint Configuration:
1. `shoulder_pan_joint` - Base rotation (Z-axis)
2. `shoulder_lift_joint` - Shoulder elevation (Y-axis)  
3. `elbow_joint` - Elbow flexion (Y-axis)
4. `wrist_1_joint` - Wrist rotation (Z-axis)
5. `wrist_2_joint` - Wrist elevation (Y-axis)
6. `wrist_3_joint` - Tool rotation (Z-axis)

## Robot Mirroring

The system automatically mirrors the physical robot:

- **Automatic start**: Mirroring begins when simulation starts
- **Calibration**: Hardcoded joint offsets for accurate mapping
- **Real-time sync**: 5Hz update rate for smooth mirroring
- **Error recovery**: Continues running even with connection issues

### Joint Calibration:
```python
# Hardcoded calibration offsets (radians)
offsets = [0.0, -90°, 90°, 0.0, -180°, 0.0]
directions = [1, 1, 1, 1, 1, 1]  # All normal direction
mapping = [0, 1, 2, 3, 4, 5]     # Direct joint mapping
```

## GUI Controls

The MuJoCo viewer supports:
- **Mouse rotation**: Click and drag to rotate view
- **Mouse zoom**: Scroll wheel to zoom in/out
- **Mouse pan**: Right-click and drag to pan
- **Reset view**: Press 'R' to reset camera
- **Contact visualization**: Shows collision contacts
- **Force visualization**: Shows contact forces

## Interactive Commands

Available commands in the simulation:

- **`mirror`** - Start real-time mirroring from physical robot
- **`stop` or `s`** - Stop mirroring
- **`current`** - Show current pose and joint angles
- **`workspace`** - Show workspace limits
- **`commands`** - Show available commands
- **`q`** - Quit simulation

## Coordinate System

- **X-axis**: Forward (red)
- **Y-axis**: Left (green)  
- **Z-axis**: Up (blue)
- **Units**: Meters
- **Origin**: Base of the robot

## Workspace Limits

Safe workspace for the SO-100 arm:

```
X (left/right): -0.338 to +0.338 meters (±34cm)
Y (forward):    -0.383 to -0.045 meters (4.5-38cm forward)
Z (up/down):     0.096 to  0.315 meters (10-32cm high)
```

**Note**: Y must be NEGATIVE (negative = forward from base)

## Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError: No module named 'mujoco'"**
   ```bash
   pip install mujoco numpy
   ```

2. **Robot mirroring not working**
   - Check robot IP address in main.py
   - Ensure robot is accessible on network
   - Verify robot API is running

3. **Viewer not opening**
   - Ensure you have OpenGL support
   - Try updating graphics drivers
   - Check if display is available (especially on remote systems)

4. **Threading conflicts**
   - System automatically handles MuJoCo threading issues
   - Mirroring continues even with occasional conflicts
   - Check console for threading error messages

5. **Connection errors**
   - Verify robot IP address (default: 192.168.178.190)
   - Check network connectivity
   - Ensure robot API is accessible

6. **Web API issues**
   - Check phosphobot server is running
   - Verify simulation endpoints are registered
   - Check process permissions for mjpython

## Configuration

### Robot Connection
Update robot IP in `main.py`:
```python
self.phosphobot_ip = "192.168.178.190"
self.phosphobot_port = 80
```

### Mirroring Frequency
Adjust update rate in `main.py`:
```python
self.mirror_frequency = 0.2  # 5 Hz update rate
```

### Web API Settings
The simulation endpoints are configured in `phosphobot/endpoints/simulation.py`:
- Process timeout: 5 seconds for graceful shutdown
- Working directory: mujoco folder
- Command: `mjpython main.py`

## File Structure

```
mujoco/
├── so100_simulation.py    # Main simulation class with SO100Simulation
├── main.py               # Integrated control with robot mirroring
├── so100_arm.xml         # MuJoCo model definition
├── rgbd_feed.py          # Record3D RGB-D camera helper
├── README.md             # Complete documentation
└── assets/               # 3D model files and textures
```