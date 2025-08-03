# SO-100 Robotic Arm Simulation

A robust MuJoCo simulation of the SO-100 robotic arm with interactive GUI, inverse kinematics, and real robot integration capabilities.

## Features

- **Real-time 3D visualization** with MuJoCo viewer
- **Robust threading** - no more viewer conflicts during collisions
- **Interactive banana object** for manipulation testing
- **6-DOF robotic arm** simulation with gripper
- **Real robot mirroring** support
- **Thread-safe operations** with enhanced error handling

## Requirements

```bash
pip install mujoco numpy requests
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the integrated simulation with the MuJoCo viewer:

```bash
mjpython main.py
```

This starts the SO-100 arm simulation, opens the viewer, and enables inverse kinematics together with optional mirroring of the physical robot.

### 4. Use in Your Code

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
so100_sim/
├── so100_simulation.py      # Main simulation class
├── main.py                  # Simulation bootstrap + robot mirroring
├── so100_arm.xml            # MuJoCo model definition
├── rgbd_feed.py             # Record3D RGB-D camera helper (used by AI-kin)
└── assets/                  # 3D model files, textures, etc.
```

## Key Components

### SO100Simulation Class
The main simulation class with robust error handling:

- **Thread-safe operations**: Eliminates viewer conflicts during collisions
- **Enhanced error handling**: Gracefully handles known threading issues
- **Command queue system**: Safe communication between threads
- **Automatic recovery**: Continues running even during OpenGL conflicts

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

## Inverse Kinematics

The IK solver uses:
- **Jacobian pseudo-inverse method**
- **Iterative solution** with configurable parameters
- **Joint limit enforcement**
- **Convergence tolerance**: 1e-4 meters
- **Maximum iterations**: 100

### IK Parameters (configurable):
```python
sim.max_iterations = 100    # Maximum IK iterations
sim.tolerance = 1e-4        # Position error tolerance
sim.step_size = 0.1         # IK step size
```

## GUI Controls

The MuJoCo viewer supports:
- **Mouse rotation**: Click and drag to rotate view
- **Mouse zoom**: Scroll wheel to zoom in/out
- **Mouse pan**: Right-click and drag to pan
- **Reset view**: Press 'R' to reset camera
- **Contact visualization**: Shows collision contacts
- **Force visualization**: Shows contact forces

## Coordinate System

- **X-axis**: Forward (red)
- **Y-axis**: Left (green)  
- **Z-axis**: Up (blue)
- **Units**: Meters
- **Origin**: Base of the robot

## Example Target Positions

Safe positions to try:

```python
# Reachable positions
good_targets = [
    [0.3, 0.2, 0.4],    # Forward-right-up
    [0.0, 0.4, 0.3],    # Side reach
    [0.2, 0.0, 0.2],    # Forward-low
    [-0.1, 0.3, 0.5],   # Back-side-high
    [0.4, 0.1, 0.1],    # Forward-low reach
]

# Test each position
for target in good_targets:
    sim.move_to_position(target)
    time.sleep(2)
```

## Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError: No module named 'mujoco'"**
   ```bash
   pip install mujoco numpy
   ```

2. **IK not converging**
   - Try positions closer to current pose
   - Check if target is within robot workspace
   - Increase max_iterations or adjust step_size

3. **Viewer not opening**
   - Ensure you have OpenGL support
   - Try updating graphics drivers
   - Check if display is available (especially on remote systems)

4. **Robot moving erratically**
   - Reduce IK step_size for smoother motion
   - Ensure target positions are reachable
   - Check joint limits aren't being violated

## File Structure

```
so100_sim/
├── so100_simulation.py    # Main simulation class with SO100Simulation
├── demo.py               # Interactive and automated demo scripts
├── quick_test.py         # Quick verification test
├── requirements.txt      # Dependencies (mujoco, numpy)
└── README.md            # Complete documentation
```