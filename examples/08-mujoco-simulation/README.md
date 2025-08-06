# PhosphoBot: MuJoCo Simulation Example

This example demonstrates how to use MuJoCo simulation to visualize and control a robot arm in a virtual environment. The simulation provides real-time 3D visualization and can mirror movements from a physical robot.

## Prerequisites

- Python 3.6+
- A robot running the PhosphoBot server (optional, for mirroring)
- MuJoCo simulation environment
- Required Python packages (install via `pip install -r requirements.txt`)

## Configuration

The script `mujoco_simulation.py` contains several configurable parameters:

```python
# Robot connection settings
PHOSPHOBOT_IP: str = "192.168.178.190"  # IP address of the robot
PHOSPHOBOT_PORT: int = 80               # Port of the robot's API server
MIRROR_FREQUENCY: float = 0.2           # Mirroring update frequency (Hz)

# Joint calibration for SO-100 robot
JOINT_CALIBRATION = {
    "offsets": [0.0, -90.0, 90.0, 0.0, -180.0, 0.0],  # Degrees
    "directions": [1, 1, 1, 1, 1, 1],
    "mapping": [0, 1, 2, 3, 4, 5]
}
```

Modify these values according to your setup.

## How to Run

1. Make sure your robot is powered on and the PhosphoBot server is running (optional)
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Update the `PHOSPHOBOT_IP` and `PHOSPHOBOT_PORT` variables in the script if needed
4. Run the script:
   ```
   python mujoco_simulation.py
   ```

## What the Script Does

1. Initializes a MuJoCo simulation environment with a 6-DOF robot arm
2. Opens a 3D viewer where you can interact with the simulation (rotate, zoom, etc.)
3. Automatically starts mirroring the physical robot's movements (if connected)
4. Provides interactive commands to control the simulation
5. Displays real-time joint angles and end-effector positions

## Features

- **3D Visualization**: Real-time 3D rendering of the robot arm
- **Robot Mirroring**: Automatically mirrors movements from a physical robot
- **Interactive Controls**: Command-line interface for simulation control
- **Workspace Limits**: Shows the robot's reachable workspace
- **Joint Calibration**: Handles differences between simulation and physical robot

## Available Commands

- `mirror` - Start real-time mirroring from physical robot
- `stop` or `s` - Stop mirroring
- `current` - Show current pose and joint angles
- `workspace` - Show workspace limits
- `commands` - Show this help
- `q` - Quit the simulation

## Customization

You can modify the script to change:

- The robot model by editing the XML file
- The mirroring frequency for different update rates
- The joint calibration for different robot models
- The workspace limits and visualization settings

## Troubleshooting

- **MuJoCo viewer not opening**: Make sure MuJoCo is properly installed
- **Robot not connecting**: Check IP address and network connectivity
- **Joint angles incorrect**: Adjust the calibration offsets for your robot model 