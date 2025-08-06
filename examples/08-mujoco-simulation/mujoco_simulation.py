#!/usr/bin/env python3
"""
MuJoCo Simulation Example for PhosphoBot
Demonstrates real-time robot simulation with optional physical robot mirroring
"""

import time
import numpy as np
import requests
import math
import threading
import queue
import mujoco as mj
import mujoco.viewer
from typing import List, Optional, Tuple

# Configuration
PHOSPHOBOT_IP: str = "192.168.178.190"
PHOSPHOBOT_PORT: int = 80
MIRROR_FREQUENCY: float = 0.2

# Joint calibration for SO-100 robot
JOINT_CALIBRATION = {
    "offsets": [0.0, math.radians(-90.0), math.radians(90.0), 0.0, math.radians(-180.0), 0.0],
    "directions": [1, 1, 1, 1, 1, 1],
    "mapping": [0, 1, 2, 3, 4, 5]
}

class SO100Simulation:
    """SO-100 MuJoCo simulation with GUI and robot mirroring"""
    
    def __init__(self):
        """Initialize the SO-100 MuJoCo simulation with GUI."""
        print("Initializing SO-100 simulation...")
        
        # Load the SO-100 model
        self.model = self._load_so100_model()
        self.data = mj.MjData(self.model)

        # Threading control with proper synchronization
        self.data_lock = threading.RLock()
        self.viewer_lock = threading.RLock()
        
        # Simulation control
        self.running = True
        self.paused = False
        
        # Command queue for thread-safe communication
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize viewer with better error handling
        self._init_viewer()

        # Get actuator names and joint information
        actuator_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

        # Get important body and joint indices
        self.end_effector_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'Fixed_Jaw')
        if self.end_effector_id == -1:
            possible_names = ['hand', 'gripper', 'tool', 'tcp', 'wrist_3_link']
            for name in possible_names:
                self.end_effector_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
                if self.end_effector_id != -1:
                    break
        
        # Get joint information
        self.joint_names = []
        self.joint_ids = []
        for i in range(self.model.njnt):
            joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
            if joint_name in actuator_names:
                self.joint_names.append(joint_name)
                self.joint_ids.append(i)

        self.dof_ids = []
        for joint_id in self.joint_ids:
            self.dof_ids.append(self.model.jnt_dofadr[joint_id])
        
        print(f"Found {len(self.joint_names)} joints: {self.joint_names}")
        print(f"End effector body ID: {self.end_effector_id}")
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()
        
        print("✓ SO-100 simulation initialized successfully!")

    def _init_viewer(self):
        """Initialize viewer with better error handling."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self._viewer_context = mj.viewer.launch_passive(self.model, self.data)
                self.viewer = self._viewer_context.__enter__()
                print(f"✓ Viewer initialized successfully (attempt {attempt + 1})")
                return
            except Exception as e:
                print(f"Viewer initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.5)

    def _load_so100_model(self):
        """Load the SO-100 model from the XML file."""
        try:
            return mj.MjModel.from_xml_path("so100_arm.xml")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread."""
        while self.running:
            if not self.paused:
                with self.data_lock:
                    mj.mj_step(self.model, self.data)
                
                with self.viewer_lock:
                    if hasattr(self, 'viewer') and self.viewer:
                        self.viewer.sync()
            
            time.sleep(0.01)  # 100 Hz simulation rate

    def get_end_effector_position(self, data_override=None) -> np.ndarray:
        """Get the current end effector position."""
        data_to_use = data_override if data_override is not None else self.data
        
        if self.end_effector_id != -1:
            pos = data_to_use.xpos[self.end_effector_id]
            return pos.copy()
        else:
            # Fallback: use the last joint position
            return data_to_use.xpos[-1].copy()

    def get_current_joint_angles(self) -> np.ndarray:
        """Get the current joint angles."""
        with self.data_lock:
            angles = []
            for dof_id in self.dof_ids:
                if dof_id < len(self.data.qpos):
                    angles.append(self.data.qpos[dof_id])
            return np.array(angles)

    def set_joint_angles(self, joint_angles: List[float]):
        """Set the joint angles in the simulation."""
        with self.data_lock:
            for i, dof_id in enumerate(self.dof_ids):
                if i < len(joint_angles) and dof_id < len(self.data.qpos):
                    self.data.qpos[dof_id] = joint_angles[i]
            
            # Reset velocities
            self.data.qvel[:] = 0.0

    def close(self):
        """Close the simulation and cleanup."""
        print("Closing SO-100 simulation...")
        self.running = False
        
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=2)
        
        if hasattr(self, '_viewer_context'):
            self._viewer_context.__exit__(None, None, None)
        
        print("✓ SO-100 simulation closed")

class MuJoCoSimulation:
    def __init__(self):
        print("=" * 60)
        print("MUJOCO SIMULATION EXAMPLE")
        print("Real-time Robot Visualization & Mirroring")
        print("=" * 60)
        
        # Initialize SO-100 simulation
        self.sim = SO100Simulation()
        
        # Robot connection settings
        self.joints_read_url = f"http://{PHOSPHOBOT_IP}:{PHOSPHOBOT_PORT}/joints/read"
        self.mirroring_active = False
        self.mirror_thread = None
        
        print("Using joint calibration:")
        print("  Joint 1 (Pitch): -90° offset")
        print("  Joint 2 (Elbow): +90° offset") 
        print("  Joint 4 (Wrist_Roll): -180° offset")
        
        print("System ready!")
        print("You can interact with the MuJoCo viewer (rotate, zoom, etc.)")
        
        time.sleep(2)
        
        num_joints = len(self.sim.joint_names)
        print(f"Robot has {num_joints} joints:")
        for i, name in enumerate(self.sim.joint_names):
            print(f"  {i}: {name}")

    def apply_calibration(self, raw_angles):
        """Apply joint calibration to convert physical robot angles to simulation angles"""
        if len(raw_angles) != 6:
            return raw_angles
            
        calibrated_angles = [0.0] * 6
        
        for sim_joint_idx in range(6):
            phys_joint_idx = JOINT_CALIBRATION["mapping"][sim_joint_idx]
            if phys_joint_idx < len(raw_angles):
                angle = raw_angles[phys_joint_idx]
                angle *= JOINT_CALIBRATION["directions"][sim_joint_idx]
                angle += JOINT_CALIBRATION["offsets"][sim_joint_idx]
                calibrated_angles[sim_joint_idx] = angle
        
        return calibrated_angles

    def read_physical_robot_angles(self):
        """Read joint angles from the physical robot"""
        try:
            response = requests.post(self.joints_read_url, timeout=2)
            response.raise_for_status()
            data = response.json()
            
            if "angles" in data and data["angles"]:
                return data["angles"]
            return None
        except Exception as e:
            if hasattr(self, '_last_error_time'):
                if time.time() - self._last_error_time > 5:
                    print(f"Error reading robot angles: {e}")
                    self._last_error_time = time.time()
            else:
                self._last_error_time = time.time()
            return None

    def mirror_robot_worker(self):
        """Worker thread that continuously mirrors physical robot movements"""
        print(f"Starting robot mirroring at {1/MIRROR_FREQUENCY:.1f} Hz")
        print("Physical robot movements will be mirrored in simulation")
        print("Press 's' + Enter to stop mirroring")
        
        consecutive_errors = 0
        max_consecutive_errors = 20
        
        time.sleep(2.0)
        
        while self.mirroring_active:
            try:
                raw_angles = self.read_physical_robot_angles()
                
                if raw_angles is not None and len(raw_angles) == 6:
                    calibrated_angles = self.apply_calibration(raw_angles)
                    
                    # Set joint angles in simulation
                    self.sim.set_joint_angles(calibrated_angles)
                    
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors, stopping mirroring")
                        break
                        
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Mirroring error: {e}")
                    break
            
            time.sleep(1/MIRROR_FREQUENCY)

    def start_mirroring(self):
        """Start the robot mirroring thread"""
        if self.mirroring_active:
            print("Mirroring is already active")
            return
            
        self.mirroring_active = True
        self.mirror_thread = threading.Thread(target=self.mirror_robot_worker)
        self.mirror_thread.daemon = True
        self.mirror_thread.start()
        print("Robot mirroring started")

    def stop_mirroring(self):
        """Stop the robot mirroring thread"""
        if not self.mirroring_active:
            print("Mirroring is not active")
            return
            
        self.mirroring_active = False
        if self.mirror_thread:
            self.mirror_thread.join(timeout=2)
        print("Mirroring stopped")

    def show_workspace_info(self):
        """Display workspace limits"""
        print("\n" + "="*50)
        print("WORKSPACE LIMITS:")
        print("  X (left/right): -0.338 to +0.338 meters (±34cm)")
        print("  Y (forward):    -0.383 to -0.045 meters (4.5-38cm forward)")
        print("  Z (up/down):     0.096 to  0.315 meters (10-32cm high)")
        print("\nREMEMBER: Y must be NEGATIVE (negative = forward from base)")
        print("="*50)
    
    def show_commands(self):
        """Display available commands"""
        print("\n" + "="*50)
        print("AVAILABLE COMMANDS:")
        print("\nROBOT MIRRORING:")
        print("  'mirror' - Start real-time mirroring from physical robot")
        print("  'stop' or 's' - Stop mirroring")
        print("\nINFORMATION:")
        print("  'current' - Show current pose")
        print("  'workspace' - Show workspace limits")
        print("  'commands' - Show this help")
        print("\nEXIT:")
        print("  'q' - Quit")
        print("="*50)
    
    def get_current_pose(self):
        """Display current robot pose and joint angles"""
        try:
            # Get end effector position
            end_effector_pos = self.sim.get_end_effector_position()
            
            # Get joint angles
            joint_angles = self.sim.get_current_joint_angles()
            
            print(f"\nCURRENT STATE:")
            print(f"  End effector position: [{end_effector_pos[0]:.3f}, {end_effector_pos[1]:.3f}, {end_effector_pos[2]:.3f}] m")
            print(f"  Joint angles (degrees): {', '.join([f'{np.degrees(a):.1f}' for a in joint_angles])}")
            
            if self.mirroring_active:
                print(f"  Robot mirroring: ACTIVE")
            else:
                print(f"  Robot mirroring: INACTIVE")
                
        except Exception as e:
            print(f"  Could not get current state: {e}")
    
    def run(self):
        """Main simulation loop"""
        self.show_workspace_info()
        self.get_current_pose()
        self.show_commands()

        print("\nAuto-starting robot mirroring...")
        self.start_mirroring()

        try:
            while self.sim.viewer.is_running():
                # Check for user input (non-blocking)
                try:
                    user_input = input("\nEnter command (or press Enter to continue): ").strip()
                    
                    if user_input.lower() == 'q':
                        break
                    elif user_input.lower() == 'current':
                        self.get_current_pose()
                    elif user_input.lower() == 'workspace':
                        self.show_workspace_info()
                    elif user_input.lower() == 'commands':
                        self.show_commands()
                    elif user_input.lower() == 'mirror':
                        self.start_mirroring()
                    elif user_input.lower() in ['stop', 's']:
                        self.stop_mirroring()
                        
                except EOFError:
                    # No input available, continue simulation
                    pass
                
                time.sleep(0.1)  # Small delay
                
        except KeyboardInterrupt:
            print("\n\nShutting down system...")
        finally:
            if self.mirroring_active:
                self.stop_mirroring()
            print("Closing simulation...")
            self.sim.close()
            print("System shutdown complete.")

def main():
    """Main entry point"""
    try:
        simulation = MuJoCoSimulation()
        simulation.run()
        
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure so100_arm.xml exists in the current directory")
        print("2. Check that MuJoCo is properly installed")
        print("3. Verify all dependencies are installed")
        print("4. Ensure the robot IP address is correct")

if __name__ == "__main__":
    main() 