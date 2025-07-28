#!/usr/bin/env python3
"""
SO-100 Integrated Control System
Combines IK solver with real-time simulation control and physical robot mirroring
"""
import time
import numpy as np
import requests
import math
import threading
from so100_simulation import SO100Simulation
from so100_ik_solver import SO100IKSolver

class SO100IntegratedControl:
    def __init__(self):
        """Initialize the integrated control system."""
        print("=" * 70)
        print("SO-100 INTEGRATED CONTROL SYSTEM")
        print("IK Solver + Real-time Simulation + Robot Mirroring")
        print("=" * 70)
        
        # Initialize IK solver (without viewer)
        print("Initializing IK solver...")
        self.ik_solver = SO100IKSolver()
        
        # Initialize simulation with viewer
        print("\nInitializing simulation with MuJoCo viewer...")
        self.sim = SO100Simulation()
        
        # Robot mirroring configuration
        self.phosphobot_ip = "192.168.178.191"
        self.phosphobot_port = 80
        self.joints_read_url = f"http://{self.phosphobot_ip}:{self.phosphobot_port}/joints/read"
        self.mirror_frequency = 0.2  # 5 Hz - very conservative to avoid MuJoCo conflicts
        self.mirroring_active = False
        self.mirror_thread = None
        
        # Hardcoded joint calibration (determined through testing)
        self.joint_calibration = {
            "offsets": [0.0, math.radians(-90.0), math.radians(90.0), 0.0, math.radians(-180.0), 0.0],  # Radians
            "directions": [1, 1, 1, 1, 1, 1],            # All normal direction
            "mapping": [0, 1, 2, 3, 4, 5]                # Direct joint mapping
        }
        
        print("✓ Using hardcoded joint calibration:")
        print("  Joint 1 (Pitch): -90° offset")
        print("  Joint 2 (Elbow): +90° offset") 
        print("  Joint 4 (Wrist_Roll): -180° offset")
        
        print("\n✓ System ready!")
        print("You can interact with the MuJoCo viewer (rotate, zoom, etc.)")
        
        # Give the simulation time to initialize
        time.sleep(2)
        
        # Show joint information
        num_joints = len(self.sim.joint_names)
        print(f"\nRobot has {num_joints} joints:")
        for i, name in enumerate(self.sim.joint_names):
            print(f"  {i}: {name}")

    def apply_calibration(self, raw_angles):
        """Apply hardcoded calibration to raw joint angles from physical robot."""
        if len(raw_angles) != 6:
            return raw_angles
            
        calibrated_angles = [0.0] * 6
        
        # Apply mapping, direction, and offset
        for sim_joint_idx in range(6):
            phys_joint_idx = self.joint_calibration["mapping"][sim_joint_idx]
            if phys_joint_idx < len(raw_angles):
                angle = raw_angles[phys_joint_idx]
                angle *= self.joint_calibration["directions"][sim_joint_idx]
                angle += self.joint_calibration["offsets"][sim_joint_idx]
                calibrated_angles[sim_joint_idx] = angle
        
        return calibrated_angles

    def read_physical_robot_angles(self):
        """Read joint angles from the physical robot."""
        try:
            response = requests.post(self.joints_read_url, timeout=2)
            response.raise_for_status()
            data = response.json()
            
            if "angles" in data and data["angles"]:
                return data["angles"]  # Returns angles in radians
            return None
        except Exception as e:
            # Only print error occasionally to avoid spam
            if hasattr(self, '_last_error_time'):
                if time.time() - self._last_error_time > 5:  # Print every 5 seconds
                    print(f"Error reading robot angles: {e}")
                    self._last_error_time = time.time()
            else:
                self._last_error_time = time.time()
            return None

    def mirror_robot_worker(self):
        """Worker thread for continuous robot mirroring."""
        print(f"🔄 Starting robot mirroring at {1/self.mirror_frequency:.1f} Hz")
        print("Physical robot movements will be mirrored in simulation")
        print("Press 's' + Enter to stop mirroring\n")
        
        consecutive_errors = 0
        max_consecutive_errors = 20  # Increased tolerance
        threading_errors = 0
        
        # Add initial delay to let simulation stabilize
        time.sleep(2.0)  # Longer initial delay
        
        while self.mirroring_active:
            try:
                # Read angles from physical robot
                raw_angles = self.read_physical_robot_angles()
                
                if raw_angles is not None and len(raw_angles) == 6:
                    # Apply calibration
                    calibrated_angles = self.apply_calibration(raw_angles)
                    
                    # Apply to simulation - single attempt, skip if conflict
                    try:
                        self.sim.set_joint_angles(calibrated_angles)
                        consecutive_errors = 0
                        
                        # Optional: Print current angles (uncomment for debugging)
                        # angles_deg = [math.degrees(a) for a in calibrated_angles]
                        # print(f"Mirroring: {' '.join([f'{a:.1f}°' for a in angles_deg])}", end='\r')
                        
                    except Exception as e:
                        if "copyDataVisual" in str(e) or "stack is in use" in str(e):
                            # Threading conflict - just skip this update
                            threading_errors += 1
                            if threading_errors % 10 == 0:
                                print(f"ℹ️  Skipped {threading_errors} updates due to MuJoCo threading (normal)")
                        else:
                            # Different error - count it
                            consecutive_errors += 1
                            if consecutive_errors % 5 == 0:
                                print(f"Simulation update error ({consecutive_errors}): {e}")
                else:
                    consecutive_errors += 1
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors % 5 == 0:  # Print every 5th error
                    print(f"Mirror error ({consecutive_errors}): {e}")
            
            # Stop mirroring if too many consecutive errors (but not threading errors)
            if consecutive_errors >= max_consecutive_errors:
                print(f"\n❌ Stopping mirroring due to {consecutive_errors} consecutive errors")
                self.mirroring_active = False
                break
                
            time.sleep(self.mirror_frequency)
        
        if threading_errors > 0:
            print(f"ℹ️  Total threading conflicts handled: {threading_errors}")
        print("\n🛑 Robot mirroring stopped")

    def start_mirroring(self):
        """Start real-time robot mirroring."""
        if self.mirroring_active:
            print("⚠️  Mirroring is already active")
            return
            
        # Test connection first
        print("Testing connection to physical robot...")
        test_angles = self.read_physical_robot_angles()
        if test_angles is None:
            print("❌ Cannot connect to physical robot")
            print(f"Make sure robot is accessible at {self.joints_read_url}")
            return
            
        print("✅ Connection successful!")
        
        self.mirroring_active = True
        self.mirror_thread = threading.Thread(target=self.mirror_robot_worker, daemon=True)
        self.mirror_thread.start()

    def stop_mirroring(self):
        """Stop real-time robot mirroring."""
        if not self.mirroring_active:
            print("⚠️  Mirroring is not active")
            return
            
        self.mirroring_active = False
        if self.mirror_thread:
            self.mirror_thread.join(timeout=2)
        print("🛑 Mirroring stopped")

    def show_workspace_info(self):
        """Display workspace information for user reference."""
        print("\n" + "="*50)
        print("WORKSPACE LIMITS:")
        print("  X (left/right): -0.338 to +0.338 meters (±34cm)")
        print("  Y (forward):    -0.383 to -0.045 meters (4.5-38cm forward)")
        print("  Z (up/down):     0.096 to  0.315 meters (10-32cm high)")
        print("\nREMEMBER: Y must be NEGATIVE (negative = forward from base)")
        print("="*50)
    
    def show_examples(self):
        """Show example inputs."""
        print("\nEXAMPLE INPUTS:")
        print("  Position only (3 values):")
        print("    0.0 -0.3 0.2     ← Forward center")
        print("    0.15 -0.2 0.15   ← Right side")
        print("    -0.1 -0.25 0.18  ← Left forward")
        print()
        print("  Full pose (6 values):")
        print("    0.0 -0.25 0.18 0 45 0    ← Forward with pitch")
        print("    0.1 -0.2 0.15 0 0 90     ← Right with wrist rotation")
        print()

    def show_commands(self):
        """Show all available commands."""
        print("\n" + "="*50)
        print("AVAILABLE COMMANDS:")
        print("\n🎯 POSE CONTROL:")
        print("  Enter target pose (3 or 6 values)")
        print("  Example: 0.0 -0.3 0.2")
        print("\n🔄 ROBOT MIRRORING:")
        print("  'mirror' - Start real-time mirroring from physical robot")
        print("  'stop' or 's' - Stop mirroring")
        print("\n📋 INFORMATION:")
        print("  'current' - Show current pose")
        print("  'examples' - Show example inputs")
        print("  'workspace' - Show workspace limits")
        print("  'commands' - Show this help")
        print("\n🚪 EXIT:")
        print("  'q' - Quit")
        print("="*50)
    
    def get_current_pose(self):
        """Get and display current end effector pose."""
        try:
            pos = self.sim.get_end_effector_position()
            angles = self.sim.get_current_joint_angles()
            
            print(f"\nCURRENT STATE:")
            print(f"  End effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
            print(f"  Joint angles (degrees): {', '.join([f'{np.degrees(a):.1f}' for a in angles])}")
            
            # Show mirroring status
            if self.mirroring_active:
                print(f"  🔄 Robot mirroring: ACTIVE")
            else:
                print(f"  🔄 Robot mirroring: INACTIVE")
                
        except Exception as e:
            print(f"  Could not get current state: {e}")
    
    def move_to_pose(self, target_input: str) -> bool:
        """
        Process pose input and move robot.
        
        Args:
            target_input: Space-separated pose values
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse input
            values = [float(x) for x in target_input.split()]
            
            if len(values) == 3:
                # Position only
                target_pos = values
                print(f"\n🎯 Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
                print("Solving position-only IK...")
                
                joint_angles, success = self.ik_solver.solve_ik_position_only(target_pos)
                
            elif len(values) == 6:
                # Full pose
                target_pos = values[:3]
                target_rot_deg = values[3:]
                target_rot_rad = np.deg2rad(target_rot_deg)
                
                print(f"\n🎯 Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
                print(f"🎯 Target rotation: [{target_rot_deg[0]:.1f}, {target_rot_deg[1]:.1f}, {target_rot_deg[2]:.1f}] deg")
                print("Solving full pose IK...")
                
                # Try full pose first
                joint_angles, success = self.ik_solver.solve_ik(target_pos, target_rot_rad)
                
                if not success:
                    print("⚠️  Full pose IK failed, trying position-only...")
                    joint_angles, success = self.ik_solver.solve_ik_position_only(target_pos)
                    
            else:
                print("❌ Error: Please provide 3 values (x y z) or 6 values (x y z rx ry rz)")
                return False
            
            if success:
                print("✅ IK Solution found!")
                
                # Display joint angles
                print("Joint angles computed:")
                for i, (name, angle) in enumerate(zip(self.ik_solver.joint_names, joint_angles)):
                    print(f"  {name}: {np.degrees(angle):.2f}°")
                
                # Apply to simulation
                print("\n🤖 Moving robot...")
                self.sim.set_joint_angles(joint_angles)
                
                # Wait a moment for movement
                time.sleep(1)
                
                # Verify final position
                final_pos = self.sim.get_end_effector_position()
                if len(values) >= 3:
                    error = np.linalg.norm(np.array(values[:3]) - final_pos)
                    print(f"✓ Movement complete!")
                    print(f"  Final position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
                    print(f"  Position error: {error:.4f} m")
                
                return True
                
            else:
                print("❌ IK failed to converge - target may be outside workspace")
                return False
                
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by spaces.")
            return False
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            return False
    
    def run(self):
        """Main control loop."""
        self.show_workspace_info()
        self.show_examples()
        self.get_current_pose()
        self.show_commands()
        
        try:
            while self.sim.viewer.is_running():
                print("\n" + "-"*40)
                user_input = input("Enter command or target pose: ").strip()
                
                if user_input.lower() == 'q':
                    break
                elif user_input.lower() == 'current':
                    self.get_current_pose()
                elif user_input.lower() == 'examples':
                    self.show_examples()
                elif user_input.lower() == 'workspace':
                    self.show_workspace_info()
                elif user_input.lower() == 'commands':
                    self.show_commands()
                elif user_input.lower() == 'mirror':
                    self.start_mirroring()
                elif user_input.lower() in ['stop', 's']:
                    self.stop_mirroring()
                elif user_input:
                    # Stop mirroring if active before manual movement
                    if self.mirroring_active:
                        print("⚠️  Stopping mirroring for manual control...")
                        self.stop_mirroring()
                        time.sleep(0.5)
                    self.move_to_pose(user_input)
                
        except KeyboardInterrupt:
            print("\n\nShutting down system...")
        finally:
            if self.mirroring_active:
                self.stop_mirroring()
            print("Closing simulation...")
            self.sim.close()
            print("✓ System shutdown complete.")

def main():
    """Main function."""
    try:
        # Create and run integrated control system
        control_system = SO100IntegratedControl()
        control_system.run()
        
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure so100_arm.xml exists")
        print("2. Check that MuJoCo viewer can launch")
        print("3. Verify all dependencies are installed")

if __name__ == "__main__":
    main() 