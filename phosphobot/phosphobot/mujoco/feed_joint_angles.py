#!/usr/bin/env python3
"""
A simple, direct script for controlling the SO-100 robot arm by setting joint angles.
"""
import time
import numpy as np
from so100_simulation import SO100Simulation

def main():
    """Main function to run the joint control demo."""
    print("=" * 60)
    print("SO-100 MUJOCO SIMULATION - DIRECT JOINT CONTROL")
    print("=" * 60)

    # Create simulation
    print("Initializing SO-100 simulation...")
    sim = SO100Simulation()

    print("\nSimulation started! The MuJoCo viewer should be open.")
    print("You can interact with the viewer (rotate, zoom, etc.)")

    # Give the simulation time to initialize
    time.sleep(2)

    num_joints = len(sim.joint_names)
    print(f"\nThis robot has {num_joints} joints:")
    for i, name in enumerate(sim.joint_names):
        print(f"  {i}: {name}")

    print("\nEnter target joint angles in degrees, separated by commas.")
    print(f"Example: 45, -30, 0, 90, 0, 0\n")

    try:
        while sim.viewer.is_running():
            print("-" * 50)
            user_input = input(f"Enter {num_joints} joint angles (or 'q' to quit): ").strip()

            if user_input.lower() == 'q':
                break

            try:
                # Parse the input string into a list of floats
                angle_degrees = [float(x.strip()) for x in user_input.split(',')]

                if len(angle_degrees) != num_joints:
                    print(f"Error: Please provide exactly {num_joints} angles.")
                    continue

                # Convert angles from degrees to radians for MuJoCo
                angle_radians = np.deg2rad(angle_degrees)

                # Set the robot's joint angles
                print(f"Setting joint angles to (degrees): {angle_degrees}")
                sim.set_joint_angles(angle_radians)

            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    except KeyboardInterrupt:
        print("\nExiting demo.")
    finally:
        print("Closing simulation...")
        sim.close()

if __name__ == "__main__":
    main() 