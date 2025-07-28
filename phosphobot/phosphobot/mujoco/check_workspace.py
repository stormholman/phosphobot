#!/usr/bin/env python3
"""
Check SO-100 robot workspace without viewer
"""
import mujoco as mj
import numpy as np

def check_workspace():
    """Check the robot's workspace by testing different joint configurations."""
    # Load model without viewer
    model = mj.MjModel.from_xml_path("so100_arm.xml")
    data = mj.MjData(model)
    
    # Get end effector body ID
    end_effector_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Fixed_Jaw')
    
    print("SO-100 Robot Workspace Analysis")
    print("=" * 40)
    
    # Test different configurations
    configurations = [
        ("Home/Zero", [0, 0, 0, 0, 0, 0]),
        ("Reach Forward", [0, -0.5, -0.5, 0, 0, 0]),
        ("Reach Up", [0, -1.0, 0.5, 0, 0, 0]),
        ("Reach Right", [1.57, 0, 0, 0, 0, 0]),
        ("Reach Left", [-1.57, 0, 0, 0, 0, 0]),
    ]
    
    # Get actuator names to understand joint mapping
    actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print(f"Actuators: {actuator_names}")
    print()
    
    reachable_positions = []
    
    for config_name, joint_angles in configurations:
        # Set joint positions
        data.qpos[:len(joint_angles)] = joint_angles
        
        # Forward kinematics
        mj.mj_forward(model, data)
        
        # Get end effector position
        if end_effector_id != -1:
            pos = data.xpos[end_effector_id]
        else:
            pos = data.xpos[-1]  # Use last body as fallback
        
        reachable_positions.append(pos.copy())
        print(f"{config_name:15}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
    
    print("\nWorkspace Summary:")
    positions = np.array(reachable_positions)
    print(f"X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    print("\nSuggested target positions for IK testing:")
    # Generate some reasonable targets within the workspace
    center = positions.mean(axis=0)
    for i, pos in enumerate([center * 0.8, center * 1.1, center + [0.1, 0.1, 0]], 1):
        print(f"  Target {i}: {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")

if __name__ == "__main__":
    check_workspace() 