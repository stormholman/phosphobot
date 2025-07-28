#!/usr/bin/env python3
"""
Inverse Kinematics Solver for SO-100 Robot Arm
Computes joint angles from target end effector pose (x,y,z,rx,ry,rz)
"""
import mujoco as mj
import numpy as np
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as R

class SO100IKSolver:
    def __init__(self, model_path: str = "so100_arm.xml"):
        """Initialize the IK solver with the SO-100 model."""
        # Load the SO-100 model
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # Get actuator names
        actuator_names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

        # Get important body and joint indices
        self.end_effector_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'Fixed_Jaw')
        if self.end_effector_id == -1:
            # Try alternative names for end effector
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
        
        print(f"IK Solver initialized with {len(self.joint_names)} joints: {self.joint_names}")
        print(f"End effector body ID: {self.end_effector_id}")
        
        # IK solver parameters
        self.max_iterations = 1500
        self.position_tolerance = 1e-3
        self.orientation_tolerance = 5e-2
        self.initial_step_size = 0.3
        self.min_step_size = 1e-3
        self.step_reduction_factor = 0.8
        self.orientation_weight = 0.05  # Reduced weight for orientation
        
        # Reset to neutral pose for consistent starting point
        self.reset_to_neutral()
    
    def reset_to_neutral(self):
        """Reset robot to a neutral pose."""
        # Set all joints to zero (or neutral positions)
        self.data.qpos[:] = 0
        # Move base rotation to a more reasonable starting position
        if len(self.joint_ids) > 0:
            # Slight upward angle for better reachability
            self.data.qpos[self.model.jnt_qposadr[self.joint_ids[1]]] = -0.5  # Pitch up
        mj.mj_forward(self.model, self.data)
    
    def get_end_effector_pose(self, data_override=None) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end effector position and orientation."""
        d = data_override if data_override is not None else self.data
        
        if self.end_effector_id != -1:
            pos = d.xpos[self.end_effector_id].copy()
            # Get rotation matrix and convert to quaternion
            rot_mat = d.xmat[self.end_effector_id].reshape(3, 3)
            quat = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
        else:
            # Use TCP site if available
            tcp_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, 'tcp')
            if tcp_id != -1:
                pos = d.site_xpos[tcp_id].copy()
                rot_mat = d.site_xmat[tcp_id].reshape(3, 3)
                quat = R.from_matrix(rot_mat).as_quat()
            else:
                # Fallback: use last body position
                pos = d.xpos[-1].copy()
                rot_mat = d.xmat[-1].reshape(3, 3)
                quat = R.from_matrix(rot_mat).as_quat()
        
        return pos, quat
    
    def get_jacobian(self, data_override=None) -> Tuple[np.ndarray, np.ndarray]:
        """Get Jacobian matrices for the end effector."""
        d = data_override if data_override is not None else self.data
        # Initialize Jacobian matrices
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        if self.end_effector_id != -1:
            # Get Jacobian for end effector body
            mj.mj_jac(self.model, d, jacp, jacr, d.xpos[self.end_effector_id], self.end_effector_id)
        else:
            # Use TCP site if available
            tcp_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, 'tcp')
            if tcp_id != -1:
                mj.mj_jacSite(self.model, d, jacp, jacr, tcp_id)
        
        return jacp, jacr
    
    def euler_to_quat(self, rx, ry, rz, sequence='xyz'):
        """Convert Euler angles (in radians) to quaternion [x, y, z, w]."""
        return R.from_euler(sequence, [rx, ry, rz]).as_quat()
    
    def quat_error(self, q_desired, q_current):
        """Compute quaternion error for orientation control."""
        # Ensure both quaternions are unit quaternions
        q_desired = q_desired / np.linalg.norm(q_desired)
        q_current = q_current / np.linalg.norm(q_current)
        
        # Compute relative quaternion
        q_current_conj = np.array([-q_current[0], -q_current[1], -q_current[2], q_current[3]])
        q_error = self.quat_multiply(q_desired, q_current_conj)
        
        # Return the vector part (angular velocity)
        return q_error[:3]
    
    def quat_multiply(self, q1, q2):
        """Multiply two quaternions [x, y, z, w]."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        
        return np.array([x, y, z, w])
    
    def solve_ik_position_only(self, target_pos: List[float], max_iterations: Optional[int] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for target position only (simpler, more robust).
        """
        target_pos = np.array(target_pos)
        max_iter = max_iterations or self.max_iterations

        # Create a temporary data instance for IK solving
        temp_data = mj.MjData(self.model)
        mj.mju_copy(temp_data.qpos, self.data.qpos)
        mj.mju_copy(temp_data.qvel, self.data.qvel)
        mj.mju_copy(temp_data.act, self.data.act)
        if self.model.nmocap > 0:
            mj.mju_copy(temp_data.mocap_pos, self.data.mocap_pos)
            mj.mju_copy(temp_data.mocap_quat, self.data.mocap_quat)
        mj.mju_copy(temp_data.userdata, self.data.userdata)
        temp_data.time = self.data.time

        step_size = self.initial_step_size
        prev_error_norm = float('inf')
        stagnation_count = 0
        success = False

        for iteration in range(max_iter):
            # Forward kinematics on the temporary data
            mj.mj_forward(self.model, temp_data)
            
            # Get current end effector position from temp_data
            current_pos, _ = self.get_end_effector_pose(data_override=temp_data)
            
            # Calculate position error
            position_error = target_pos - current_pos
            error_norm = np.linalg.norm(position_error)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration+1}: Pos error = {error_norm:.6f}, Step size = {step_size:.4f}")
            
            # Check convergence
            if error_norm < self.position_tolerance:
                print(f"IK converged in {iteration+1} iterations")
                success = True
                break
            
            # Adaptive step size
            if error_norm >= prev_error_norm:
                stagnation_count += 1
                if stagnation_count > 3:
                    step_size *= self.step_reduction_factor
                    stagnation_count = 0
                    if step_size < self.min_step_size:
                        break
            else:
                stagnation_count = 0
            
            prev_error_norm = error_norm
            
            # Get Jacobian from temp_data
            jacp, _ = self.get_jacobian(data_override=temp_data)
            
            # Use only position Jacobian for the joints we can control
            J = jacp[:, self.dof_ids]
            
            # Pseudo-inverse Jacobian method with damping
            try:
                damping = 1e-4
                J_damped = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
                delta_q = J_damped @ position_error
                
                # Apply step size
                delta_q *= step_size
                
                # Update joint positions in temp_data
                for i, joint_id in enumerate(self.joint_ids):
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    temp_data.qpos[qpos_adr] += delta_q[i]
                        
                    # Respect joint limits
                    if self.model.jnt_limited[joint_id]:
                        qmin = self.model.jnt_range[joint_id][0]
                        qmax = self.model.jnt_range[joint_id][1]
                        temp_data.qpos[qpos_adr] = np.clip(temp_data.qpos[qpos_adr], qmin, qmax)
                
            except np.linalg.LinAlgError:
                print("Singular Jacobian, stopping IK")
                success = False
                break
        
        else:
            print(f"IK did not converge after {max_iter} iterations")
            success = False
        
        # Return joint angles for the controlled joints from temp_data
        joint_angles = []
        for joint_id in self.joint_ids:
            joint_angles.append(temp_data.qpos[self.model.jnt_qposadr[joint_id]])
        
        return np.array(joint_angles), success
    
    def solve_ik(self, target_pos: List[float], target_rot: List[float], 
                 rotation_sequence: str = 'xyz', max_iterations: Optional[int] = None,
                 position_only: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for target pose (position + orientation).
        
        Args:
            target_pos: Target position [x, y, z] in meters
            target_rot: Target rotation [rx, ry, rz] in radians
            rotation_sequence: Euler angle sequence (default: 'xyz')
            max_iterations: Maximum number of iterations
            position_only: If True, only solve for position (more robust)
            
        Returns:
            Tuple of (joint_angles, success_flag)
        """
        # If position_only or all rotations are zero, use simpler position-only solver
        if position_only or np.allclose(target_rot, 0):
            return self.solve_ik_position_only(target_pos, max_iterations)
        
        target_pos = np.array(target_pos)
        target_quat = self.euler_to_quat(target_rot[0], target_rot[1], target_rot[2], rotation_sequence)
        max_iter = max_iterations or self.max_iterations

        # Create a temporary data instance for IK solving
        temp_data = mj.MjData(self.model)
        mj.mju_copy(temp_data.qpos, self.data.qpos)
        mj.mju_copy(temp_data.qvel, self.data.qvel)
        mj.mju_copy(temp_data.act, self.data.act)
        if self.model.nmocap > 0:
            mj.mju_copy(temp_data.mocap_pos, self.data.mocap_pos)
            mj.mju_copy(temp_data.mocap_quat, self.data.mocap_quat)
        mj.mju_copy(temp_data.userdata, self.data.userdata)
        temp_data.time = self.data.time
        
        step_size = self.initial_step_size
        prev_error_norm = float('inf')
        stagnation_count = 0
        success = False

        for iteration in range(max_iter):
            # Forward kinematics on the temporary data
            mj.mj_forward(self.model, temp_data)
            
            # Get current end effector pose from temp_data
            current_pos, current_quat = self.get_end_effector_pose(data_override=temp_data)
            
            # Calculate position and orientation errors
            position_error = target_pos - current_pos
            orientation_error = self.quat_error(target_quat, current_quat)
            
            position_error_norm = np.linalg.norm(position_error)
            orientation_error_norm = np.linalg.norm(orientation_error)
            total_error_norm = position_error_norm + self.orientation_weight * orientation_error_norm
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration+1}: Pos error = {position_error_norm:.6f}, Rot error = {orientation_error_norm:.6f}")
            
            # Check convergence
            if (position_error_norm < self.position_tolerance and 
                orientation_error_norm < self.orientation_tolerance):
                print(f"IK converged in {iteration+1} iterations")
                success = True
                break
            
            # Adaptive step size
            if total_error_norm >= prev_error_norm:
                stagnation_count += 1
                if stagnation_count > 3:
                    step_size *= self.step_reduction_factor
                    stagnation_count = 0
                    if step_size < self.min_step_size:
                        break
            else:
                stagnation_count = 0
            
            prev_error_norm = total_error_norm
            
            # Get Jacobian from temp_data
            jacp, jacr = self.get_jacobian(data_override=temp_data)
            
            # Use only Jacobians for the joints we can control
            J_pos = jacp[:, self.dof_ids]
            J_rot = jacr[:, self.dof_ids]
            
            # Combine position and rotation Jacobians with weighting
            J_combined = np.vstack([J_pos, self.orientation_weight * J_rot])
            error_combined = np.hstack([position_error, self.orientation_weight * orientation_error])
            
            # Pseudo-inverse Jacobian method with damping
            try:
                damping = 1e-4
                J_damped = J_combined.T @ np.linalg.inv(J_combined @ J_combined.T + damping * np.eye(J_combined.shape[0]))
                delta_q = J_damped @ error_combined
                
                # Apply step size
                delta_q *= step_size
                
                # Update joint positions in temp_data
                for i, joint_id in enumerate(self.joint_ids):
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    temp_data.qpos[qpos_adr] += delta_q[i]
                        
                    # Respect joint limits
                    if self.model.jnt_limited[joint_id]:
                        qmin = self.model.jnt_range[joint_id][0]
                        qmax = self.model.jnt_range[joint_id][1]
                        temp_data.qpos[qpos_adr] = np.clip(temp_data.qpos[qpos_adr], qmin, qmax)
                
            except np.linalg.LinAlgError:
                print("Singular Jacobian, stopping IK")
                success = False
                break
        
        else:
            print(f"IK did not converge after {max_iter} iterations")
            success = False
        
        # Return joint angles for the controlled joints from temp_data
        joint_angles = []
        for joint_id in self.joint_ids:
            joint_angles.append(temp_data.qpos[self.model.jnt_qposadr[joint_id]])
        
        return np.array(joint_angles), success

def main():
    """Main function to demonstrate the IK solver."""
    print("=" * 60)
    print("SO-100 INVERSE KINEMATICS SOLVER")
    print("=" * 60)
    
    # Create IK solver
    print("Initializing IK solver...")
    ik_solver = SO100IKSolver()
    
    print("\nIK solver ready!")
    print("Commands:")
    print("  x y z                    - Position only IK (more robust)")
    print("  x y z rx ry rz          - Full pose IK")
    print("  Example: 0.3 0.2 0.4    - Position only")
    print("  Example: 0.3 0.2 0.4 0 90 0  - Full pose")
    print("Position in meters, rotation in degrees")
    print("Type 'q' to quit\n")
    
    try:
        while True:
            print("-" * 50)
            user_input = input("Enter target pose: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            try:
                # Parse input
                values = [float(x) for x in user_input.split()]
                
                if len(values) == 3:
                    # Position only
                    target_pos = values
                    print(f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
                    joint_angles, success = ik_solver.solve_ik_position_only(target_pos)
                
                elif len(values) == 6:
                    # Full pose
                    target_pos = values[:3]
                    target_rot_deg = values[3:]
                    target_rot_rad = np.deg2rad(target_rot_deg)
                    
                    print(f"Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m")
                    print(f"Target rotation: [{target_rot_deg[0]:.1f}, {target_rot_deg[1]:.1f}, {target_rot_deg[2]:.1f}] deg")
                    
                    # Try position-only first if full pose fails
                    joint_angles, success = ik_solver.solve_ik(target_pos, target_rot_rad)
                    
                    if not success:
                        print("Full pose IK failed, trying position-only...")
                        joint_angles, success = ik_solver.solve_ik_position_only(target_pos)
                else:
                    print("Error: Please provide 3 values (x y z) or 6 values (x y z rx ry rz)")
                    continue
                
                if success:
                    print("\n✓ IK Solution found!")
                    print("Joint angles (radians):")
                    for i, (name, angle) in enumerate(zip(ik_solver.joint_names, joint_angles)):
                        print(f"  {name}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
                    
                    print("\nJoint angles (degrees) - Copy this to feed_joint_angles.py:")
                    angles_deg = [f"{np.degrees(angle):.2f}" for angle in joint_angles]
                    print(f"  {', '.join(angles_deg)}")
                else:
                    print("\n✗ IK failed to converge")
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces.")
            except Exception as e:
                print(f"An error occurred: {e}")
    
    except KeyboardInterrupt:
        print("\nExiting IK solver.")

if __name__ == "__main__":
    main() 