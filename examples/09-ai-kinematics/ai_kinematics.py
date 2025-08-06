#!/usr/bin/env python3
"""
AI Kinematics Example for PhosphoBot
Demonstrates AI-powered vision control of a robot arm using RGB-D cameras
"""

import numpy as np
import cv2
import requests
import os
import time
import threading
import base64
import json
import tempfile
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import anthropic

# Configuration
ROBOT_ENDPOINT: str = "http://192.168.178.190/move/absolute?robot_id=0"
POSITION_TOLERANCE: float = 0.03  # meters
ORIENTATION_TOLERANCE: float = 0.2  # radians

# Camera settings
ARUCO_MARKER_ID: int = 0
ARUCO_MARKER_SIZE: float = 0.04  # meters

# AI settings
ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')

class VisionAnalyzer:
    """Vision analyzer that uses Claude's vision API to identify objects and return pixel coordinates"""
    
    def __init__(self):
        self.client = None
        self.setup_api_key()
        
    def setup_api_key(self):
        """Set up Anthropic API key"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not api_key:
            print("[Vision] Warning: No API key found. AI vision features will not work.")
            print("[Vision] Set ANTHROPIC_API_KEY environment variable to enable AI features.")
            return
            
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            print("[Vision] Claude client initialized successfully")
        except Exception as e:
            print(f"[Vision] Error initializing Claude client: {e}")
            self.client = None
    
    def opencv_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image (RGB)"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def image_to_base64(self, cv_image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        pil_image = self.opencv_to_pil(cv_image)
        
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def analyze_image(self, rgb_image: np.ndarray, target_description: str, depth_frame: np.ndarray = None) -> Optional[Dict[str, Any]]:
        """Analyze image using Claude vision API"""
        if not self.client:
            return None
            
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(rgb_image)
            
            # Create prompt for object detection
            prompt = f"""
            You are a robotic vision system. Look at this image and find the {target_description}.
            
            Please respond with a JSON object containing:
            {{
                "confidence": <confidence_score_0_to_1>,
                "target_position": [<x_pixel>, <y_pixel>],
                "description": "<what_you_found>",
                "depth_estimate": <estimated_depth_in_meters>
            }}
            
            Focus on finding the {target_description} and provide accurate pixel coordinates.
            """
            
            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            try:
                result = json.loads(response.content[0].text)
                return result
            except json.JSONDecodeError:
                print("Failed to parse AI response as JSON")
                return None
                
        except Exception as e:
            print(f"AI vision error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if AI vision is available"""
        return self.client is not None

class RGBDFeed:
    """Simplified RGB-D feed for demonstration"""
    
    def __init__(self):
        self.depth = None
        self.rgb = None
        self.intrinsic_mat = None
        self.last_mouse_pos = None
        self.depth_value = None
        
    def get_dummy_frames(self):
        """Generate dummy RGB-D frames for demonstration"""
        # Create dummy RGB image
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(rgb, "RGB Stream", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create dummy depth image
        depth = np.random.uniform(0.5, 2.0, (480, 640)).astype(np.float32)
        
        # Create dummy intrinsic matrix
        intrinsic_mat = np.array([
            [640, 0, 320],
            [0, 480, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return rgb, depth, intrinsic_mat

class AIKinematicsApp:
    def __init__(self):
        print("=" * 60)
        print("AI KINEMATICS EXAMPLE")
        print("Vision-based Robot Control")
        print("=" * 60)
        
        # Initialize RGB-D feed
        self.rgbd_feed = RGBDFeed()
        
        # Initialize vision analyzer
        self.vision_analyzer = VisionAnalyzer()
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # State variables
        self.current_rgb = None
        self.current_depth = None
        self.rgb_intrinsic_mat = None
        self.depth_intrinsic_mat = None
        self.clicked_pixel = None
        self.clicked_marker = None
        self.current_pose_T_cam_marker = None
        self.vision_mode = False
        self.vision_result = None
        self.vision_target_description = "red cup"
        
        # Robot control settings
        self.robot_endpoint = ROBOT_ENDPOINT
        self.position_tolerance = POSITION_TOLERANCE
        self.orientation_tolerance = ORIENTATION_TOLERANCE
        
        print("System initialized!")
        print("Controls:")
        print("  - Click on RGB image to select target")
        print("  - Press 'q' to quit")
        print("  - Press 'm' to toggle AI mode")
        print("  - Press 'r' to reset robot")
        print("  - Press 'a' to analyze with AI")

    def estimate_pose_from_corners(self, corners_2d):
        """Estimate pose from ArUco marker corners"""
        obj_pts = np.array([[0, 0, 0],
                            [ARUCO_MARKER_SIZE, 0, 0],
                            [ARUCO_MARKER_SIZE, ARUCO_MARKER_SIZE, 0],
                            [0, ARUCO_MARKER_SIZE, 0]], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(obj_pts, corners_2d, self.rgb_intrinsic_mat, self.dist_coeffs)
        if not success:
            return None
            
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def detect_aruco_marker(self):
        """Detect ArUco marker in the current RGB image"""
        if self.current_rgb is None or self.rgb_intrinsic_mat is None:
            return False
            
        gray = cv2.cvtColor(self.current_rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None and ARUCO_MARKER_ID in ids.flatten():
            idx = list(ids.flatten()).index(ARUCO_MARKER_ID)
            self.current_pose_T_cam_marker = self.estimate_pose_from_corners(corners[idx][0])
            return self.current_pose_T_cam_marker is not None
        else:
            self.current_pose_T_cam_marker = None
            return False

    def pixel_to_3d_position(self, u, v, depth_value, camera_matrix=None):
        """Convert pixel coordinates to 3D position"""
        if camera_matrix is None:
            camera_matrix = self.rgb_intrinsic_mat
            
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Convert to camera coordinates
        x = (u - cx) * depth_value / fx
        y = (v - cy) * depth_value / fy
        z = depth_value
        
        return np.array([x, y, z])

    def convert_to_robot_frame(self, camera_coords):
        """Convert camera coordinates to robot frame"""
        if self.current_pose_T_cam_marker is None:
            print("No ArUco marker detected - using default transformation")
            # Default transformation (camera to robot)
            T_cam_robot = np.array([
                [0, 0, 1, 0.1],
                [-1, 0, 0, 0.2],
                [0, -1, 0, 0.3],
                [0, 0, 0, 1]
            ])
        else:
            # Use ArUco marker for calibration
            T_cam_robot = self.current_pose_T_cam_marker @ np.array([
                [0, 0, 1, 0],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]
            ])
        
        # Transform point
        point_cam = np.append(camera_coords, 1)
        point_robot = T_cam_robot @ point_cam
        return point_robot[:3]

    def move_robot_to_position(self, position):
        """Move robot to the specified position"""
        try:
            data = {
                "x": position[0],
                "y": position[1], 
                "z": position[2],
                "rx": 0,
                "ry": 0,
                "rz": 0,
                "open": 1
            }
            
            response = requests.post(self.robot_endpoint, json=data, timeout=5)
            response.raise_for_status()
            print(f"Robot moved to position: {position}")
            return True
            
        except Exception as e:
            print(f"Failed to move robot: {e}")
            return False

    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse events for target selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_pixel = (x, y)
            print(f"Selected pixel: ({x}, {y})")
            
            # Get depth at this pixel
            if self.current_depth is not None:
                depth_val = self.current_depth[y, x]
                if depth_val > 0:
                    # Convert to 3D position
                    pos_3d = self.pixel_to_3d_position(x, y, depth_val)
                    print(f"3D position: {pos_3d}")
                    
                    # Convert to robot frame
                    robot_pos = self.convert_to_robot_frame(pos_3d)
                    print(f"Robot position: {robot_pos}")
                    
                    # Move robot
                    self.move_robot_to_position(robot_pos)
                else:
                    print("Invalid depth at selected pixel")

    def process_ai_vision(self):
        """Process AI vision if available"""
        if not self.vision_analyzer.is_available() or self.current_rgb is None:
            return
            
        try:
            result = self.vision_analyzer.analyze_image(self.current_rgb, self.vision_target_description, self.current_depth)
            self.vision_result = result
            
            if result and result.get("confidence", 0) > 0.7:
                target_pos = result.get("target_position", [320, 240])
                x, y = target_pos
                
                print(f"AI detected target: {result.get('description', 'object')}")
                print(f"Target position: ({x}, {y})")
                
                # Get depth at AI-detected position
                if self.current_depth is not None and 0 <= y < self.current_depth.shape[0] and 0 <= x < self.current_depth.shape[1]:
                    depth_val = self.current_depth[y, x]
                    if depth_val > 0:
                        # Convert to 3D position
                        pos_3d = self.pixel_to_3d_position(x, y, depth_val)
                        print(f"3D position: {pos_3d}")
                        
                        # Convert to robot frame
                        robot_pos = self.convert_to_robot_frame(pos_3d)
                        print(f"Robot position: {robot_pos}")
                        
                        # Move robot
                        self.move_robot_to_position(robot_pos)
                    else:
                        print("Invalid depth at AI-detected position")
                
        except Exception as e:
            print(f"AI vision error: {e}")

    def run(self):
        """Main application loop"""
        print("Starting AI Kinematics application...")
        print("Press 'q' to quit, 'm' to toggle AI mode, 'r' to reset, 'a' for AI analysis")
        
        # Create windows
        cv2.namedWindow('RGB Stream')
        cv2.setMouseCallback('RGB Stream', self.on_mouse)
        
        cv2.namedWindow('Depth Stream')
        
        try:
            while True:
                # Get RGB-D frames
                self.current_rgb, self.current_depth, intrinsic_mat = self.rgbd_feed.get_dummy_frames()
                self.rgb_intrinsic_mat = intrinsic_mat
                self.depth_intrinsic_mat = intrinsic_mat
                
                # Detect ArUco marker
                marker_detected = self.detect_aruco_marker()
                
                # Process AI vision if in AI mode
                if self.vision_mode and self.vision_analyzer.is_available():
                    self.process_ai_vision()
                
                # Display images
                rgb_display = self.current_rgb.copy()
                if marker_detected:
                    cv2.putText(rgb_display, "Marker Detected", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.vision_mode:
                    cv2.putText(rgb_display, "AI MODE", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(rgb_display, "MANUAL MODE", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show AI result if available
                if self.vision_result:
                    cv2.putText(rgb_display, f"AI: {self.vision_result.get('description', '')}", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Normalize depth for display
                depth_display = (self.current_depth * 255 / 2.0).astype(np.uint8)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.putText(depth_display, "Depth Stream", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('RGB Stream', rgb_display)
                cv2.imshow('Depth Stream', depth_display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.vision_mode = not self.vision_mode
                    print(f"Switched to {'AI' if self.vision_mode else 'Manual'} mode")
                elif key == ord('r'):
                    print("Resetting robot position...")
                    self.move_robot_to_position([0, 0, 0.2])
                elif key == ord('a'):
                    print("Running AI analysis...")
                    self.process_ai_vision()
                
                time.sleep(0.1)  # Small delay
                
        except KeyboardInterrupt:
            print("Application stopped by user")
        finally:
            cv2.destroyAllWindows()
            print("Application closed")

def main():
    """Main entry point"""
    try:
        app = AIKinematicsApp()
        app.run()
        
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check that the robot endpoint is correct")
        print("3. Verify your Anthropic API key is set (for AI features)")
        print("4. Ensure the robot is powered on and accessible")

if __name__ == "__main__":
    main() 