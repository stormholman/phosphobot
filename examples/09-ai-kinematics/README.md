# Kinematics AI

A computer vision and robotics system that combines RGBD cameras, ArUco marker tracking, and AI-powered object detection for precise robotic manipulation. Designed for non-interactive operation with web application integration.

## Features

- **RGBD Stream Processing**: Real-time RGB-D data from iPhone using Record3D
- **ArUco Marker Tracking**: Precise 6DOF pose estimation for coordinate frame reference
- **AI Vision Analysis**: Claude-powered object detection with natural language descriptions
- **3D Coordinate Transformation**: Pixel → Camera → ArUco Marker → Robot coordinate conversion
- **Robot Control**: HTTP API integration for robotic arm movement
- **Non-Interactive Operation**: Designed for web application integration without user prompts
- **Multiple Operation Modes**: Manual and AI Vision modes
- **Web API Integration**: Remote control via phosphobot framework endpoints

## System Architecture

```
iPhone (Record3D) → RGBD Stream → Computer Vision Pipeline
                                        ↓
ArUco Detection ← RGB Frame ← Coordinate Transformation → Robot Commands
                                        ↓
AI Object Detection ← Depth Frame ← 3D Position Calculation
```

## Installation

### Prerequisites

- Python 3.8+
- iPhone with Record3D app
- ArUco marker (4x4cm, ID 0, DICT_ARUCO_ORIGINAL)
- Anthropic API key for AI vision features

### Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `opencv-python` - Computer vision and ArUco detection
- `numpy` - Numerical computations
- `record3d` - iPhone RGBD streaming
- `anthropic` - AI vision analysis
- `requests` - Robot HTTP communication
- `Pillow` - Image processing

## Quick Start

### Direct Launch
1. **Setup ArUco Marker**
   ```bash
   python generate_aruco.py --id 0 --size 200 --output marker_0.png
   ```
   Print the generated marker at 4x4cm size.

2. **Configure API Key**
   ```bash
   export ANTHROPIC_API_KEY="your_api_key_here"
   ```

3. **Start Record3D** on your iPhone and connect to the same network.

4. **Run the Application**
```bash
# Manual mode
python main.py manual

# AI mode with default task
python main.py ai

# AI mode with custom task
python main.py ai "red cup"
```

### Web API Control
The AI kinematics system can be controlled remotely through the phosphobot web interface:

- **Set API Key**: `POST /kinematics/set-api-key`
- **Check API Key Status**: `GET /kinematics/api-key-status`
- **Launch Kinematics**: `POST /kinematics/launch?mode=manual|ai`
- **Stop Kinematics**: `POST /kinematics/stop`
- **Check Status**: `GET /kinematics/status`

The web API automatically handles process management, API key management, and provides real-time status feedback.

## Web API Integration

The AI kinematics system integrates with the phosphobot web framework through REST API endpoints:

### Available Endpoints

#### API Key Management
- **`POST /kinematics/set-api-key`** - Set Anthropic API key for AI features
  - Body: `{"api_key": "sk-ant-api03-..."}`
  - Returns: Status confirmation
  - Validation: Checks for valid Anthropic API key format

- **`GET /kinematics/api-key-status`** - Check if API key is configured
  - Returns: `{"api_key_set": true/false}`
  - Used by frontend to enable/disable AI mode

#### Process Control
- **`POST /kinematics/launch`** - Launch AI kinematics process
  - Query params: `mode=manual|ai`
  - Body (optional): `{"task": "object description"}` for AI mode
  - Returns: Status confirmation with process PID
  - Background: Launches `python main.py <mode> [task]` in ai_kinematics directory

- **`POST /kinematics/stop`** - Stop running kinematics process
  - Returns: Status confirmation
  - Process: Graceful termination with 5-second timeout

- **`GET /kinematics/status`** - Check kinematics process status
  - Returns: Current status (running/stopped) with PID if active

### API Response Format
```json
{
  "status": "ok",
  "message": "AI-kinematics launched in ai mode"
}
```

### Error Handling
- **Already running**: Returns 400 if kinematics process is already active
- **Missing API key**: Returns 400 if AI mode requested without API key
- **Invalid API key**: Returns 400 for malformed Anthropic API keys
- **Process not found**: Returns 500 if main.py script is missing
- **Launch failure**: Returns 500 with detailed error information
- **Device connection**: Returns 500 with Record3D connection troubleshooting

### Integration with phosphobot
The kinematics endpoints are automatically registered with the main phosphobot FastAPI application and can be accessed through the web dashboard. The system supports:

- **Non-interactive operation**: No user prompts or confirmations
- **Automatic API key management**: Secure in-memory storage
- **Process lifecycle management**: Automatic cleanup and error recovery
- **Real-time status monitoring**: Process state tracking
- **Debug output streaming**: All logs visible in web application console

## Operation Modes

### Manual Mode
- Mouse hover shows real-time 3D coordinates
- Left-click sends coordinates to robot automatically
- No user confirmation required
- Precise manual control

### AI Vision Mode  
- Describe objects in natural language
- AI automatically finds and targets objects
- First frame is analyzed automatically
- Examples: "yellow banana", "red cup on table"

## Key Components

### `main.py`
Main application with RGBD processing, ArUco detection, and robot control. Non-interactive operation only.

### `vision_analyzer.py`  
AI-powered object detection using Claude vision API with accuracy improvements:
- Image upsampling for better detection
- Bounding box validation
- Depth-based pixel refinement
- Precision-focused prompting

### `aruco_transformer.py`
Standalone ArUco detection and coordinate transformation (legacy).

### `aruco_rgbd_stream.py`
Simplified ArUco testing application for pixel-to-marker coordinate conversion.

### `rgbd_feed.py`
Base RGBD streaming functionality using Record3D.

### `position_calculator.py`
3D coordinate calculation utilities.

## Coordinate Systems

### ArUco Marker Frame
- **Origin**: Top-left corner of marker
- **+X**: Rightward along top edge  
- **+Y**: Downward along left edge
- **+Z**: Out of marker plane toward camera

### Robot Frame Mapping
```python
robot_x = -aruco_y    # Robot X = -ArUco Y
robot_y = -aruco_x    # Robot Y = -ArUco X  
robot_z = -aruco_z - 0.13  # Robot Z = -ArUco Z - 13cm offset
```

## Configuration

### Robot Endpoint
Update in `main.py`:
```python
self.robot_endpoint = "http://192.168.178.190/move/absolute?robot_id=0"
```

### ArUco Settings
```python
self.aruco_marker_id = 0           # Marker ID
self.aruco_marker_size = 0.04      # 4cm physical size
```

### Camera Calibration
Automatic calibration from Record3D intrinsics. Manual override available in code.

### Web API Settings
The kinematics endpoints are configured in `phosphobot/endpoints/kinematics.py`:
- Process timeout: 5 seconds for graceful shutdown
- Working directory: ai_kinematics folder
- Command: `python main.py <mode> [task]`
- API key storage: In-memory with validation

## Usage Examples

### Manual Targeting
1. Start application: `python main.py manual`
2. Hover mouse over objects to see 3D coordinates
3. Left-click to send coordinates to robot automatically
4. Robot movement is confirmed automatically

### AI Object Detection
1. Start application: `python main.py ai "red cup"`
2. AI analyzes first frame and finds object
3. Coordinates automatically sent to robot
4. No user interaction required

### Web API Usage
```bash
# Set API key
curl -X POST "http://localhost:8000/kinematics/set-api-key" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "sk-ant-api03-..."}'

# Launch AI mode
curl -X POST "http://localhost:8000/kinematics/launch?mode=ai" \
  -H "Content-Type: application/json" \
  -d '{"task": "red cup"}'

# Check status
curl "http://localhost:8000/kinematics/status"

# Stop process
curl -X POST "http://localhost:8000/kinematics/stop"
```

## Keyboard Controls

- `q` - Quit application

## Web Application Integration

The system is designed to work with web applications through the phosphobot framework:

- **Non-interactive operation**: No user prompts or confirmations
- **Command-line arguments**: Mode and task specified at startup
- **Automatic robot movement**: No manual confirmation required
- **Debug output**: All logs visible in web application console
- **API key management**: Secure storage and validation
- **Process lifecycle**: Automatic startup, monitoring, and cleanup

## Troubleshooting

### ArUco Detection Issues
- Ensure good lighting and contrast
- Check marker is 4x4cm and ID 0
- Verify marker is DICT_ARUCO_ORIGINAL type

### Depth Accuracy
- Ensure RGB and depth frames are aligned
- Check for proper Record3D connection
- Verify camera intrinsics are loaded

### AI Vision Accuracy
- Use specific object descriptions
- Ensure objects are clearly visible
- Good lighting improves detection
- Check API key configuration

### Robot Communication
- Verify robot endpoint URL
- Check network connectivity
- Confirm robot API is accessible
- Test with curl commands

### API Key Issues
- Ensure `ANTHROPIC_API_KEY` environment variable is set
- Verify API key is valid and has sufficient credits
- Check network connectivity to Anthropic API

### Web API Issues
- Check phosphobot server is running
- Verify kinematics endpoints are registered
- Check process permissions for Python execution
- Ensure Record3D device is connected and accessible

### Process Management Issues
- Check if kinematics process is already running
- Verify main.py script exists in ai_kinematics directory
- Check Python dependencies are installed
- Monitor process logs for detailed error information

## API Reference

### VisionAnalyzer.analyze_image()
```python
result = analyzer.analyze_image(rgb_image, "target description", depth_frame)
```

Returns:
```python
{
    "success": True,
    "bbox": [x0, y0, x1, y1],
    "target_pixel": [x, y], 
    "confidence": 0.95,
    "object_description": "found object",
    "reasoning": "why these coordinates"
}
```

### Coordinate Transformation
```python
# Pixel to 3D camera coordinates
pos_3d = pixel_to_3d_position(u, v, depth, camera_matrix)

# Camera to ArUco marker coordinates  
marker_coords = pixel_to_marker_coordinates(rgb_x, rgb_y)

# ArUco to robot coordinates
robot_coords = convert_to_robot_frame(marker_coords)
```

## Development

### Testing ArUco Detection
```bash
python aruco_rgbd_stream.py
```

### Generating Test Markers
```bash
python generate_aruco.py --help
```

### Vision System Testing
```bash
python vision_analyzer.py test
```

### Web Application Testing
```bash
# Test manual mode
python main.py manual

# Test AI mode
python main.py ai "test object"
```

### API Endpoint Testing
```bash
# Test API key management
curl -X POST "http://localhost:8000/kinematics/set-api-key" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "sk-ant-api03-..."}'

# Test process launch
curl -X POST "http://localhost:8000/kinematics/launch?mode=manual"

# Test status check
curl "http://localhost:8000/kinematics/status"
```

## Command Line Usage

```bash
# Manual mode - mouse-based targeting
python main.py manual

# AI mode with default task
python main.py ai

# AI mode with custom task
python main.py ai "specific object description"

# Show help
python main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Record3D for iPhone RGBD streaming
- OpenCV for computer vision capabilities  
- Anthropic Claude for AI vision analysis
- ArUco marker detection system 