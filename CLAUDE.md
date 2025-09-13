# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
# Single-model detection (original implementation)
uv run python main.py [--model MODEL] [--confidence 0.5] [--camera 0] [--pose]

# Multi-model detection (supports both YOLO and MediaPipe)
uv run python main_multi.py [--detector yolo|mediapipe] [--model MODEL] [--confidence 0.5]

# Performance benchmarking
uv run python benchmark.py

# Test scripts
uv run python test_camera.py          # Basic camera test
uv run python simple_camera_test.py   # Simple camera validation
uv run python test_colors.py          # Color detection test
uv run python test_multi_person.py    # Multi-person detection test
```

### Environment Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name

# Add MediaPipe support (optional)
uv add mediapipe
```

## Code Architecture

This is a **dual-architecture human detection system** with two main entry points and a unified detector interface.

### Core Architecture Pattern
The project implements a **Strategy Pattern** for different detection backends:

1. **Unified Interface**: `BaseDetector` abstract class in `detectors.py`
2. **Detection Strategies**: `YOLODetector` and `MediaPipeDetector` implementations
3. **Factory Pattern**: `create_detector()` function for detector instantiation
4. **Unified Results**: `DetectionResult` dataclass for consistent output format

### Main Components

**Entry Points:**
- `main.py` - Original single-model implementation (YOLO-focused)
- `main_multi.py` - Multi-model implementation with detector selection

**Core Detection Framework:**
- `detectors.py` - Abstract detector interface and implementations
  - `BaseDetector` - Abstract base class
  - `YOLODetector` - YOLO model wrapper (detection/segmentation/pose)  
  - `MediaPipeDetector` - MediaPipe wrapper with multi-person support
  - `DetectionResult` - Unified result data structure
  - `create_detector()` - Factory function

**Configuration:**
- `config.py` - Centralized configuration (colors, models, camera settings)

### Detection Capabilities Auto-Discovery
Each detector reports its capabilities via `get_capabilities()`:
```python
{
    'bbox': True,
    'pose': True/False,
    'segmentation': True/False, 
    'multi_person': True/False,
    'keypoints_count': 17/33,
    'model_type': 'YOLO'/'MediaPipe'
}
```

### Keypoint Format Conversion
The system handles two keypoint formats:
- **YOLO**: 17 keypoints (COCO format)
- **MediaPipe**: 33 keypoints (full body landmarks)

MediaPipe keypoints are automatically converted to COCO format for consistent visualization using the mapping in `MEDIAPIPE_TO_COCO_MAP`.

### Multi-Person Detection Strategy
- **YOLO**: Native multi-person support
- **MediaPipe**: Hybrid approach using YOLO for person detection + MediaPipe for pose estimation on cropped regions

### Model Types and Capabilities
**YOLO Models:**
- Detection: `yolov8n.pt`, `yolov8s.pt`, etc.
- Segmentation: `yolov8n-seg.pt`, `yolov8s-seg.pt`, etc.  
- Pose: `yolov8n-pose.pt`, `yolov8s-pose.pt`, etc.

**MediaPipe:**
- Single detector with pose, segmentation, and landmark capabilities
- Requires `mediapipe` package installation

### Interactive Controls
Both applications support runtime visualization toggles:
- `b` - Toggle bounding boxes
- `m` - Toggle segmentation masks
- `p` - Toggle pose keypoints  
- `h` - Toggle help text
- `q` - Quit

### Performance Characteristics
- **YOLO**: ~110-200ms inference, high accuracy, excellent multi-person
- **MediaPipe**: ~5-20ms inference, good accuracy, primarily single-person
- **Camera**: AVFoundation backend on macOS, 640x480 @ 30fps default

## Project Structure Notes
- Model files (`.pt`) are downloaded automatically on first run
- Test files demonstrate specific functionality aspects
- Chinese documentation in `MULTI_MODEL_GUIDE.md` provides detailed usage examples
- Configuration centralized in `config.py` with available model mappings
- No formal test framework - uses manual test scripts