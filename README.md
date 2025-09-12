# Human Detection with YOLOv8

A real-time human detection system using YOLOv8 that captures video from your MacBook camera and provides both bounding box detection and segmentation masks for humans in the scene.

## Features

- <¥ **Real-time Detection**: Live video processing from MacBook camera
- =d **Human Detection**: Detects multiple humans simultaneously
- =æ **Bounding Boxes**: Green rectangles around detected humans
- <­ **Segmentation Masks**: Semi-transparent overlays showing human silhouettes
- ™ **Configurable**: Adjustable confidence threshold and model selection
- =€ **High Performance**: ~130-200ms inference time per frame
- =' **Modern Tools**: Uses UV for dependency management

## Installation

### Prerequisites
- macOS (tested on macOS with AVFoundation)
- Python 3.10+
- UV package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/supernashg/HumanDetection.git
   cd HumanDetection
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

## Usage

### Basic Usage
Run the detection with default settings:
```bash
uv run python main.py
```

### Advanced Options
```bash
# Use higher confidence threshold (default: 0.5)
uv run python main.py --confidence 0.7

# Use different YOLOv8 model (default: yolov8n-seg.pt)
uv run python main.py --model yolov8s-seg.pt

# Use different camera (if you have multiple cameras)
uv run python main.py --camera 1

# Combine options
uv run python main.py --model yolov8m-seg.pt --confidence 0.6
```

### Controls
- Press **'q'** to quit the detection window
- The first run will automatically download the YOLOv8 model

## Available Models

The system supports various YOLOv8 models with different performance/accuracy trade-offs:

- `yolov8n-seg.pt` - Nano (fastest, default)
- `yolov8s-seg.pt` - Small
- `yolov8m-seg.pt` - Medium
- `yolov8l-seg.pt` - Large
- `yolov8x-seg.pt` - Extra Large (most accurate)

## Configuration

You can modify detection settings in `config.py`:

- Detection colors
- Camera resolution
- Font settings
- Available model paths

## Technical Details

- **Framework**: YOLOv8 with Ultralytics
- **Backend**: OpenCV with AVFoundation (macOS optimized)
- **Model**: COCO-trained models (class 0 = person)
- **Input**: 640x480 video frames
- **Output**: Real-time bounding boxes and segmentation masks

## Troubleshooting

### Camera Permission Issues
If you get camera access errors:
1. Go to **System Preferences > Security & Privacy > Camera**
2. Enable camera access for your terminal application
3. Restart the application

### Performance Issues
- Use smaller models (nano/small) for better performance
- Lower the confidence threshold if missing detections
- Ensure good lighting conditions

## Dependencies

- ultralytics (YOLOv8)
- opencv-python (computer vision)
- numpy (numerical operations)
- pillow (image processing)
- torch & torchvision (deep learning backend)

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project!