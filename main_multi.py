import cv2
import numpy as np
import argparse
import sys
from detectors import create_detector, convert_mediapipe_to_coco, DetectionResult
from typing import List


class MultiModelHumanDetector:
    """Multi-model human detector supporting various models"""
    
    def __init__(self, detector_type='yolo', model_path='yolov8n.pt', confidence=0.5,
                 show_bbox=True, show_mask=True, show_pose=True):
        self.detector = create_detector(
            detector_type, 
            model_path=model_path if detector_type == 'yolo' else None,
            confidence_threshold=confidence
        )
        self.cap = None
        self.capabilities = self.detector.get_capabilities()
        
        # Visualization toggles
        self.show_bbox = show_bbox
        self.show_mask = show_mask and self.capabilities['segmentation']
        self.show_pose = show_pose and self.capabilities['pose']
        self.show_help = True
        
        # COCO pose connections (17 keypoints)
        self.coco_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # MediaPipe pose connections (33 keypoints)
        self.mediapipe_connections = [
            # Face connections
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Torso
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32)
        ]
        
        print(f"Initialized {self.capabilities['model_type']} detector")
        print(f"Model capabilities: {self.capabilities}")
    
    def setup_camera(self, camera_id=0):
        """Setup camera"""
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        print(f"Camera {camera_id} opened successfully!")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        import time
        time.sleep(2)
        
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("摄像头已打开但无法获取帧")
        
        print(f"摄像头测试成功 - 帧尺寸: {test_frame.shape}, 均值: {test_frame.mean():.2f}")
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """绘制检测结果"""
        annotated_frame = frame.copy()
        
        # Define colors for different persons
        person_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for detection in detections:
            # Get color for this person
            person_color = person_colors[detection.person_id % len(person_colors)]
            
            # Draw bounding box
            if self.show_bbox and detection.bbox:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), person_color, 2)
                
                model_name = self.capabilities['model_type']
                label = f'Person {detection.person_id} {model_name} {detection.confidence:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
            
            # Draw segmentation mask
            if self.show_mask and detection.mask is not None:
                # Resize mask to match frame size if needed
                if detection.mask.shape[:2] != frame.shape[:2]:
                    mask_resized = cv2.resize(detection.mask, (frame.shape[1], frame.shape[0]))
                else:
                    mask_resized = detection.mask
                
                mask_colored = np.zeros_like(frame)
                # Use person-specific color for mask
                b, g, r = person_color
                mask_colored[:, :, 0] = (mask_resized * b / 255).astype(np.uint8)
                mask_colored[:, :, 1] = (mask_resized * g / 255).astype(np.uint8) 
                mask_colored[:, :, 2] = (mask_resized * r / 255).astype(np.uint8)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, mask_colored, 0.2, 0)
            
            # Draw pose keypoints
            if self.show_pose and detection.keypoints is not None:
                annotated_frame = self.draw_pose_keypoints(annotated_frame, detection.keypoints, person_color)
        
        # 绘制帮助文本
        if self.show_help:
            annotated_frame = self.draw_help_text(annotated_frame)
        
        return annotated_frame
    
    def draw_pose_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, base_color: tuple = (0, 255, 0)) -> np.ndarray:
        """Draw pose keypoints with person-specific base color"""
        annotated_frame = frame.copy()
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # Only draw visible keypoints
                x, y = int(x), int(y)
                
                # For multi-person, use variations of the base color for different body parts
                # Different colors for different body parts based on keypoint format
                if len(keypoints) == 33:  # MediaPipe 33-point format
                    if i <= 10:  # Face landmarks (0-10)
                        color = self._adjust_color(base_color, 0.8)  # Lighter
                    elif i <= 16:  # Upper body (11-16: shoulders, elbows, wrists)
                        color = base_color  # Base color
                    elif i <= 22:  # Hands (17-22)
                        color = self._adjust_color(base_color, 1.2)  # Brighter
                    elif i <= 28:  # Lower body (23-28: hips, knees, ankles)
                        color = self._adjust_color(base_color, 0.6)  # Darker
                    else:  # Feet (29-32)
                        color = self._adjust_color(base_color, 1.4)  # Much brighter
                else:  # COCO 17-point format
                    if i < 5:  # Head
                        color = self._adjust_color(base_color, 0.8)  # Lighter
                    elif i < 11:  # Arms
                        color = base_color  # Base color
                    else:  # Legs
                        color = self._adjust_color(base_color, 0.6)  # Darker
                
                cv2.circle(annotated_frame, (x, y), 4, color, -1)
                cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), 2)
        
        # Draw skeleton connections
        # Choose connections based on keypoint count
        connections = self.mediapipe_connections if len(keypoints) == 33 else self.coco_connections
        
        for connection in connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                
                if pt1[2] > 0.5 and pt2[2] > 0.5:  # Both points are visible
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), base_color, 2)
        
        return annotated_frame
    
    def _adjust_color(self, color: tuple, factor: float) -> tuple:
        """Adjust color brightness by factor"""
        b, g, r = color
        b = min(255, max(0, int(b * factor)))
        g = min(255, max(0, int(g * factor)))
        r = min(255, max(0, int(r * factor)))
        return (b, g, r)
    
    def draw_help_text(self, frame: np.ndarray) -> np.ndarray:
        """Draw help text"""
        help_lines = [
            f"Model: {self.capabilities['model_type']}",
            "Controls:",
            "  'b' - Toggle bounding box",
            "  'm' - Toggle segmentation mask", 
            "  'p' - Toggle pose keypoints",
            "  'k' - Toggle keypoint format (MediaPipe only)",
            "  'h' - Toggle help text",
            "  'q' - Exit"
        ]
        
        # Status indicators
        status_lines = [
            "",
            f"Bounding box: {'On' if self.show_bbox else 'Off'}",
            f"Segmentation mask: {'On' if self.show_mask else 'Off' if self.capabilities['segmentation'] else 'Unavailable'}",
            f"Pose: {'On' if self.show_pose else 'Off' if self.capabilities['pose'] else 'Unavailable'}",
            f"Multi-person: {'On' if self.capabilities['multi_person'] else 'Off'}",
            f"Keypoints count: {self.capabilities['keypoints_count']}"
        ]
        
        all_lines = help_lines + status_lines
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, len(all_lines) * 25 + 20), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        for i, line in enumerate(all_lines):
            color = (0, 255, 255) if line.endswith(':') else (255, 255, 255)
            cv2.putText(frame, line, (20, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        return frame
    
    def handle_key_press(self, key):
        """Handle key press"""
        if key == ord('b'):
            self.show_bbox = not self.show_bbox
            print(f"Bounding box: {'On' if self.show_bbox else 'Off'}")
        elif key == ord('m'):
            if self.capabilities['segmentation']:
                self.show_mask = not self.show_mask
                print(f"Segmentation mask: {'On' if self.show_mask else 'Off'}")
            else:
                print(f"Segmentation mask not available for {self.capabilities['model_type']} model")
        elif key == ord('p'):
            if self.capabilities['pose']:
                self.show_pose = not self.show_pose
                print(f"Pose detection: {'On' if self.show_pose else 'Off'}")
            else:
                print(f"Pose detection not available for current {self.capabilities['model_type']} model")
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help text: {'On' if self.show_help else 'Off'}")
        elif key == ord('k'):
            if self.capabilities.get('can_toggle_keypoints', False):
                self.detector.toggle_keypoint_format()
                self.capabilities = self.detector.get_capabilities()
                format_type = "COCO 17-point" if self.detector.use_coco_format else "MediaPipe 33-point"
                print(f"Keypoint format switched to: {format_type}")
            else:
                print("Current model doesn't support keypoint format toggle")
    
    def run(self):
        """Run detection"""
        try:
            self.setup_camera()
            print("Starting human detection")
            print("控制键: 'b'=边界框, 'm'=掩码, 'p'=姿态, 'k'=关键点格式, 'h'=帮助, 'q'=退出")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Unable to capture frame")
                    break
                
                # Detection
                detections = self.detector.detect(frame)
                
                # Draw results
                annotated_frame = self.draw_detections(frame, detections)
                
                cv2.imshow(f'Human Detection - {self.capabilities["model_type"]}', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key != 255:
                    self.handle_key_press(key)
        
        except KeyboardInterrupt:
            print("\n\nStopping detection...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Multi-model human detection and pose estimation')
    parser.add_argument('--detector', choices=['yolo', 'mediapipe'], default='yolo',
                       help='Choose detector type')
    parser.add_argument('--model', default='yolov8n-seg.pt', 
                       help='YOLO model path (for YOLO detector only)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    
    # Visualization toggles
    parser.add_argument('--no-bbox', action='store_true', help='Start with bounding boxes disabled')
    parser.add_argument('--no-mask', action='store_true', help='Start with segmentation masks disabled')
    parser.add_argument('--no-pose', action='store_true', help='Start with pose display disabled')
    parser.add_argument('--no-help', action='store_true', help='Start with help text hidden')
    
    args = parser.parse_args()
    
    try:
        detector = MultiModelHumanDetector(
            detector_type=args.detector,
            model_path=args.model,
            confidence=args.confidence,
            show_bbox=not args.no_bbox,
            show_mask=not args.no_mask,
            show_pose=not args.no_pose
        )
        
        detector.show_help = not args.no_help
        detector.run()
        
    except ImportError as e:
        print(f"Dependency error: {e}")
        print("Please install required packages:")
        if 'mediapipe' in str(e):
            print("  pip install mediapipe")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()