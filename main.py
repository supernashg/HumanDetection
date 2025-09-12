import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import sys


class HumanDetector:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5, enable_pose=False):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.cap = None
        self.enable_pose = enable_pose
        
        # COCO pose keypoint connections for drawing skeleton
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Keypoint names for COCO pose
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def setup_camera(self, camera_id=0):
        # Use AVFoundation backend explicitly for macOS
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id} with AVFoundation backend")
        
        print(f"Camera {camera_id} opened successfully!")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Give camera time to initialize
        import time
        time.sleep(2)
        
        # Test frame grab
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera opened but cannot grab frames")
        
        print(f"Camera test successful - Frame shape: {test_frame.shape}, Mean: {test_frame.mean():.2f}")
        
    def detect_humans(self, frame):
        if self.enable_pose:
            # For pose models, don't filter by class as pose models only detect persons
            results = self.model(frame, conf=self.confidence)
        else:
            # For detection/segmentation models, filter for person class
            results = self.model(frame, conf=self.confidence, classes=[0])
        return results[0] if results else None
        
    def draw_detections(self, frame, result):
        if result is None:
            return frame
            
        annotated_frame = frame.copy()
        
        # Draw bounding boxes if available
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = 'Pose' if self.enable_pose else 'Human'
                cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw segmentation masks if available
        if result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 1] = mask_resized  # Green channel
                
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, mask_colored, 0.2, 0)
        
        # Draw pose keypoints and skeleton if available
        if self.enable_pose and result.keypoints is not None:
            annotated_frame = self.draw_pose_keypoints(annotated_frame, result.keypoints)
                
        return annotated_frame
    
    def draw_pose_keypoints(self, frame, keypoints):
        annotated_frame = frame.copy()
        
        for person_keypoints in keypoints.data:
            # Convert keypoints to numpy array
            kpts = person_keypoints.cpu().numpy()
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(kpts):
                if conf > 0.5:  # Only draw visible keypoints
                    x, y = int(x), int(y)
                    # Different colors for different body parts
                    if i < 5:  # Head keypoints (nose, eyes, ears)
                        color = (255, 0, 0)  # Blue
                    elif i < 11:  # Arms (shoulders, elbows, wrists)
                        color = (0, 255, 0)  # Green
                    else:  # Legs (hips, knees, ankles)
                        color = (0, 0, 255)  # Red
                    
                    cv2.circle(annotated_frame, (x, y), 4, color, -1)
                    cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), 2)
            
            # Draw skeleton connections
            for connection in self.pose_connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(kpts) and pt2_idx < len(kpts):
                    pt1 = kpts[pt1_idx]
                    pt2 = kpts[pt2_idx]
                    
                    # Only draw if both keypoints are visible
                    if pt1[2] > 0.5 and pt2[2] > 0.5:
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        return annotated_frame
        
    def run(self):
        try:
            self.setup_camera()
            print("Starting human detection. Press 'q' to quit.")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                result = self.detect_humans(frame)
                annotated_frame = self.draw_detections(frame, result)
                
                cv2.imshow('Human Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping detection...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Human Detection and Pose Estimation using YOLOv8')
    parser.add_argument('--model', default='yolov8n-seg.pt', help='YOLOv8 model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--pose', action='store_true', help='Enable pose estimation (requires pose model)')
    
    args = parser.parse_args()
    
    # Auto-detect pose models and enable pose estimation
    enable_pose = args.pose or 'pose' in args.model.lower()
    
    # Suggest pose model if --pose is used with non-pose model
    if args.pose and 'pose' not in args.model.lower():
        print(f"Warning: --pose flag used with model '{args.model}' which may not be a pose model.")
        print("Consider using a pose model like 'yolov8n-pose.pt' for best results.")
    
    detector = HumanDetector(model_path=args.model, confidence=args.confidence, enable_pose=enable_pose)
    detector.run()


if __name__ == "__main__":
    main()
