import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import sys


class HumanDetector:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.cap = None
        
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
        results = self.model(frame, conf=self.confidence, classes=[0])  # class 0 is person in COCO
        return results[0] if results else None
        
    def draw_detections(self, frame, result):
        if result is None or result.boxes is None:
            return frame
            
        annotated_frame = frame.copy()
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Human {conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        if result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :, 1] = mask_resized  # Green channel
                
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, mask_colored, 0.2, 0)
                
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
    parser = argparse.ArgumentParser(description='Human Detection using YOLOv8')
    parser.add_argument('--model', default='yolov8n-seg.pt', help='YOLOv8 model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    
    args = parser.parse_args()
    
    detector = HumanDetector(model_path=args.model, confidence=args.confidence)
    detector.run()


if __name__ == "__main__":
    main()
