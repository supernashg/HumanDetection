"""
Multi-model human detection architecture
Supports both YOLO and MediaPipe detection methods
"""
import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """Unified detection result data structure"""
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    confidence: float = 0.0
    keypoints: Optional[np.ndarray] = None  # (N, 3) 格式: [x, y, confidence]
    mask: Optional[np.ndarray] = None
    person_id: int = 0


class BaseDetector(ABC):
    """Base detector class"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect humans and return results"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> dict:
        """Return detector capabilities"""
        pass


class YOLODetector(BaseDetector):
    """YOLO detector wrapper class"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        super().__init__(confidence_threshold)
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.has_segmentation = 'seg' in model_path.lower()
        self.has_pose = 'pose' in model_path.lower()
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect using YOLO"""
        results = []
        
        # Choose detection method based on model type
        if self.has_pose:
            yolo_results = self.model(frame, conf=self.confidence_threshold)
        else:
            yolo_results = self.model(frame, conf=self.confidence_threshold, classes=[0])  # Only detect persons
        
        if not yolo_results or not yolo_results[0].boxes:
            return results
        
        result = yolo_results[0]
        
        # 处理每个检测到的人
        for i, box in enumerate(result.boxes):
            detection = DetectionResult()
            
            # 边界框
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detection.bbox = (x1, y1, x2, y2)
            detection.confidence = float(box.conf[0].cpu().numpy())
            detection.person_id = i
            
            # 关键点（如果有）
            if self.has_pose and result.keypoints is not None and i < len(result.keypoints.data):
                keypoints = result.keypoints.data[i].cpu().numpy()
                detection.keypoints = keypoints
            
            # Segmentation mask (if available)
            if self.has_segmentation and result.masks is not None and i < len(result.masks.data):
                mask = result.masks.data[i].cpu().numpy().astype(np.uint8) * 255
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                detection.mask = mask_resized
            
            results.append(detection)
        
        return results
    
    def get_capabilities(self) -> dict:
        return {
            'bbox': True,
            'pose': self.has_pose,
            'segmentation': self.has_segmentation,
            'multi_person': True,
            'keypoints_count': 17 if self.has_pose else 0,
            'model_type': 'YOLO'
        }


class MediaPipeDetector(BaseDetector):
    """MediaPipe detector wrapper class with multi-person support"""
    
    def __init__(self, confidence_threshold: float = 0.5, use_coco_format: bool = False, enable_multi_person: bool = True, 
                 use_native_multi_person: bool = False, num_poses: int = 10):
        super().__init__(confidence_threshold)
        self.mp_pose = None
        self.pose = None
        self.pose_landmarker = None
        self.use_coco_format = use_coco_format
        self.enable_multi_person = enable_multi_person
        self.use_native_multi_person = use_native_multi_person
        self.num_poses = num_poses
        self.yolo_detector = None
        self._initialize_mediapipe()
        
        # Initialize YOLO for person detection if multi-person is enabled and not using native
        if self.enable_multi_person and not self.use_native_multi_person:
            self._initialize_yolo_for_detection()
    
    def toggle_keypoint_format(self):
        """Toggle between MediaPipe 33-point and COCO 17-point format"""
        self.use_coco_format = not self.use_coco_format
    
    def _initialize_yolo_for_detection(self):
        """Initialize YOLO for person detection only"""
        try:
            from ultralytics import YOLO
            self.yolo_detector = YOLO('yolov8n.pt')  # Use lightweight model for person detection
        except Exception as e:
            print(f"Warning: Could not initialize YOLO for multi-person detection: {e}")
            print("Falling back to single-person MediaPipe detection")
            self.enable_multi_person = False
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe (requires mediapipe package)"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            
            if self.use_native_multi_person:
                # Use newer MediaPipe Tasks API with num_poses parameter
                try:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision
                    
                    # Create base options and PoseLandmarker options
                    base_options = python.BaseOptions()
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        num_poses=self.num_poses,
                        min_pose_detection_confidence=self.confidence_threshold,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                    print(f"Initialized MediaPipe native multi-person detection with num_poses={self.num_poses}")
                except Exception as e:
                    print(f"Warning: Could not initialize native multi-person MediaPipe: {e}")
                    print("Falling back to standard MediaPipe Pose")
                    self.use_native_multi_person = False
            
            if not self.use_native_multi_person:
                # Standard MediaPipe Pose (single-person)
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,  # Disable segmentation for multi-person to avoid size conflicts
                    min_detection_confidence=self.confidence_threshold,
                    min_tracking_confidence=0.5
                )
        except ImportError:
            raise ImportError("Please install mediapipe: pip install mediapipe")
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect using MediaPipe with multi-person support"""
        if self.use_native_multi_person and self.pose_landmarker:
            return self._detect_native_multi_person(frame)
        elif not self.pose:
            return []
        elif self.enable_multi_person and self.yolo_detector:
            return self._detect_multi_person(frame)
        else:
            return self._detect_single_person(frame)
    
    def _detect_single_person(self, frame: np.ndarray) -> List[DetectionResult]:
        """Original single-person MediaPipe detection"""
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.pose.process(rgb_frame)
        
        if mp_results.pose_landmarks:
            detection = self._create_detection_from_landmarks(
                mp_results.pose_landmarks.landmark, 
                frame.shape[:2], 
                mp_results.segmentation_mask,
                person_id=0
            )
            if detection:
                results.append(detection)
        
        return results
    
    def _detect_native_multi_person(self, frame: np.ndarray) -> List[DetectionResult]:
        """Native MediaPipe multi-person detection using PoseLandmarker with num_poses"""
        results = []
        
        try:
            import mediapipe as mp
            
            # Convert frame to MediaPipe Image format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Get timestamp for video mode
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # Detect poses
            detection_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.pose_landmarks:
                for i, pose_landmarks in enumerate(detection_result.pose_landmarks):
                    # Convert landmarks to our format
                    landmarks_list = []
                    for landmark in pose_landmarks:
                        landmarks_list.append(landmark)
                    
                    detection = self._create_detection_from_landmarks(
                        landmarks_list,
                        frame.shape[:2],
                        segmentation_mask=None,  # Native method doesn't provide segmentation
                        person_id=i
                    )
                    if detection:
                        results.append(detection)
                        
        except Exception as e:
            print(f"Error in native multi-person detection: {e}")
            return []
        
        return results
    
    def _detect_multi_person(self, frame: np.ndarray) -> List[DetectionResult]:
        """Multi-person detection using YOLO + MediaPipe"""
        results = []
        
        # First, use YOLO to detect people
        yolo_results = self.yolo_detector(frame, conf=self.confidence_threshold, classes=[0])  # Only detect persons
        
        if not yolo_results or not yolo_results[0].boxes:
            return results
        
        yolo_result = yolo_results[0]
        
        # Process each detected person
        for i, box in enumerate(yolo_result.boxes):
            # Get person bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Add some padding to the crop
            padding = 20
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)
            
            # Crop the person from the frame
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if person_crop.size == 0:
                continue
            
            # Run MediaPipe pose estimation on the cropped person
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            mp_results = self.pose.process(rgb_crop)
            
            if mp_results.pose_landmarks:
                # Adjust coordinates back to original frame
                detection = self._create_detection_from_landmarks(
                    mp_results.pose_landmarks.landmark,
                    person_crop.shape[:2],
                    mp_results.segmentation_mask,
                    person_id=i,
                    offset=(crop_x1, crop_y1)
                )
                if detection:
                    results.append(detection)
        
        return results
    
    def _create_detection_from_landmarks(self, landmarks, frame_shape, segmentation_mask=None, person_id=0, offset=(0, 0)):
        """Create DetectionResult from MediaPipe landmarks"""
        detection = DetectionResult()
        h, w = frame_shape
        offset_x, offset_y = offset
        
        keypoints = []
        x_coords, y_coords = [], []
        
        for landmark in landmarks:
            x = int(landmark.x * w) + offset_x
            y = int(landmark.y * h) + offset_y
            confidence = landmark.visibility
            keypoints.append([x, y, confidence])
            x_coords.append(x)
            y_coords.append(y)
        
        # Convert keypoints format if needed
        if self.use_coco_format:
            keypoints_array = np.array(keypoints)
            detection.keypoints = convert_mediapipe_to_coco(keypoints_array)
        else:
            detection.keypoints = np.array(keypoints)
        
        # Generate bounding box
        if x_coords and y_coords:
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            # Add margin
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame_shape[1] + offset_x, x2 + margin)
            y2 = min(frame_shape[0] + offset_y, y2 + margin)
            detection.bbox = (x1, y1, x2, y2)
        
        # Calculate overall confidence (average visibility)
        detection.confidence = float(np.mean([kp[2] for kp in keypoints]))
        detection.person_id = person_id
        
        # Segmentation mask (if available)
        if segmentation_mask is not None:
            mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
            # Resize mask to match the crop size first
            mask_resized = cv2.resize(mask, (w, h))
            
            # For multi-person detection, we need to handle offset properly
            if offset != (0, 0):
                # Create a mask that matches the original frame size
                original_frame_h = frame_shape[0] + offset_y
                original_frame_w = frame_shape[1] + offset_x
                full_mask = np.zeros((original_frame_h, original_frame_w), dtype=np.uint8)
                
                # Place the resized mask at the correct offset
                end_y = min(offset_y + h, original_frame_h)
                end_x = min(offset_x + w, original_frame_w)
                actual_h = end_y - offset_y
                actual_w = end_x - offset_x
                
                if actual_h > 0 and actual_w > 0:
                    mask_to_place = cv2.resize(mask_resized, (actual_w, actual_h))
                    full_mask[offset_y:end_y, offset_x:end_x] = mask_to_place
                detection.mask = full_mask
            else:
                detection.mask = mask_resized
        
        return detection
    
    def get_capabilities(self) -> dict:
        model_type = 'MediaPipe'
        if self.use_native_multi_person:
            model_type += ' (Native Multi-Person)'
        elif self.enable_multi_person:
            model_type += ' + YOLO'
            
        return {
            'bbox': True,
            'pose': True,
            'segmentation': not self.enable_multi_person and not self.use_native_multi_person,  # Segmentation disabled for multi-person
            'multi_person': self.enable_multi_person or self.use_native_multi_person,
            'keypoints_count': 17 if self.use_coco_format else 33,
            'model_type': model_type,
            'can_toggle_keypoints': True,
            'native_multi_person': self.use_native_multi_person
        }


def create_detector(detector_type: str, **kwargs) -> BaseDetector:
    """Detector factory function"""
    if detector_type.lower() == 'yolo':
        return YOLODetector(**kwargs)
    elif detector_type.lower() == 'mediapipe':
        # MediaPipe doesn't need model_path parameter
        mediapipe_kwargs = {k: v for k, v in kwargs.items() if k != 'model_path'}
        # Enable multi-person by default for MediaPipe
        if 'enable_multi_person' not in mediapipe_kwargs:
            mediapipe_kwargs['enable_multi_person'] = True
        return MediaPipeDetector(**mediapipe_kwargs)
    elif detector_type.lower() == 'mediapipe-native':
        # MediaPipe with native multi-person detection
        mediapipe_kwargs = {k: v for k, v in kwargs.items() if k != 'model_path'}
        mediapipe_kwargs['use_native_multi_person'] = True
        mediapipe_kwargs['enable_multi_person'] = True
        return MediaPipeDetector(**mediapipe_kwargs)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")


# MediaPipe and YOLO keypoint mapping
MEDIAPIPE_TO_COCO_MAP = {
    # MediaPipe 33 points mapped to COCO 17 points
    0: 0,   # nose
    2: 1,   # left_eye  
    5: 2,   # right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16, # right_ankle
}


def convert_mediapipe_to_coco(mp_keypoints: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-point format to COCO 17-point format"""
    coco_keypoints = np.zeros((17, 3))
    
    for mp_idx, coco_idx in MEDIAPIPE_TO_COCO_MAP.items():
        if mp_idx < len(mp_keypoints):
            coco_keypoints[coco_idx] = mp_keypoints[mp_idx]
    
    return coco_keypoints