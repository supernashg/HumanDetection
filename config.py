class Config:
    MODEL_PATH = "yolov8n-seg.pt"
    
    CONFIDENCE_THRESHOLD = 0.5
    
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    DETECTION_COLOR = (0, 255, 0)  # Green
    MASK_COLOR = (0, 255, 0)  # Green
    MASK_ALPHA = 0.2
    
    # Pose estimation colors
    POSE_HEAD_COLOR = (255, 0, 0)      # Blue for head keypoints
    POSE_ARMS_COLOR = (0, 255, 0)      # Green for arms
    POSE_LEGS_COLOR = (0, 0, 255)      # Red for legs
    POSE_SKELETON_COLOR = (255, 255, 0) # Yellow for skeleton lines
    POSE_KEYPOINT_RADIUS = 4
    POSE_SKELETON_THICKNESS = 2
    POSE_CONFIDENCE_THRESHOLD = 0.5
    
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    
    AVAILABLE_MODELS = {
        # Detection models
        "nano": "yolov8n.pt",
        "small": "yolov8s.pt",
        "medium": "yolov8m.pt",
        "large": "yolov8l.pt",
        "xlarge": "yolov8x.pt",
        
        # Segmentation models
        "nano-seg": "yolov8n-seg.pt", 
        "small-seg": "yolov8s-seg.pt",
        "medium-seg": "yolov8m-seg.pt",
        "large-seg": "yolov8l-seg.pt",
        "xlarge-seg": "yolov8x-seg.pt",
        
        # Pose estimation models
        "nano-pose": "yolov8n-pose.pt",
        "small-pose": "yolov8s-pose.pt",
        "medium-pose": "yolov8m-pose.pt",
        "large-pose": "yolov8l-pose.pt",
        "xlarge-pose": "yolov8x-pose.pt"
    }