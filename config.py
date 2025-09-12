class Config:
    MODEL_PATH = "yolov8n-seg.pt"
    
    CONFIDENCE_THRESHOLD = 0.5
    
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    DETECTION_COLOR = (0, 255, 0)  # Green
    MASK_COLOR = (0, 255, 0)  # Green
    MASK_ALPHA = 0.2
    
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 2
    
    AVAILABLE_MODELS = {
        "nano": "yolov8n.pt",
        "nano-seg": "yolov8n-seg.pt", 
        "small": "yolov8s.pt",
        "small-seg": "yolov8s-seg.pt",
        "medium": "yolov8m.pt",
        "medium-seg": "yolov8m-seg.pt",
        "large": "yolov8l.pt",
        "large-seg": "yolov8l-seg.pt",
        "xlarge": "yolov8x.pt",
        "xlarge-seg": "yolov8x-seg.pt"
    }