#!/usr/bin/env python3
"""
Demonstration script for person tracking functionality
Shows how tracking maintains consistent person IDs
"""

import cv2
import numpy as np
from person_tracker import PersonTracker
from detectors import DetectionResult
import time


def create_sample_detection(x, y, w, h, confidence=0.8, keypoints=None):
    """Create a sample detection for testing"""
    detection = DetectionResult(
        bbox=(x, y, x + w, y + h),
        confidence=confidence,
        person_id=0  # Will be updated by tracker
    )
    
    if keypoints is None:
        # Create sample COCO format keypoints (17 points)
        detection.keypoints = np.random.rand(17, 3)
        detection.keypoints[:, 0] = detection.keypoints[:, 0] * w + x  # X coordinates
        detection.keypoints[:, 1] = detection.keypoints[:, 1] * h + y  # Y coordinates  
        detection.keypoints[:, 2] = 0.8  # Confidence
    else:
        detection.keypoints = keypoints
        
    return detection


def simulate_tracking_scenario():
    """Simulate a multi-person tracking scenario"""
    print("🎯 Person Tracking Demonstration")
    print("=" * 50)
    
    # Initialize tracker
    tracker = PersonTracker(
        max_disappeared=5,
        min_hits=2,  # Lower for demo
        iou_threshold=0.3,
        distance_threshold=0.7
    )
    
    print("✅ Tracker initialized")
    
    # Simulate frames with moving people
    scenarios = [
        {
            "frame": 1,
            "description": "Two people appear",
            "detections": [
                create_sample_detection(100, 100, 80, 160),  # Person 1
                create_sample_detection(300, 120, 75, 155),  # Person 2
            ]
        },
        {
            "frame": 2,
            "description": "Both people move slightly",
            "detections": [
                create_sample_detection(110, 105, 80, 160),  # Person 1 moves right
                create_sample_detection(295, 125, 75, 155),  # Person 2 moves left
            ]
        },
        {
            "frame": 3,
            "description": "Person 1 moves more, Person 2 stable",
            "detections": [
                create_sample_detection(125, 110, 80, 160),  # Person 1
                create_sample_detection(298, 128, 75, 155),  # Person 2
            ]
        },
        {
            "frame": 4,
            "description": "Third person appears",
            "detections": [
                create_sample_detection(130, 115, 80, 160),  # Person 1
                create_sample_detection(300, 130, 75, 155),  # Person 2
                create_sample_detection(500, 90, 70, 150),   # Person 3 (new)
            ]
        },
        {
            "frame": 5,
            "description": "Person 2 temporarily disappears",
            "detections": [
                create_sample_detection(135, 120, 80, 160),  # Person 1
                create_sample_detection(510, 95, 70, 150),   # Person 3
            ]
        },
        {
            "frame": 6,
            "description": "Person 2 reappears",
            "detections": [
                create_sample_detection(140, 125, 80, 160),  # Person 1
                create_sample_detection(305, 135, 75, 155),  # Person 2 (back)
                create_sample_detection(520, 100, 70, 150),  # Person 3
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📹 Frame {scenario['frame']}: {scenario['description']}")
        
        # Update tracker
        tracked_detections = tracker.update(scenario['detections'])
        
        # Show results
        print(f"   Input detections: {len(scenario['detections'])}")
        print(f"   Tracked detections: {len(tracked_detections)}")
        print(f"   {tracker.get_track_info()}")
        
        # Show person IDs
        for det in tracked_detections:
            x1, y1, x2, y2 = det.bbox
            print(f"   👤 Person ID {det.person_id}: bbox=({x1},{y1},{x2},{y2}), conf={det.confidence:.2f}")
        
        time.sleep(0.5)  # Simulate frame processing time
    
    print(f"\n🎉 Tracking demonstration complete!")
    print(f"📊 Final stats: {tracker.get_track_info()}")
    print(f"🏃 Active tracks: {tracker.get_active_track_count()}")


if __name__ == "__main__":
    simulate_tracking_scenario()