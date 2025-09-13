"""
Person tracking system for multi-person pose estimation
Maintains consistent person IDs across video frames using multi-metric matching
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import cv2
from scipy.optimize import linear_sum_assignment
from detectors import DetectionResult


@dataclass
class Track:
    """Represents a person track across multiple frames"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]    # (cx, cy)
    keypoints: Optional[np.ndarray] = None
    confidence: float = 0.0
    age: int = 0  # Number of frames since track started
    time_since_update: int = 0  # Frames since last detection match
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    hits: int = 1  # Number of times this track was matched
    hit_streak: int = 1  # Consecutive matches
    
    def update(self, detection: DetectionResult):
        """Update track with new detection"""
        new_centroid = self._bbox_to_centroid(detection.bbox)
        
        # Calculate velocity
        if self.time_since_update == 0:  # Only if track was active last frame
            self.velocity = (
                new_centroid[0] - self.centroid[0],
                new_centroid[1] - self.centroid[1]
            )
        
        # Update track properties
        self.bbox = detection.bbox
        self.centroid = new_centroid
        self.keypoints = detection.keypoints
        self.confidence = detection.confidence
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
    def predict(self):
        """Predict next position based on velocity"""
        if self.velocity != (0.0, 0.0):
            predicted_centroid = (
                self.centroid[0] + self.velocity[0],
                self.centroid[1] + self.velocity[1]
            )
            # Update bbox based on predicted centroid
            bbox_w = self.bbox[2] - self.bbox[0]
            bbox_h = self.bbox[3] - self.bbox[1]
            self.bbox = (
                int(predicted_centroid[0] - bbox_w/2),
                int(predicted_centroid[1] - bbox_h/2),
                int(predicted_centroid[0] + bbox_w/2),
                int(predicted_centroid[1] + bbox_h/2)
            )
            self.centroid = predicted_centroid
    
    def mark_missed(self):
        """Mark track as missed this frame"""
        self.time_since_update += 1
        self.hit_streak = 0
        self.age += 1
    
    @staticmethod
    def _bbox_to_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Convert bounding box to centroid"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PersonTracker:
    """
    Multi-person tracking system using multiple distance metrics
    """
    
    def __init__(self, 
                 max_disappeared: int = 10,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 distance_threshold: float = 0.7):
        """
        Args:
            max_disappeared: Max frames a track can be missing before deletion
            min_hits: Minimum hits required to confirm a track
            iou_threshold: Minimum IoU for bbox matching
            distance_threshold: Maximum distance for track matching
        """
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.frame_count = 0
        
    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Update tracker with new detections and return tracked detections
        """
        self.frame_count += 1
        
        # Convert detections to tracks format for easier processing
        if not detections:
            # Mark all tracks as missed
            for track in self.tracks.values():
                track.mark_missed()
            self._cleanup_tracks()
            return []
        
        # Predict current positions for existing tracks
        for track in self.tracks.values():
            track.predict()
        
        # Match detections to existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, list(self.tracks.values())
        )
        
        # Update matched tracks
        for detection_idx, track_idx in matched_pairs:
            track_id = list(self.tracks.keys())[track_idx]
            self.tracks[track_id].update(detections[detection_idx])
            detections[detection_idx].person_id = track_id
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = Track(
                id=self.next_id,
                bbox=detections[detection_idx].bbox,
                centroid=Track._bbox_to_centroid(detections[detection_idx].bbox),
                keypoints=detections[detection_idx].keypoints,
                confidence=detections[detection_idx].confidence
            )
            self.tracks[self.next_id] = new_track
            detections[detection_idx].person_id = self.next_id
            self.next_id += 1
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track_id = list(self.tracks.keys())[track_idx]
            self.tracks[track_id].mark_missed()
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        # Return only detections from confirmed tracks
        tracked_detections = []
        for detection in detections:
            track_id = detection.person_id
            if track_id in self.tracks and self.tracks[track_id].hits >= self.min_hits:
                tracked_detections.append(detection)
        
        return tracked_detections
    
    def _associate_detections_to_tracks(self, detections: List[DetectionResult], tracks: List[Track]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm
        Returns: (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Build cost matrix
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, detection in enumerate(detections):
            for t, track in enumerate(tracks):
                cost = self._calculate_distance(detection, track)
                cost_matrix[d, t] = cost
        
        # Use Hungarian algorithm for optimal assignment
        detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter assignments that exceed distance threshold
        matched_pairs = []
        for d_idx, t_idx in zip(detection_indices, track_indices):
            if cost_matrix[d_idx, t_idx] <= self.distance_threshold:
                matched_pairs.append((d_idx, t_idx))
        
        # Find unmatched detections and tracks
        matched_detection_indices = set([pair[0] for pair in matched_pairs])
        matched_track_indices = set([pair[1] for pair in matched_pairs])
        
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _calculate_distance(self, detection: DetectionResult, track: Track) -> float:
        """
        Calculate multi-metric distance between detection and track
        Lower values indicate better matches
        """
        # Weight factors for different metrics
        iou_weight = 0.4
        centroid_weight = 0.3
        pose_weight = 0.3
        
        total_distance = 0.0
        
        # 1. IoU distance (bbox overlap)
        iou = self._calculate_iou(detection.bbox, track.bbox)
        iou_distance = 1.0 - iou  # Convert IoU to distance
        total_distance += iou_weight * iou_distance
        
        # 2. Centroid distance (normalized by frame dimensions)
        centroid_distance = self._calculate_centroid_distance(detection, track)
        total_distance += centroid_weight * centroid_distance
        
        # 3. Pose similarity distance
        if detection.keypoints is not None and track.keypoints is not None:
            pose_distance = self._calculate_pose_distance(detection.keypoints, track.keypoints)
            total_distance += pose_weight * pose_distance
        else:
            # If no pose data, rely more on spatial metrics
            total_distance += pose_weight * 0.5  # Neutral penalty
        
        return total_distance
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_centroid_distance(self, detection: DetectionResult, track: Track) -> float:
        """Calculate normalized centroid distance"""
        det_centroid = Track._bbox_to_centroid(detection.bbox)
        
        # Euclidean distance
        distance = np.sqrt(
            (det_centroid[0] - track.centroid[0]) ** 2 + 
            (det_centroid[1] - track.centroid[1]) ** 2
        )
        
        # Normalize by image diagonal (assume 640x480 for now)
        # TODO: Make this configurable based on actual frame size
        frame_diagonal = np.sqrt(640**2 + 480**2)
        normalized_distance = distance / frame_diagonal
        
        return min(normalized_distance, 1.0)  # Cap at 1.0
    
    def _calculate_pose_distance(self, keypoints1: np.ndarray, keypoints2: np.ndarray) -> float:
        """
        Calculate pose similarity distance using Object Keypoint Similarity (OKS)
        """
        if keypoints1.shape != keypoints2.shape:
            return 1.0  # Maximum distance for incompatible poses
        
        # Handle different keypoint formats
        if keypoints1.shape[0] == 17:  # COCO format
            return self._calculate_coco_oks_distance(keypoints1, keypoints2)
        elif keypoints1.shape[0] == 33:  # MediaPipe format
            return self._calculate_mediapipe_pose_distance(keypoints1, keypoints2)
        else:
            return 1.0  # Unknown format
    
    def _calculate_coco_oks_distance(self, kpts1: np.ndarray, kpts2: np.ndarray) -> float:
        """Calculate OKS-based distance for COCO 17-point poses"""
        # COCO keypoint sigmas (from COCO dataset)
        sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035,  # head
            0.079, 0.079, 0.072, 0.072, 0.062, 0.062,  # arms
            0.107, 0.107, 0.087, 0.087, 0.089, 0.089   # legs
        ])
        
        # Calculate distances for visible keypoints
        distances = []
        for i in range(17):
            if kpts1[i, 2] > 0.5 and kpts2[i, 2] > 0.5:  # Both keypoints visible
                dx = kpts1[i, 0] - kpts2[i, 0]
                dy = kpts1[i, 1] - kpts2[i, 1]
                d = np.sqrt(dx*dx + dy*dy)
                
                # Normalize by keypoint uncertainty (sigma) and object scale
                # Estimate object scale from bounding box area
                object_scale = 100  # Approximate scale factor
                normalized_d = d / (2 * sigmas[i] * object_scale)
                distances.append(np.exp(-normalized_d*normalized_d))
        
        if not distances:
            return 1.0  # No visible keypoints
        
        oks = np.mean(distances)
        return 1.0 - oks  # Convert OKS to distance
    
    def _calculate_mediapipe_pose_distance(self, kpts1: np.ndarray, kpts2: np.ndarray) -> float:
        """Calculate cosine similarity-based distance for MediaPipe 33-point poses"""
        # Filter visible keypoints
        visible_mask = (kpts1[:, 2] > 0.5) & (kpts2[:, 2] > 0.5)
        
        if not np.any(visible_mask):
            return 1.0  # No visible keypoints
        
        # Extract visible keypoint coordinates
        visible_kpts1 = kpts1[visible_mask, :2].flatten()
        visible_kpts2 = kpts2[visible_mask, :2].flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(visible_kpts1, visible_kpts2)
        norm1 = np.linalg.norm(visible_kpts1)
        norm2 = np.linalg.norm(visible_kpts2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_distance = 1.0 - cosine_similarity
        
        return max(0.0, min(1.0, cosine_distance))  # Clamp to [0, 1]
    
    def _cleanup_tracks(self):
        """Remove old tracks that haven't been updated recently"""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
    
    def get_active_track_count(self) -> int:
        """Get number of currently active tracks"""
        return len([track for track in self.tracks.values() 
                   if track.time_since_update == 0 and track.hits >= self.min_hits])
    
    def get_track_info(self) -> str:
        """Get tracking information for display"""
        active_tracks = self.get_active_track_count()
        total_tracks = len(self.tracks)
        return f"Active: {active_tracks}, Total: {total_tracks}, Frame: {self.frame_count}"