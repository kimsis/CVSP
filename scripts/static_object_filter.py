"""
Static Object Filter for Person Tracking
Filters out static false positives based on movement analysis.
"""

import numpy as np
from collections import defaultdict


class StaticObjectFilter:
    def __init__(self, frames_threshold=30, movement_threshold=10):
        """
        Filter out static false positives based on movement.
        
        Args:
            frames_threshold: Number of frames to consider for static detection
            movement_threshold: Minimum pixel movement to consider object as moving
        """
        self.frames_threshold = frames_threshold
        self.movement_threshold = movement_threshold
        
        # Track position history for each ID
        self.position_history = defaultdict(list)
        
        # Track if an ID is marked as static
        self.static_objects = set()
    
    def get_box_center(self, box):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_movement(self, positions):
        """Calculate total movement from position history"""
        if len(positions) < 2:
            return float('inf')  # Not enough data, assume moving
        
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_movement += np.sqrt(dx**2 + dy**2)
        
        # Average movement per frame
        avg_movement = total_movement / (len(positions) - 1)
        return avg_movement
    
    def update(self, track_id, box):
        """
        Update tracking history and determine if object is static.
        
        Args:
            track_id: Unique identifier for the tracked object
            box: Bounding box coordinates [x1, y1, x2, y2]
        
        Returns:
            bool: True if object should be kept (moving), False if static
        """
        center = self.get_box_center(box)
        self.position_history[track_id].append(center)
        
        # Keep only recent history
        if len(self.position_history[track_id]) > self.frames_threshold:
            self.position_history[track_id].pop(0)
        
        # Check if we have enough frames to analyze
        if len(self.position_history[track_id]) >= self.frames_threshold:
            movement = self.calculate_movement(self.position_history[track_id])
            
            if movement < self.movement_threshold:
                self.static_objects.add(track_id)
                return False  # Filter out static object
            else:
                # Object is moving, remove from static set if it was there
                self.static_objects.discard(track_id)
                return True
        
        # Not enough data yet, keep the detection
        return True
    
    def is_static(self, track_id):
        """
        Check if a track ID is marked as static.
        
        Args:
            track_id: Unique identifier for the tracked object
        
        Returns:
            bool: True if object is static, False otherwise
        """
        return track_id in self.static_objects
    
    def cleanup_old_tracks(self, active_ids):
        """
        Remove tracking data for IDs that are no longer active.
        
        Args:
            active_ids: List of currently active track IDs
        """
        all_ids = set(self.position_history.keys())
        inactive_ids = all_ids - set(active_ids)
        
        for track_id in inactive_ids:
            del self.position_history[track_id]
            self.static_objects.discard(track_id)
    
    def get_stats(self):
        """
        Get statistics about tracked objects.
        
        Returns:
            dict: Statistics including number of tracked objects and static objects
        """
        return {
            'total_tracked': len(self.position_history),
            'static_count': len(self.static_objects),
            'moving_count': len(self.position_history) - len(self.static_objects)
        }