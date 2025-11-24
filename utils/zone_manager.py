"""
Zone Management Module
Handles zones with counting for all object types and violations for ALL objects
"""

import cv2
import numpy as np
from shapely.geometry import Point, Polygon


class Zone:
    """Represents a surveillance zone with multi-object tracking"""
    
    def __init__(self, name, points, zone_type='restricted', color=(0, 0, 255), restricted_objects=None):
        """
        Initialize a zone
        
        Args:
            name: Zone identifier
            points: List of (x, y) tuples defining polygon corners
            zone_type: 'restricted' or 'counting'
            color: BGR color for visualization
            restricted_objects: List of object types that trigger violations (None = all objects)
        """
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.zone_type = zone_type
        self.color = color
        self.polygon = Polygon(points)
        self.violations = []
        self.object_count = 0
        self.object_ids_in_zone = set()
        self.objects_by_type = {}
        # If None, all objects trigger violations
        self.restricted_objects = restricted_objects
    
    def contains_point(self, point):
        """Check if a point is inside the zone"""
        p = Point(point)
        return self.polygon.contains(p)
    
    def check_violation(self, detection):
        """
        Check if a detection violates this zone (for restricted zones)
        
        Args:
            detection: Detection dict with 'center' key
            
        Returns:
            bool: True if violation detected
        """
        if self.zone_type == 'restricted':
            center = detection['center']
            class_name = detection['class_name']
            
            # Check if object is in zone
            if self.contains_point(center):
                # Trigger violation for all objects if restricted_objects is None
                if self.restricted_objects is None or class_name in self.restricted_objects:
                    self.violations.append(detection)
                    return True
        return False
    
    def count_objects(self, detections):
        """
        Count all objects in this zone by type
        
        Args:
            detections: List of all detections
            
        Returns:
            count: Total number of objects in zone
        """
        count = 0
        current_ids = set()
        objects_by_type = {}
        
        for i, det in enumerate(detections):
            center = det['center']
            if self.contains_point(center):
                count += 1
                current_ids.add(i)
                
                # Track by object type
                class_name = det['class_name']
                objects_by_type[class_name] = objects_by_type.get(class_name, 0) + 1
        
        self.object_count = count
        self.object_ids_in_zone = current_ids
        self.objects_by_type = objects_by_type
        return count
    
    def draw(self, frame, alpha=0.3):
        """
        Draw zone on frame with transparency
        
        Args:
            frame: Input image
            alpha: Transparency level (0-1)
            
        Returns:
            frame: Frame with zone drawn
        """
        overlay = frame.copy()
        
        # Draw filled polygon
        cv2.fillPoly(overlay, [self.points], self.color)
        
        # Blend with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw zone border
        cv2.polylines(frame, [self.points], True, self.color, 3)
        
        # Draw zone label
        center_x = int(np.mean(self.points[:, 0]))
        center_y = int(np.mean(self.points[:, 1]))
        
        if self.zone_type == 'restricted':
            label = f"{self.name}"
        else:
            label = f"{self.name}: {self.object_count}"
        
        # Background for text
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            frame,
            (center_x - label_size[0]//2 - 5, center_y - label_size[1] - 5),
            (center_x + label_size[0]//2 + 5, center_y + 5),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (center_x - label_size[0]//2, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return frame


class ZoneManager:
    """Manages multiple zones and performs analysis"""
    
    def __init__(self):
        self.zones = []
        self.total_violations = 0
        self.alerts = []
    
    def add_zone(self, zone):
        """Add a zone to monitor"""
        self.zones.append(zone)
    
    def create_default_zones(self, frame_width, frame_height):
        """
        Create default zones - restricted area positioned lower
        Restricted zone triggers on ALL objects
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        # Restricted zone (red) - lower left area
        # restricted_objects=None means ALL objects trigger violations
        restricted_zone = Zone(
            name="Restricted Area",
            points=[
                (50, int(frame_height * 0.5)),  # Start at 50% height
                (int(frame_width * 0.25), int(frame_height * 0.5)),
                (int(frame_width * 0.25), int(frame_height * 0.85)),
                (50, int(frame_height * 0.85))
            ],
            zone_type='restricted',
            color=(0, 0, 255),  # Red
            restricted_objects=None  # None = ALL objects trigger violations
        )
        self.add_zone(restricted_zone)
        
        # Counting zone 1 (green) - center area
        counting_zone_1 = Zone(
            name="Main Area",
            points=[
                (frame_width // 3, frame_height // 4),
                (2 * frame_width // 3, frame_height // 4),
                (2 * frame_width // 3, 3 * frame_height // 4),
                (frame_width // 3, 3 * frame_height // 4)
            ],
            zone_type='counting',
            color=(0, 255, 0)  # Green
        )
        self.add_zone(counting_zone_1)
        
        # Counting zone 2 (blue) - right side
        counting_zone_2 = Zone(
            name="Exit Zone",
            points=[
                (2 * frame_width // 3 + 30, frame_height // 5),
                (frame_width - 50, frame_height // 5),
                (frame_width - 50, 4 * frame_height // 5),
                (2 * frame_width // 3 + 30, 4 * frame_height // 5)
            ],
            zone_type='counting',
            color=(255, 0, 0)  # Blue
        )
        self.add_zone(counting_zone_2)
    
    def analyze_frame(self, detections):
        """
        Analyze all detections against all zones
        checks ALL objects for violations
        
        Args:
            detections: List of all detections
            
        Returns:
            analysis: Dict with violation and counting info
        """
        self.alerts = []
        frame_violations = 0
        violation_details = []
        
        for zone in self.zones:
            if zone.zone_type == 'restricted':
                # Check ALL objects for violations
                for det in detections:
                    if zone.check_violation(det):
                        frame_violations += 1
                        object_type = det['class_name'].title()
                        alert = f"⚠️ {object_type} detected in {zone.name}!"
                        if alert not in self.alerts:
                            self.alerts.append(alert)
                            violation_details.append({
                                'object': object_type,
                                'zone': zone.name,
                                'bbox': det['bbox']
                            })
            
            elif zone.zone_type == 'counting':
                # Count ALL objects in zone
                zone.count_objects(detections)
        
        self.total_violations += frame_violations
        
        # Prepare analysis summary with object counts
        zone_counts = {}
        zone_details = {}
        
        for zone in self.zones:
            if zone.zone_type == 'counting':
                zone_counts[zone.name] = zone.object_count
                zone_details[zone.name] = zone.objects_by_type
        
        analysis = {
            'violations_in_frame': frame_violations,
            'total_violations': self.total_violations,
            'alerts': self.alerts,
            'violation_details': violation_details,
            'zone_counts': zone_counts,
            'zone_details': zone_details
        }
        
        return analysis
    
    def draw_all_zones(self, frame):
        """Draw all zones on frame"""
        for zone in self.zones:
            frame = zone.draw(frame)
        return frame
    
    def reset_statistics(self):
        """Reset all counters and violations"""
        self.total_violations = 0
        self.alerts = []
        for zone in self.zones:
            zone.violations = []
            zone.object_count = 0
            zone.object_ids_in_zone = set()
            zone.objects_by_type = {}