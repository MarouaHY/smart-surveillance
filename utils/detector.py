"""
YOLO Detection Module for Smart Surveillance System
Handles object detection using YOLOv11 with flexible filtering
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path


class ObjectDetector:
    """YOLOv11-based object detector for commercial center surveillance"""
    
    def __init__(self, model_name='yolo11n.pt', conf_threshold=0.5, device='cpu'):
        """
        Initialize the detector
        
        Args:
            model_name: YOLO model to use (yolo11n.pt, yolo11s.pt .....)
            conf_threshold: Confidence threshold for detections
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Ensure models directory exists
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Full path to model
        model_path = models_dir / model_name
        
        # Load YOLO model 
        print(f"Loading {model_name} model from {model_path}...")
        self.model = YOLO(str(model_path))
        
        # Move to GPU if available
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')
            print("✅ Using GPU acceleration")
        else:
            print("ℹ️ Using CPU")
        
        # COCO class names 
        self.class_names = self.model.names
        
        # Commercial center relevant classes with individual thresholds
        self.commercial_classes = {
            # People & Activities
            'person': {'id': 0, 'threshold': 0.5, 'category': 'people', 'color': (0, 255, 0)},
            
            # Personal Items 
            'backpack': {'id': 24, 'threshold': 0.4, 'category': 'baggage', 'color': (255, 165, 0)},
            'handbag': {'id': 26, 'threshold': 0.4, 'category': 'baggage', 'color': (255, 165, 0)},
            'suitcase': {'id': 28, 'threshold': 0.4, 'category': 'baggage', 'color': (255, 165, 0)},
            
            # Vehicles (parking monitoring)
            'bicycle': {'id': 1, 'threshold': 0.5, 'category': 'vehicle', 'color': (255, 0, 0)},
            'car': {'id': 2, 'threshold': 0.5, 'category': 'vehicle', 'color': (255, 0, 0)},
            'motorcycle': {'id': 3, 'threshold': 0.5, 'category': 'vehicle', 'color': (255, 0, 0)},
            'bus': {'id': 5, 'threshold': 0.5, 'category': 'vehicle', 'color': (255, 0, 0)},
            'truck': {'id': 7, 'threshold': 0.5, 'category': 'vehicle', 'color': (255, 0, 0)},
            
            # Shopping carts & trolleys
            'suitcase': {'id': 28, 'threshold': 0.4, 'category': 'shopping', 'color': (0, 255, 255)},
            
            # Safety items
            'umbrella': {'id': 25, 'threshold': 0.3, 'category': 'accessory', 'color': (128, 0, 128)},
            
            # Potential security concerns
            'knife': {'id': 43, 'threshold': 0.6, 'category': 'security', 'color': (0, 0, 255)},
            'scissors': {'id': 76, 'threshold': 0.6, 'category': 'security', 'color': (0, 0, 255)},
            
            # Electronics 
            'cell phone': {'id': 67, 'threshold': 0.4, 'category': 'electronics', 'color': (147, 20, 255)},
            'laptop': {'id': 63, 'threshold': 0.5, 'category': 'electronics', 'color': (147, 20, 255)},
        }
        
        # Active filter categories
        self.active_categories = ['people', 'baggage', 'vehicle', 'shopping', 'electronics']
        # Initialize detection statistics
        self.detection_stats = {}
        self.reset_stats()
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total': 0,
            'by_class': {},
            'by_category': {cat: 0 for cat in ['people', 'baggage', 'vehicle', 'shopping', 'accessory', 'security', 'electronics']}
        }
    
    def set_active_categories(self, categories):
        """
        Set which categories to detect
        
        Args:
            categories: List of category names to enable
        """
        self.active_categories = categories
    
    def set_class_threshold(self, class_name, threshold):
        """
        Set custom threshold for specific class
        
        Args:
            class_name: Name of the class
            threshold: New threshold value (0-1)
        """
        if class_name in self.commercial_classes:
            self.commercial_classes[class_name]['threshold'] = threshold
    
    def detect(self, frame):
        """
        Perform object detection on a single frame
        Uses base confidence threshold, then applies class-specific thresholds
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            results: Detection results with boxes, confidences, and classes
        """
        # Base confidence threshold is low; further filtering is applied later
        results = self.model(frame, conf=0.3, verbose=False)[0]
        return results
    
    def process_detections(self, results, apply_filters=True):
        """
        Extract and organize detection information with filtering
        
        Args:
            results: Raw YOLO results
            apply_filters: Whether to apply class-specific thresholds and category filters
            
        Returns:
            detections: List of dicts with detection info
        """
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                class_name = self.class_names[cls]
                
                # Check if class is in our commercial classes
                class_info = None
                for name, info in self.commercial_classes.items():
                    if info['id'] == cls:
                        class_info = info
                        break
                
                if apply_filters:
                    # Apply class-specific threshold
                    if class_info:
                        if conf < class_info['threshold']:
                            continue
                        # Check if category is active
                        if class_info['category'] not in self.active_categories:
                            continue
                    else:
                        # Unknown class, use base threshold
                        if conf < self.conf_threshold:
                            continue
                
                x1, y1, x2, y2 = map(int, box)
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': class_name,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'category': class_info['category'] if class_info else 'other',
                    'color': class_info['color'] if class_info else (128, 128, 128)
                }
                detections.append(detection)
                
                # Update stats
                self.detection_stats['total'] += 1
                self.detection_stats['by_class'][class_name] = \
                    self.detection_stats['by_class'].get(class_name, 0) + 1
                if class_info:
                    self.detection_stats['by_category'][class_info['category']] += 1
        
        return detections
    
    def get_detection_summary(self, detections):
        """
        Get summary of detections by category
        
        Args:
            detections: List of detections
            
        Returns:
            summary: Dict with counts by category and class
        """
        summary = {
            'total': len(detections),
            'by_category': {},
            'by_class': {}
        }
        
        for det in detections:
            # Count by category
            category = det['category']
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Count by class
            class_name = det['class_name']
            summary['by_class'][class_name] = summary['by_class'].get(class_name, 0) + 1
        
        return summary
    
    def draw_detections(self, frame, detections, show_confidence=True):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image
            detections: List of detections from process_detections()
            show_confidence: Whether to show confidence scores
            
        Returns:
            annotated_frame: Frame with drawn boxes
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            color = det['color']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if show_confidence:
                label = f"{class_name}: {conf:.2f}"
            else:
                label = class_name
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, label_size[1] + 10)
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y + 2),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated_frame
    
    def filter_by_class(self, detections, target_classes):
        """
        Filter detections by specific classes
        
        Args:
            detections: List of all detections
            target_classes: List of class names to keep
            
        Returns:
            filtered_detections: Filtered list
        """
        return [d for d in detections if d['class_name'] in target_classes]
    
    def filter_by_category(self, detections, target_categories):
        """
        Filter detections by categories
        
        Args:
            detections: List of all detections
            target_categories: List of category names to keep
            
        Returns:
            filtered_detections: Filtered list
        """
        return [d for d in detections if d['category'] in target_categories]