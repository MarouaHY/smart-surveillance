"""
Video Processing Module 
Handles video stream processing for surveillance"""

import cv2
import numpy as np
import time
from pathlib import Path


class VideoProcessor:
    """Handles video stream processing for surveillance"""
    
    def __init__(self, detector, zone_manager, analytics):
        self.detector = detector
        self.zone_manager = zone_manager
        self.analytics = analytics
        self.is_processing = False
        self.current_fps = 0
    
    def process_frame(self, frame, show_boxes=True, show_zones=True):
        """
        Process a single frame:
        - Resize to 640x640 for YOLO
        - Run detection
        - Scale detections back to original frame size
        - Perform zone analysis
        - Update analytics
        - Draw detections and zones if requested
        
        Args:
            frame: Original BGR image
            show_boxes: Whether to draw bounding boxes
            show_zones: Whether to analyze and draw zones
            
        Returns:
            processed_frame: Annotated frame
            stats: Dictionary with detection summary, FPS, processing time
        """

        start_time = time.time()
        
        # Store original frame
        original_frame = frame.copy()
        original_h, original_w = frame.shape[:2]
        
        # Resize frame for YOLO 
        detection_frame = cv2.resize(frame, (640, 640))

        # Scale factors
        scale_x = original_w / 640
        scale_y = original_h / 640
        
        # Run YOLO
        results = self.detector.detect(detection_frame)
        detections = self.detector.process_detections(results, apply_filters=True)
        
        # Scale detections back to original resolution
        scaled_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

            sd = det.copy()
            sd['bbox'] = (x1, y1, x2, y2)
            sd['center'] = ((x1 + x2) // 2, (y1 + y2) // 2)
            scaled_detections.append(sd)
        
        # Detection summary
        detection_summary = self.detector.get_detection_summary(scaled_detections)

    
        # ZONE ANALYSIS 
        if show_zones:
            zone_analysis = self.zone_manager.analyze_frame(scaled_detections)
        else:
            zone_analysis = {
                "zone_counts": {},
                "alerts": [],
                "violations_in_frame": 0
            }
        
        # Update analytics 
        self.analytics.update(scaled_detections, zone_analysis, self.current_fps)
        
        # Prepare output frame to draw on
        processed_frame = original_frame.copy()

        # Draw zones first
        if show_zones:
            processed_frame = self.zone_manager.draw_all_zones(processed_frame)

        # Draw detections boxes
        if show_boxes:
            processed_frame = self.detector.draw_detections(processed_frame, scaled_detections)
        
        # FPS compute 
        process_time = time.time() - start_time
        self.current_fps = 1.0 / process_time if process_time > 0 else 0
        # Collect stats for this frame
        stats = {
            'detections': scaled_detections,
            'detection_summary': detection_summary,
            'zone_analysis': zone_analysis,
            'total_objects': len(scaled_detections),
            'fps': self.current_fps,
            'process_time_ms': process_time * 1000,
            'detection_size': '640x640'
        }
        
        return processed_frame, stats
    

    def process_video_file(self, video_path, output_path=None, max_frames=None, 
                           skip_frames=0, show_boxes=True, show_zones=True):
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video (optional)
            max_frames: Maximum frames to process (optional)
            skip_frames: Number of frames to skip between processing
            show_boxes: Draw detection boxes
            show_zones: Perform zone analysis
            
        Yields:
            Dictionary containing processed frame, stats, frame number, and progress
        """
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create default zones if none exist
        if not self.zone_manager.zones:
            self.zone_manager.create_default_zones(width, height)
        
        # Prepare video writer if output path is specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        self.is_processing = True
        frame_count = 0
        processed_count = 0
        
        try:
            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # ----  (skip 4 frames = process only 1/5 frames) ----
                if skip_frames > 0:
                    if (frame_count - 1) % (skip_frames + 1) != 0:
                        continue
                     
                # Process single frame
                processed_frame, stats = self.process_frame(frame, show_boxes, show_zones)
                
                # Write frame to output if writer is active
                if writer:
                    writer.write(processed_frame)
                
                processed_count += 1
                
                # Yield results for streaming or UI display
                yield {
                    'frame': processed_frame,
                    'stats': stats,
                    'frame_number': frame_count,
                    'total_frames': total_frames,
                    'progress': frame_count / total_frames if total_frames > 0 else 0
                }
                
                # Stop if max_frames is reached
                if max_frames and processed_count >= max_frames:
                    break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            self.is_processing = False

    def process_webcam(self, camera_index=0, resolution=(640, 480),
                       show_boxes=True,skip_frames=4, show_zones=True):
        """
        Process a live webcam stream.
        
        Args:
            camera_index: Index of the webcam
            resolution: Capture resolution (width, height)
            show_boxes: Draw detection boxes
            show_zones: Perform zone analysis
            
        Yields:
            Dictionary with processed frame, stats, and frame number
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam at index {camera_index}")
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Create default zones if none exist
        if not self.zone_manager.zones:
            self.zone_manager.create_default_zones(resolution[0], resolution[1])
        
        self.is_processing = True
        frame_count = 0
        
        try:
            while cap.isOpened() and self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                # ---- SKIP FRAME  ----
                if skip_frames > 0:
                    if (frame_count - 1) % (skip_frames + 1) != 0:
                        continue
                # Process single frame
                processed_frame, stats = self.process_frame(frame, show_boxes, show_zones)
                
                yield {
                    'frame': processed_frame,
                    'stats': stats,
                    'frame_number': frame_count
                }
        
        finally:
            cap.release()
            self.is_processing = False
    
    def stop_processing(self):
        """Stop any ongoing video processing loop."""
        self.is_processing = False
    
    @staticmethod
    def get_video_info(video_path):
        """
        Get basic metadata about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with width, height, fps, total frames, and duration
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        return info



