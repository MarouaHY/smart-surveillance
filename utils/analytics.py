"""
Analytics Module 
Handles statistics, export, and reporting for all object types detected in surveillance.
"""

import pandas as pd
import json
from datetime import datetime
import os
from collections import defaultdict


class SurveillanceAnalytics:
    """Tracks and exports surveillance analytics"""
    
    def __init__(self):
        self.frame_data = []
        self.session_start = datetime.now()
        self.total_frames_processed = 0
        self.total_detections = 0
        
        # Track all detected classes
        self.detection_history = defaultdict(int)
        
        # Track by category
        self.category_history = defaultdict(int)
        
        # Peak counts
        self.peak_people = 0
        self.peak_vehicles = 0
        self.peak_objects = 0
        
        # Time-based analytics
        self.hourly_stats = defaultdict(lambda: {'people': 0, 'vehicles': 0, 'objects': 0})
    
    def update(self, detections, zone_analysis, fps=None):
        """
        Update analytics with new frame data
        
        Args:
            detections: List of detections in current frame
            zone_analysis: Zone analysis results
            fps: Current processing FPS
        """
        self.total_frames_processed += 1
        self.total_detections += len(detections)
        
        # Count detections by class and category
        frame_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for det in detections:
            class_name = det['class_name']
            category = det.get('category', 'other')
            
            frame_counts[class_name] += 1
            category_counts[category] += 1
            
            self.detection_history[class_name] += 1
            self.category_history[category] += 1
        
        # Update peak counts
        people_count = frame_counts.get('person', 0)
        vehicle_count = sum(frame_counts.get(v, 0) for v in ['car', 'truck', 'bus', 'motorcycle', 'bicycle'])
        
        self.peak_people = max(self.peak_people, people_count)
        self.peak_vehicles = max(self.peak_vehicles, vehicle_count)
        self.peak_objects = max(self.peak_objects, len(detections))
        
        # Store frame data
        frame_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'frame_number': self.total_frames_processed,
            'total_detections': len(detections),
            'people_count': people_count,
            'vehicle_count': vehicle_count,
            'violations': zone_analysis.get('violations_in_frame', 0),
            'fps': fps if fps else 0
        }
        
        # Add all detected classes
        for class_name, count in frame_counts.items():
            col_name = f"count_{class_name.replace(' ', '_')}"
            frame_record[col_name] = count
        
        # Add category counts
        for category, count in category_counts.items():
            frame_record[f"category_{category}"] = count
        
        # Add zone counts
        for zone_name, count in zone_analysis.get('zone_counts', {}).items():
            safe_zone_name = zone_name.replace(' ', '_').lower()
            frame_record[f'zone_{safe_zone_name}'] = count
        
        self.frame_data.append(frame_record)
        
        # Update hourly stats
        current_hour = datetime.now().strftime('%Y-%m-%d %H:00')
        self.hourly_stats[current_hour]['people'] += people_count
        self.hourly_stats[current_hour]['vehicles'] += vehicle_count
        self.hourly_stats[current_hour]['objects'] += len(detections)
    
    def get_summary_statistics(self):
        """
        Generate comprehensive summary statistics
        
        Returns:
            dict: Summary statistics
        """
        runtime = (datetime.now() - self.session_start).total_seconds()
        avg_fps = self.total_frames_processed / runtime if runtime > 0 else 0
        
        # Calculate averages
        avg_people = sum(f.get('people_count', 0) for f in self.frame_data) / len(self.frame_data) if self.frame_data else 0
        avg_vehicles = sum(f.get('vehicle_count', 0) for f in self.frame_data) / len(self.frame_data) if self.frame_data else 0
        
        summary = {
            'session_info': {
                'start_time': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
                'runtime_seconds': round(runtime, 2),
                'runtime_formatted': self._format_duration(runtime)
            },
            'processing_stats': {
                'total_frames': self.total_frames_processed,
                'average_fps': round(avg_fps, 2),
                'total_detections': self.total_detections,
                'avg_detections_per_frame': round(
                    self.total_detections / self.total_frames_processed, 2
                ) if self.total_frames_processed > 0 else 0
            },
            'people_analytics': {
                'total_detected': self.detection_history.get('person', 0),
                'average_per_frame': round(avg_people, 2),
                'peak_count': self.peak_people
            },
            'vehicle_analytics': {
                'total_detected': sum(self.detection_history.get(v, 0) for v in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']),
                'average_per_frame': round(avg_vehicles, 2),
                'peak_count': self.peak_vehicles,
                'by_type': {
                    'cars': self.detection_history.get('car', 0),
                    'trucks': self.detection_history.get('truck', 0),
                    'buses': self.detection_history.get('bus', 0),
                    'motorcycles': self.detection_history.get('motorcycle', 0),
                    'bicycles': self.detection_history.get('bicycle', 0)
                }
            },
            'object_breakdown': {
                'by_class': dict(sorted(self.detection_history.items(), key=lambda x: x[1], reverse=True)),
                'by_category': dict(sorted(self.category_history.items(), key=lambda x: x[1], reverse=True))
            },
            'security_metrics': {
                'total_violations': sum(f.get('violations', 0) for f in self.frame_data),
                'frames_with_violations': sum(1 for f in self.frame_data if f.get('violations', 0) > 0)
            }
        }
        
        return summary
    
    def _format_duration(self, seconds):
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def get_dataframe(self):
        """
        Convert frame data to pandas DataFrame
        
        Returns:
            DataFrame: Analytics data
        """
        if not self.frame_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.frame_data)
        return df
    
    def get_class_summary(self):
        """
        Get summary of all detected classes
        
        Returns:
            DataFrame: Summary by class
        """
        if not self.detection_history:
            return pd.DataFrame()
        
        data = [
            {'class': k, 'total_detections': v, 'percentage': round(v/self.total_detections*100, 2)}
            for k, v in sorted(self.detection_history.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return pd.DataFrame(data)
    
    def get_category_summary(self):
        """
        Get summary by category
        
        Returns:
            DataFrame: Summary by category
        """
        if not self.category_history:
            return pd.DataFrame()
        
        data = [
            {'category': k, 'total_detections': v, 'percentage': round(v/self.total_detections*100, 2)}
            for k, v in sorted(self.category_history.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return pd.DataFrame(data)
    
    def export_to_csv(self, output_dir='results'):
        """
        Export analytics data to Csv files
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            dict: Paths to created files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}
        
        # Frame-by-frame data
        df = self.get_dataframe()
        if not df.empty:
            path = os.path.join(output_dir, f'frame_data_{timestamp}.csv')
            df.to_csv(path, index=False)
            files['frame_data'] = path
        
        # Class summary
        class_df = self.get_class_summary()
        if not class_df.empty:
            path = os.path.join(output_dir, f'class_summary_{timestamp}.csv')
            class_df.to_csv(path, index=False)
            files['class_summary'] = path
        
        # Category summary
        cat_df = self.get_category_summary()
        if not cat_df.empty:
            path = os.path.join(output_dir, f'category_summary_{timestamp}.csv')
            cat_df.to_csv(path, index=False)
            files['category_summary'] = path
        
        return files
    
    def export_to_json(self, output_path='results/analytics.json'):
        """
        Export full analytics report to JSON 
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            str: Path to created file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            'summary': self.get_summary_statistics(),
            'frame_data': self.frame_data[-100:], 
            'class_breakdown': dict(self.detection_history),
            'category_breakdown': dict(self.category_history),
            'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def reset(self):
        """Reset all analytics data"""
        self.frame_data = []
        self.session_start = datetime.now()
        self.total_frames_processed = 0
        self.total_detections = 0
        self.detection_history = defaultdict(int)
        self.category_history = defaultdict(int)
        self.peak_people = 0
        self.peak_vehicles = 0
        self.peak_objects = 0
        self.hourly_stats = defaultdict(lambda: {'people': 0, 'vehicles': 0, 'objects': 0})
    
    def get_real_time_stats(self):
        """
        Get real-time statistics for display
        
        Returns:
            dict: Current statistics
        """
        if not self.frame_data:
            return {
                'frames_processed': 0,
                'total_detections': 0,
                'detection_breakdown': {}
            }
        
        recent_frames = self.frame_data[-30:]  # Last 30 frames
        
        # Calculate averages from recent frames
        avg_people = sum(f.get('people_count', 0) for f in recent_frames) / len(recent_frames)
        avg_vehicles = sum(f.get('vehicle_count', 0) for f in recent_frames) / len(recent_frames)
        
        return {
            'frames_processed': self.total_frames_processed,
            'total_detections': self.total_detections,
            'avg_people_last_30': round(avg_people, 1),
            'avg_vehicles_last_30': round(avg_vehicles, 1),
            'peak_people': self.peak_people,
            'peak_vehicles': self.peak_vehicles,
            'detection_breakdown': dict(self.detection_history)
        }