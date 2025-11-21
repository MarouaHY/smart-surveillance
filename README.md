# 🏢 Smart Surveillance System for Commercial Centers

**Advanced Object Detection & Behavior Analysis using YOLOv11**

A comprehensive surveillance system designed specifically for commercial centers (shopping malls, retail spaces) that performs real-time object detection, monitors restricted zones, counts people and vehicles,  and generates detailed analytics reports. Built for the Digitup Company technical assessment.

---

##  Features

1. **Real-time Object Detection**
   - YOLOv11-based detection (Nano, Small, Medium models)
   - Detects people, vehicles, and other objects
   - Displays bounding boxes with labels and confidence scores
   - Optimized for real-time performance

2. **Behavior Analysis**
   - Configurable restricted zones with violation alerts
   - Multiple counting zones for occupancy monitoring
   - Automatic person counting in each zone
   - Real-time alert system for unauthorized access

3. **Deployable Application**
   - Clean Streamlit web interface
   - Upload video files (MP4, AVI, MOV, MKV)
   - Live webcam support with real-time processing
   - Visual display of zones and detection results


### Bonus Features

- **Export Analytics**: CSV and JSON export for all detection data
- **Customizable Zones**: Easy-to-modify zone definitions
- **Real-time Statistics**: Live metrics and counters
- **Multi-model Support**: Switch between YOLOv11n/s/m
- **Professional UI**: Clean, modern Streamlit interface

---

## Requirements

- Python 3.8 - 3.11
- Webcam (for live detection)
- 4GB+ RAM recommended
- GPU optional (CUDA-compatible for faster processing)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/smart-surveillance.git
cd smart-surveillance
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate


```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: YOLOv11 weights will download automatically on first run (~6MB for nano model).

---

##  Usage

### Running the Application

```bash
streamlit run app.py
```

### Using the Interface

#### **Upload Video Mode**
1. Select "Upload Video" option
2. Choose your video file
3. Configure detection settings in sidebar:
   - Select model (Nano recommended for speed)
   - Toggle bounding boxes and zones
4. Click "Start Processing"
5. View real-time results and statistics
6. Export analytics to CSV or JSON

#### **Webcam Mode**
1. Select "Webcam" option
2. Grant camera permissions if prompted
3. Click "Start Webcam"
4. Monitor live feed with detections
5. Click "Stop Webcam" to end session

---

##  Technical Details

### Architecture Overview

```
Input (Video/Webcam)
    ↓
ObjectDetector (YOLOv11)
    ↓
ZoneManager (Behavior Analysis)
    ↓
Analytics (Statistics & Export)
    ↓
Streamlit UI (Visualization)
```

### Key Components

#### 1. **ObjectDetector** (`utils/detector.py`)
- Loads YOLOv11 models via Ultralytics
- Performs object detection on frames
- Filters and processes detection results
- Draws bounding boxes and labels

#### 2. **ZoneManager** (`utils/zone_manager.py`)
- Defines restricted and counting zones
- Checks for zone violations
- Counts people in specific areas
- Generates alerts for unauthorized access

#### 3. **SurveillanceAnalytics** (`utils/analytics.py`)
- Tracks detections over time
- Calculates statistics and metrics
- Exports data to CSV/JSON formats
- Provides real-time summaries

#### 4. **Streamlit App** (`app.py`)
- User interface and interaction
- Video/webcam processing pipeline
- Real-time visualization
- Export functionality
---
---

---

## Limitations & Future Improvements

### Current Limitations

- Zones are static 
- No persistent object tracking
- Single camera only 
- No annotated video recording
- Local processing only 

### Planned Improvements

- Interactive zone drawing tool
- Object tracking with unique IDs
- Heatmaps & behavior recognition
- Multi-camera support
- Cloud integration (AWS/Azure)
- Advanced behavior recognition (falling, fighting, etc.)
- Email/SMS alerts
- Video recording with annotations




##  Resources

- **YOLOv11 Documentation**: https://docs.ultralytics.com/
- **OpenCV Docs**: https://docs.opencv.org/
- **Streamlit Docs**: https://docs.streamlit.io/
- **COCO Dataset Classes**: https://cocodataset.org/

---

##  Author

**Maroua Hayane**  
Candidate for AI Engineer position at Digitup Company

---

##  License

This project is created for the Digitup Company technical assessment.


##  Acknowledgments

- Ultralytics for YOLOv11
- Streamlit for the amazing UI framework
- OpenCV community
- COCO dataset contributors

