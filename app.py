"""
Smart Surveillance System for Commercial Centers 
Author: Maroua Hayane for Digitup Company

"""
# ======================================================================================================
#                                      Imports & Setup
# ======================================================================================================
import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
import tempfile
import pandas as pd

# Import custom modules 
from utils.detector import ObjectDetector
from utils.zone_manager import ZoneManager, Zone
from utils.analytics import SurveillanceAnalytics
from utils.video_processor import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================================
#                                       Custom CSS for UI
# ======================================================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.0rem;
        color: #1F77B4;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .small-info {
        background-color: #f7f9fc;
        padding: 0.48rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.88rem;
        border: 1px solid #e6e9ef;
        line-height: 1.2;
        display: inline-block;
    }
    .info-card {
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #e3e8ef;
        font-size: 0.90rem;
        color: #333;
        margin-bottom: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .compact-btns > button {
        padding: .35rem .6rem;
        font-size: 0.92rem;
    }
    .violation-box {
        background: linear-gradient(90deg, #ff3b3b, #b30000);
        color: #fff;
        padding: 8px 12px;
        border-radius: 8px;
        margin-top: 6px;
        font-size: 0.95rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 10px rgba(255,0,0,0.18);
    }
    .zone-stat {
        background-color: #e9ecef;
        padding: 0.4rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.92rem;
    }
    .small-metric {
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ======================================================================================================
#                                      Model Loading
# ======================================================================================================
@st.cache_resource
def load_model(model_name, conf_threshold):
    """Load YOLO model (cached)"""
    detector = ObjectDetector(
        model_name=model_name,
        conf_threshold=conf_threshold,
        device='cpu'
    )
    return detector
# ========================================================================================================
#                                  Session State Initialization
# ========================================================================================================
def initialize_components():
    """Initialize detector, zones, and analytics in session state"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'zone_manager' not in st.session_state:
        st.session_state.zone_manager = ZoneManager()
    if 'analytics' not in st.session_state:
        st.session_state.analytics = SurveillanceAnalytics()
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = None
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False


# ========================================================================================================
#                                         Violation Alerts
# ========================================================================================================
def display_violations_alert(alert_placeholder, stats):
    """Display modern violation alerts under video (only when zones enabled)"""
    if not st.session_state.show_zones:
        return

    alerts = stats.get('zone_analysis', {}).get('alerts', [])
    violations = stats.get('zone_analysis', {}).get('violations_in_frame', 0)

    with alert_placeholder.container():
        if violations > 0:
            st.markdown(
                f'<div class="violation-box">🚨 {violations} Violation(s) Detected!</div>',
                unsafe_allow_html=True,
            )


# ========================================================================================================
#                                Real-time Analytics Display
# ========================================================================================================

def display_real_time_analytics(stats_placeholder, stats, analytics):
    """Display compact real-time analytics in right column"""
    with stats_placeholder.container():
        st.markdown("### 📊 Real-Time Analytics")
        detection_summary = stats.get('detection_summary', {})
        total_objects = stats.get('total_objects', 0)
        fps = stats.get('fps', 0.0)

        # compact metrics
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(f"<div class='small-metric'>🎯 Total Objects: <strong>{total_objects}</strong></div>", unsafe_allow_html=True)
            people = detection_summary.get('by_category', {}).get('people', 0)
            st.markdown(f"<div class='small-metric'>👤 People: <strong>{people}</strong></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='small-metric'>⚡ FPS: <strong>{fps:.1f}</strong></div>", unsafe_allow_html=True)
            vehicles = detection_summary.get('by_category', {}).get('vehicle', 0)
            st.markdown(f"<div class='small-metric'>🚗 Vehicles: <strong>{vehicles}</strong></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Category breakdown 
        st.markdown("**📦 Categories**")
        by_category = detection_summary.get('by_category', {})
        if by_category:
            for cat, cnt in by_category.items():
                if cnt > 0:
                    st.markdown(f"• {cat.title()}: **{cnt}**")

        st.markdown("---")

        # # Object Type breakdown small
        # st.markdown("**🔍 Top Types**")
        # by_class = detection_summary.get('by_class', {})
        # if by_class:
        #     sorted_classes = sorted(by_class.items(), key=lambda x: x[1], reverse=True)[:6]
        #     for cls, count in sorted_classes:
        #         st.markdown(f"• {cls.title()}: **{count}**")

        # st.markdown("---")

        # Zone occupancy 
        if st.session_state.show_zones:
            zone_counts = stats.get('zone_analysis', {}).get('zone_counts', {})
            if zone_counts:
                st.markdown("**🎯 Zone Occupancy**")
                for zone_name, count in zone_counts.items():
                    st.markdown(f'<div class="zone-stat">📍 {zone_name}: <strong>{count}</strong></div>', unsafe_allow_html=True)
# =============================================================================================================
#                                        Video/Webcam Processing
# =============================================================================================================
def process_video_stream(video_processor, video_path=None, camera_index=None, 
                         frame_placeholder=None, stats_placeholder=None,
                         alert_placeholder=None, progress_bar=None, status_text=None):
    """Process video file or webcam stream and update UI placeholders"""
    try:
        if video_path:
            total_frames = VideoProcessor.get_video_info(video_path)['total_frames']

            for frame_data in video_processor.process_video_file(
                video_path,
                show_boxes=st.session_state.show_boxes,
                show_zones=st.session_state.show_zones
            ):
                # If user turned off zones, zone analytics are ignored
                if not st.session_state.show_zones:
                    frame_data['stats']['zone_analysis'] = {"zone_counts": {}, "alerts": [], "violations_in_frame": 0}

                # Update display
                frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update violations alert 
                display_violations_alert(alert_placeholder, frame_data['stats'])

                # Update progress & status
                if progress_bar:
                    progress_bar.progress(frame_data.get('progress', 0))
                if status_text:
                    status_text.text(
                        f"Processing: Frame {frame_data.get('frame_number', 0)}/{total_frames} "
                        f"({frame_data['stats'].get('fps', 0.0):.1f} FPS)"
                    )

                # Update real-time stats
                display_real_time_analytics(stats_placeholder, frame_data['stats'], st.session_state.analytics)

                time.sleep(0.01)

            if status_text:
                status_text.success(f"✅ Video processing complete! Processed {total_frames} frames")

        elif camera_index is not None:
            for frame_data in video_processor.process_webcam(
                camera_index=camera_index,
                show_boxes=st.session_state.show_boxes,
                show_zones=st.session_state.show_zones
            ):
                # Stop condition
                if not st.session_state.get('webcam_active', False):
                    break

                if not st.session_state.show_zones:
                    frame_data['stats']['zone_analysis'] = {"zone_counts": {}, "alerts": [], "violations_in_frame": 0}

                frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                display_violations_alert(alert_placeholder, frame_data['stats'])
                display_real_time_analytics(stats_placeholder, frame_data['stats'], st.session_state.analytics)

                time.sleep(0.01)

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")

# ========================================================================================================
#                                             Main Function
# ========================================================================================================

def main():
    """Main application"""
    initialize_components()

    # Header
    st.markdown('<h1 class="main-header"> Smart Surveillance System</h1>', unsafe_allow_html=True)
    st.markdown("Object Detection & Behavior Analysis for Retail Environments")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_options = {
            'YOLOv11n (Nano - Fastest)': 'yolo11n.pt',
            'YOLOv11s (Small - Balanced)': 'yolo11s.pt',
            'YOLOv11m (Medium - Accurate)': 'yolo11m.pt'
        }
        selected_model = st.selectbox(
            "Select Detection Model",
            options=list(model_options.keys()),
            index=0,
            help="Nano model recommended for real-time performance"
        )
        model_name = model_options[selected_model]

        conf_threshold = st.slider(
            "Base Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Base threshold - individual classes have their own thresholds"
        )

        st.markdown("---")
        st.subheader("🎯 Detection Categories")
        categories = {
            'people': st.checkbox("👤 People", value=True),
            'vehicle': st.checkbox("🚗 Vehicles", value=True),
            'baggage': st.checkbox("🎒 Baggage/Backpacks", value=True),
            'shopping': st.checkbox("🛒 Shopping Items", value=True),
        }
        active_categories = [cat for cat, enabled in categories.items() if enabled]

        st.markdown("---")
        st.subheader("👁️ Display Options")
        st.session_state.show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        # Zones checkbox 
        st.session_state.show_zones = st.checkbox("Show Surveillance Zones ", value=True)
        st.markdown("---")
        st.info("💡 All frames processed at 640x640 for optimal YOLO performance")

    # Load or update model
    if st.session_state.detector is None or st.session_state.get('last_model') != model_name:
        with st.spinner("🔄 Loading YOLO model..."):
            st.session_state.detector = load_model(model_name, conf_threshold)
            st.session_state.detector.set_active_categories(active_categories)
            st.session_state.last_model = model_name
        st.success("✅ Model loaded successfully!")
    else:
        st.session_state.detector.conf_threshold = conf_threshold
        st.session_state.detector.set_active_categories(active_categories)

    # Mode selection
    st.header("📹 Select Input Source")
    mode = st.radio(
        "Choose video source:",
        options=["📁 Upload Video", "📹 Live Webcam"],
        horizontal=True
    )

    # --- ------------------------UPLOAD VIDEO MODE ----------------------------------
    if mode == "📁 Upload Video":
        st.subheader("📁 Upload Video File")
        uploaded_file = st.file_uploader(
            "Choose a video file to analyze",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload surveillance footage from your computer"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            # Get video info
            video_info = VideoProcessor.get_video_info(video_path)

            st.markdown(
                f"""
                <div class="info-card">
                    <strong>📐 Resolution:</strong> {video_info['width']}x{video_info['height']} &nbsp;|&nbsp;
                    <strong>🎞 FPS:</strong> {video_info['fps']} &nbsp;|&nbsp;
                    <strong>⏱ Duration:</strong> {video_info['duration_seconds']}s &nbsp;|&nbsp;
                    <strong>📊 Frames:</strong> {video_info['total_frames']}
                </div>
                """,
                unsafe_allow_html=True
            )

            center_col, right_col = st.columns([6, 2])

            with center_col:
                frame_placeholder = st.empty()
                alert_placeholder = st.empty()  

                with st.container():
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
                    with btn_col1:
                        start_processing = st.button("▶️ Start Processing", key="start_vid")
                    with btn_col2:
                        reset_btn = st.button("🔄 Reset Analytics", key="reset_vid")
                    with btn_col3:
                        export_btn = st.button("💾 Export Results", key="export_vid")

                progress_bar = st.progress(0)
                status_text = st.empty()

            with right_col:
                stats_placeholder = st.empty()

            if st.session_state.show_zones and not st.session_state.zone_manager.zones:
                cap = cv2.VideoCapture(video_path)
                ret, first_frame = cap.read()
                cap.release()
                if ret:
                    st.session_state.zone_manager.create_default_zones(first_frame.shape[1], first_frame.shape[0])

            # Initialize video processor
            st.session_state.video_processor = VideoProcessor(
                st.session_state.detector,
                st.session_state.zone_manager,
                st.session_state.analytics
            )

            # Button actions
            if reset_btn:
                st.session_state.analytics.reset()
                st.success("Analytics reset!")

            if start_processing:
                st.session_state.analytics.reset()
                process_video_stream(
                    st.session_state.video_processor,
                    video_path=video_path,
                    frame_placeholder=frame_placeholder,
                    stats_placeholder=stats_placeholder,
                    alert_placeholder=alert_placeholder,
                    progress_bar=progress_bar,
                    status_text=status_text
                )

            if export_btn:
                files = st.session_state.analytics.export_to_csv()
                st.markdown("---")
                st.subheader("💾 Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    if files:
                        st.success(f"✅ Exported {len(files)} CSV files!")
                        for file_type, path in files.items():
                            with open(path, 'rb') as f:
                                st.download_button(
                                    f"⬇️ {file_type.replace('_', ' ').title()}",
                                    f,
                                    file_name=Path(path).name,
                                    mime="text/csv",
                                    key=f"download_{file_type}"
                                )
                with col2:
                    json_path = st.session_state.analytics.export_to_json()
                    with open(json_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Complete JSON Report",
                            f,
                            file_name="surveillance_report.json",
                            mime="application/json",
                            key="download_json"
                        )

    # --- ----------------------- WEBCAM MODE --------------------------------------
    elif mode == "📹 Live Webcam":
        st.subheader("📹 Live Webcam Surveillance")
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0,
                                      help="Usually 0 for built-in camera")

        cam_info = {"width": 640, "height": 480, "fps": "N/A", "duration_seconds": "N/A", "total_frames": "N/A"}

        st.markdown(
            f"""
            <div class="info-card">
                <strong>📐 Resolution:</strong> {cam_info['width']}x{cam_info['height']} &nbsp;|&nbsp;
                <strong>🎞 FPS:</strong> {cam_info['fps']} &nbsp;|&nbsp;
                <strong>⏱ Duration:</strong> {cam_info['duration_seconds']} &nbsp;|&nbsp;
                <strong>📊 Frames:</strong> {cam_info['total_frames']}
            </div>
            """,
            unsafe_allow_html=True
        )

        center_col, right_col = st.columns([6, 2])

        with center_col:
            frame_placeholder = st.empty()
            alert_placeholder = st.empty()

            with st.container():
                btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
                with btn_col1:
                    start_cam = st.button("▶️ Start Webcam", key="start_cam")
                with btn_col2:
                    stop_cam = st.button("⏹️ Stop Webcam", key="stop_cam")
                with btn_col3:
                    export_webcam = st.button("💾 Export Results", key="export_cam")

            progress_bar = st.progress(0)
            status_text = st.empty()

        with right_col:
            stats_placeholder = st.empty()

        if st.session_state.show_zones and not st.session_state.zone_manager.zones:
            st.session_state.zone_manager.create_default_zones(640, 480)

        if st.session_state.video_processor is None:
            st.session_state.video_processor = VideoProcessor(
                st.session_state.detector,
                st.session_state.zone_manager,
                st.session_state.analytics
            )

        if start_cam:
            st.session_state.webcam_active = True
            st.session_state.analytics.reset()
            st.info("📹 Webcam active - Processing live feed...")
            process_video_stream(
                st.session_state.video_processor,
                camera_index=camera_index,
                frame_placeholder=frame_placeholder,
                stats_placeholder=stats_placeholder,
                alert_placeholder=alert_placeholder,
                progress_bar=progress_bar,
                status_text=status_text
            )

        if stop_cam:
            st.session_state.webcam_active = False
            # If VideoProcessor has stop_processing, call it
            try:
                st.session_state.video_processor.stop_processing()
            except Exception:
                pass
            st.success("Webcam stopped.")

        if export_webcam:
            files = st.session_state.analytics.export_to_csv()
            if files:
                st.success("✅ Results exported!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
             Surveillance System | YOLOv11 + OpenCV + Streamlit<br>
            Developed by <strong>Maroua Hayane</strong> for <strong>Digitup Company</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
