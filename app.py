import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO

# Streamlit setup
st.set_page_config(page_title="Smart Parking Detector", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .metric-box {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #ffffff;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label {
        color: #555;
        font-size: 16px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='color:#0d6efd;'>🚘 Smart Parking Detector (High Accuracy)</h1>", unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Define videos
video_options = {
    "Easy 1": r"G:\Car-Parking-Detection\video.mp4\easy1.mp4",
    "Easy 2": r"G:\Car-Parking-Detection\video.mp4\easy2.mp4",
    "Easy 3": r"G:\Car-Parking-Detection\video.mp4\easy3.mp4"
}

# Select video
selected_video_label = st.selectbox("Select a predefined video", list(video_options.keys()))
use_upload = st.checkbox("📤 Or upload your own video instead")

cap = None
temp_file_path = None

# Prepare parking_slots directory
slot_dir = "parking_slots"
os.makedirs(slot_dir, exist_ok=True)

# Get slot file name based on selected video
if not use_upload:
    video_path = video_options[selected_video_label]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    slot_file = os.path.join(slot_dir, f"parking_slots_{video_name}.npy")
else:
    slot_file = None  # For uploaded video, no pre-saved slots by default

# Load slots for selected video
if slot_file and os.path.exists(slot_file):
    slot_list = np.load(slot_file, allow_pickle=True)
    st.success(f"📂 Loaded slots from {slot_file}")
else:
    if not use_upload:
        st.warning(f"🆕 No saved slots found for {selected_video_label}. Please draw and save slots first.")
    slot_list = []

# Handle video input
if use_upload:
    uploaded_video = st.file_uploader("Upload a parking lot video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        temp_file_path = temp_file.name
        cap = cv2.VideoCapture(temp_file_path)
else:
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        st.error(f"🚫 File not found: {video_path}")
        cap = None

if cap:
    st.markdown("---")
    col1, col2 = st.columns((2, 1))
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25, iou=0.45, verbose=False)[0]
        cars = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cars.append((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        occupied, vacant = 0, 0
        for i, polygon in enumerate(slot_list):
            poly_np = np.array(polygon, np.int32)
            status = "Vacant"
            for cx, cy in cars:
                if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                    status = "Occupied"
                    break

            color = (0, 0, 255) if status == "Occupied" else (0, 255, 0)
            cv2.polylines(frame, [poly_np], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, f"{i+1}: {status}", polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if status == "Occupied":
                occupied += 1
            else:
                vacant += 1

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        stats_html = f"""
        <h3>📊 Live Slot Status</h3>
        <div class='metric-box'>
            <div class='metric-label'>🅿️ Total Slots</div>
            <div class='metric-value'>{len(slot_list)}</div>
        </div>
        <div class='metric-box'>
            <div class='metric-label'>✅ Vacant Slots</div>
            <div class='metric-value' style='color:green;'>{vacant}</div>
        </div>
        <div class='metric-box'>
            <div class='metric-label'>❌ Occupied Slots</div>
            <div class='metric-value' style='color:red;'>{occupied}</div>
        </div>
        """
        stats_placeholder.markdown(stats_html, unsafe_allow_html=True)

    cap.release()
    if temp_file_path:
        os.remove(temp_file_path)
