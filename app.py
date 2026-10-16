from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent

# 1. Page Configuration
st.set_page_config(
    page_title="PCB AOI Inspection Dashboard",
    page_icon="🔌",
    layout="wide"
)

# 2. Load the Trained YOLO Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "latestmodel.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path))

model = load_model()

# 3. Sidebar Configuration for Controls
st.sidebar.title("AOI Control Panel")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("Upload a high-resolution PCB image to run automated component detection.")

# 4. Main Dashboard Layout
st.title("🔌 Real-Time Edge-AI AOI System for PCB Inspection")
st.markdown("Automated Surface-Mount Component Localization and Quality Control Dashboard.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original PCB Input")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI Inspection Output")
        
        with st.spinner("Running YOLOv8 inference..."):
            # Convert PIL image to numpy array for OpenCV/YOLO
            img_array = np.array(image)
            
          # Ensure the image has 3 channels (RGB) by stripping transparency if present
            image_rgb = image.convert("RGB")
            img_array = np.array(image_rgb)
            
            # Run model prediction
            results = model.predict(
                source=img_array,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=1024
            )
            
            # Extract plotted image with bounding boxes
            res_plotted = results[0].plot() # Returns BGR numpy array
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Display annotated image
            st.image(res_plotted_rgb, use_container_width=True)

    # 5. Summary Statistics Section
    st.markdown("---")
    st.subheader("📊 Component Detection Summary")
    
    # Extract detected boxes and classes
    boxes = results[0].boxes
    class_names = model.names
    
    # Count occurrences of each detected class
    class_counts = {}
    for c in boxes.cls:
        cls_name = class_names[int(c)]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
    if class_counts:
        # Display metrics in columns
        metric_cols = st.metric(label="Total Components Detected", value=len(boxes))
        
        # Show breakdown table or metrics
        st.write("Detected Components Breakdown:")
        st.json(class_counts)
    else:
        st.warning("No components detected above the current confidence threshold. Try lowering the threshold on the sidebar.")