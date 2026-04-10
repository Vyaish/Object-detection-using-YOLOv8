
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("👥 People Detection using YOLOv8")
st.write("Upload one or more images to detect people")

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📸 {uploaded_file.name}")

        # Convert uploaded file to OpenCV format
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Run detection
        results = model(image_np)

        # Annotated image
        annotated_image = results[0].plot()

        # Count people (class 0 = person)
        person_count = sum(
            1 for c in results[0].boxes.cls if int(c) == 0
        )

        # Display results
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        st.success(f"👤 People detected: {person_count}")

