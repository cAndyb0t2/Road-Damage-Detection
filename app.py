
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import pandas as pd
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RoadGuard AI | Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #273449;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f9fafb;
}

.hero {
    background: linear-gradient(120deg, #142d38, #152238 65%, #20253d);
    border: 1px solid #304452;
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero-label {
    color: #4ade80;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #ffffff;
    font-size: 35px;
    font-weight: 800;
    margin: 10px 0;
}

.hero p {
    color: #bdc9d8;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #f9fafb;
    margin: 20px 0 12px 0;
}

div[data-testid="stMetric"] {
    background: #141e30;
    border: 1px solid #29374b;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #9caec2;
}

div[data-testid="stMetricValue"] {
    color: #4ade80;
}

.stButton > button {
    background: #22c55e;
    color: #07130b;
    font-weight: 800;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #4ade80;
    color: #07130b;
    border: none;
}

.stDownloadButton > button {
    border-radius: 10px;
    min-height: 42px;
}

div[data-testid="stFileUploader"] {
    background: #121d2e;
    border: 1px dashed #46566e;
    border-radius: 14px;
    padding: 12px;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

hr {
    border-color: #29374b;
}

.small-note {
    color: #94a3b8;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO(r"D:\RoadDamageDetection\Models\best.pt")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "result_data" not in st.session_state:
    st.session_state.result_data = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🛣️ RoadGuard AI")
    st.caption("INTELLIGENT ROAD INSPECTION")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Lower values show more possible detections, including uncertain ones."
    )

    st.markdown("### 🧠 AI Model")
    st.write("YOLOv8 object detector")
    st.write("5 model classes")

    st.divider()
    st.markdown(
        '<p class="small-note">Academic Project · Deep Learning</p>',
        unsafe_allow_html=True
    )

# ---------------- LOAD MODEL ----------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# ---------------- ABOUT PAGE ----------------
if page == "About Project":
    st.markdown("""
    <div class="hero">
        <div class="hero-label">PROJECT OVERVIEW</div>
        <h1>Road Damage Detection System</h1>
        <p>A computer vision application for identifying potential road damage
        in uploaded road images using a trained YOLOv8 model.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Objective")
    st.write(
        "To assist road inspection by automatically locating potential damage "
        "in images and presenting the detected regions for review."
    )

    st.markdown("### 🔬 Technology Stack")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Deep Learning**")
    c1.write("YOLOv8 · PyTorch")
    c2.markdown("**Interface**")
    c2.write("Python · Streamlit")
    c3.markdown("**Image Processing**")
    c3.write("Pillow · OpenCV through model plotting")

    st.markdown("### ⚙️ Workflow")
    st.write("1. Upload a road image.")
    st.write("2. The trained model processes the image.")
    st.write("3. Potential damage regions are marked with bounding boxes.")
    st.write("4. Review detection confidence and download the result.")

    st.info(
        "The displayed class IDs are the model's current class labels. "
        "Use the verified dataset class mapping before assigning damage names."
    )

# ---------------- DASHBOARD ----------------
else:
    st.markdown("""
    <div class="hero">
        <div class="hero-label">AI-POWERED ROAD INSPECTION</div>
        <h1>Road Damage Detection</h1>
        <p>Upload a road image, run AI inference, and review the detected regions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📤 Upload Road Image</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        help="For best results, use a clear image where the road surface is visible."
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown('<div class="section-title">🖼️ Image Preview</div>',
                    unsafe_allow_html=True)

        preview_col, info_col = st.columns([1.7, 1])

        with preview_col:
            st.image(image, caption="Uploaded road image",
                     use_container_width=True)

        with info_col:
            st.markdown("#### Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Resolution:** {image.width} × {image.height}")
            st.write(f"**Format:** {uploaded_file.type or 'Image'}")

            st.markdown("#### Inference Configuration")
            st.write("**Model:** YOLOv8")
            st.write(f"**Image size:** 640")
            st.write(f"**Confidence:** {confidence:.0%}")

        if st.button("🚀 Detect Road Damage", type="primary",
                     use_container_width=True):
            with st.spinner("AI model is analysing the road image..."):
                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )

                result = results[0]
                annotated = result.plot()[:, :, ::-1].copy()
                detections = []

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    x1, y1, x2, y2 = [
                        round(float(v), 1)
                        for v in box.xyxy[0].tolist()
                    ]

                    detections.append({
                        "Class ID": class_id,
                        "Model Label": str(model.names[class_id]),
                        "Confidence": round(score * 100, 2),
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2
                    })

                st.session_state.result_data = {
                    "image": annotated,
                    "detections": detections,
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    "confidence": confidence
                }

                st.session_state.history.insert(0, {
                    "Time": st.session_state.result_data["timestamp"],
                    "Image": uploaded_file.name,
                    "Detections": len(detections),
                    "Threshold": f"{confidence:.0%}"
                })
                st.session_state.history = st.session_state.history[:10]

    # ---------------- RESULTS ----------------
    data = st.session_state.result_data

    if data:
        st.divider()
        st.markdown('<div class="section-title">📊 Detection Results</div>',
                    unsafe_allow_html=True)

        detections = data["detections"]
        count = len(detections)
        avg_conf = (
            sum(d["Confidence"] for d in detections) / count
            if count else 0
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects Detected", count)
        m2.metric("Average Confidence", f"{avg_conf:.1f}%")
        m3.metric("Confidence Threshold", f"{data['confidence']:.0%}")

        original_col, result_col = st.columns(2)

        with original_col:
            st.markdown("#### Original Image")
            # Show the original image only when it matches the latest result.
            if uploaded_file and uploaded_file.name == data["filename"]:
                st.image(image, use_container_width=True)
            else:
                st.caption("Showing the last processed result.")

        with result_col:
            st.markdown("#### AI Detection Output")
            st.image(data["image"], use_container_width=True)

        if count:
            st.markdown("#### 🔎 Detection Details")
            df = pd.DataFrame(detections)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence (%)",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%"
                    )
                }
            )
        else:
            st.warning(
                "No objects were detected at this threshold. "
                "You can try a lower confidence threshold."
            )

        output = io.BytesIO()
        Image.fromarray(data["image"]).save(output, format="PNG")

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                "⬇️ Download Annotated Image",
                data=output.getvalue(),
                file_name="road_damage_result.png",
                mime="image/png",
                use_container_width=True
            )

        with download_col2:
            csv_data = pd.DataFrame(detections).to_csv(index=False)
            st.download_button(
                "⬇️ Download Detection Report",
                data=csv_data,
                file_name="road_damage_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ---------------- HISTORY ----------------
    st.divider()
    st.markdown('<div class="section-title">🕘 Recent Analyses</div>',
                unsafe_allow_html=True)

    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Your recent image analyses will appear here.")

    st.markdown(
        '<p class="small-note">RoadGuard AI · Results depend on model performance, '
        'image quality, and the selected confidence threshold. '
        'Detections should be reviewed before real-world use.</p>',
        unsafe_allow_html=True
    )
