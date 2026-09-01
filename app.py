import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import time
import traceback
import inspect

def stretch_kw(fn=st.image):
    """
    Returns {'width': 'stretch'} for modern Streamlit (>= 1.40) to eliminate
    deprecation warnings, or falls back to {'use_container_width': True} for older versions.
    """
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}

# Set Streamlit page configuration first
st.set_page_config(
    page_title="Road Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_str: str):
    """
    Renders an HTML string in Streamlit by stripping newlines and extra spaces
    to prevent the markdown parser from treating it as a code block.
    """
    clean_html = " ".join([line.strip() for line in html_str.split("\n") if line.strip()])
    st.markdown(clean_html, unsafe_allow_html=True)

# ----------------- PATH & MODEL LOADING -----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

@st.cache_resource
def load_model(model_path: Path):
    """
    Loads the YOLO model and caches the resource so it is not reloaded on every run.
    """
    from ultralytics import YOLO
    if not model_path.exists():
        return None
    try:
        return YOLO(str(model_path))
    except Exception:
        return None

# Check model status
model_exists = MODEL_PATH.exists()
model = load_model(MODEL_PATH) if model_exists else None

# Custom Premium CSS Styling (Glassmorphism + Dark Slate Accents)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main Layout Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Reduce default Streamlit padding at the top of the page */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
    }
    
    /* Headers & Subtitles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Premium Header Container (Universal Light & Dark Glassmorphism) */
    .header-banner {
        background: rgba(128, 128, 128, 0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: inherit;
        box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-banner::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -8%;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(239, 68, 68, 0.18) 0%, rgba(239, 68, 68, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .header-banner::before {
        content: "";
        position: absolute;
        bottom: -40%;
        left: -8%;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.14) 0%, rgba(59, 130, 246, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .header-stats-pill {
        background: rgba(128, 128, 128, 0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 0.7rem 1.25rem;
        min-width: 160px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: inherit;
    }

    /* Sidebar metadata card */
    .metadata-card {
        background: rgba(128, 128, 128, 0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-size: 0.82rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: inherit;
    }

    /* System ready box */
    .system-ready-card {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        color: inherit;
    }

    /* Status Badge Indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 5px;
    }
    
    .status-active {
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    
    .status-missing {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    
    /* Premium Metric Card */
    .metric-card-container {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 1.75rem;
        width: 100%;
    }
    
    .metric-card-custom {
        flex: 1;
        background: rgba(128, 128, 128, 0.08);
        backdrop-filter: blur(10px);
        color: inherit;
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-left: 5px solid #cbd5e1;
    }
    
    .metric-card-custom:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card-potholes {
        border-left: 5px solid #ef4444;
    }
    
    .metric-card-confidence {
        border-left: 5px solid #10b981;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
        color: inherit;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        color: inherit;
    }
    
    /* Table Styling for Details */
    .details-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: rgba(128, 128, 128, 0.04);
    }
    
    .details-table th {
        background-color: rgba(128, 128, 128, 0.12);
        color: inherit;
        font-weight: 700;
        text-align: left;
        padding: 12px 16px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid rgba(128, 128, 128, 0.25);
    }
    
    .details-table td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.12);
        color: inherit;
        font-size: 0.95rem;
    }
    
    .details-table tr:hover {
        background-color: rgba(128, 128, 128, 0.08);
    }
    
    /* Badge styling in table */
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-mid {
        background-color: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .badge-low {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Sidebar improvements */
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        margin-top: 0px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Shift sidebar content up */
    [data-testid="stSidebarContent"] {
        padding-top: 1.2rem !important;
    }
    
    [data-testid="stSidebarContent"] > div:first-child {
        padding-top: 0rem !important;
    }
    
    /* Compact sidebar elements */
    [data-testid="stSidebar"] h3 {
        margin-top: 0.75rem !important;
        margin-bottom: 0.25rem !important;
        font-size: 1.05rem !important;
    }
    
    [data-testid="stSidebar"] .stSlider {
        margin-bottom: 0.25rem !important;
        padding-bottom: 0px !important;
    }
    
    /* Custom File Uploader Dropzone Styling */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(128, 128, 128, 0.25) !important;
        background-color: rgba(128, 128, 128, 0.05) !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.5rem !important;
        transition: all 0.25s ease-in-out !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ef4444 !important;
        background-color: rgba(239, 68, 68, 0.05) !important;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.1) !important;
    }
    
    /* Clean Modern Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        margin-bottom: 0.75rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
    }
    
    /* Primary button enhancements (gradient button) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #ff7849 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.28) !important;
        transition: all 0.25s ease !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(239, 68, 68, 0.4) !important;
        background: linear-gradient(135deg, #ff7849 0%, #ef4444 100%) !important;
    }
    div.stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }
    
    /* Custom Slider Accent Styling */
    div[data-testid="stSlider"] > div {
        color: #ef4444 !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #ef4444 !important;
        border: 2px solid #ffffff !important;
    }
    div[data-testid="stSlider"] div[data-testid="stSliderTrack"] > div > div {
        background-color: #ef4444 !important;
    }
    
    /* Custom clean scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(128, 128, 128, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(128, 128, 128, 0.5);
    }
    
    /* Image container styling */
    [data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Alert components styling */
    [data-testid="stNotification"] {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)
    
    # Model Status Card
    st.markdown("### System Status")
    if model_exists:
        if model is not None:
            st.markdown('<div class="status-badge status-active">🟢 YOLO Detector Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-missing">🔴 Model Weights Error</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-missing">🔴 Model Weights Missing</div>', unsafe_allow_html=True)
        
    st.markdown("### Parameters")
    
    # Slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher threshold displays only higher confidence detections. Lower threshold displays more potential potholes."
    )
    
    # Details Card
    st.markdown("### Model Metadata")
    render_html(f"""
    <div class="metadata-card">
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(128, 128, 128, 0.12); padding-bottom: 5px; margin-bottom: 6px;">
            <span style="opacity: 0.7;">Weights File</span>
            <b>best.pt</b>
        </div>
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(128, 128, 128, 0.12); padding-bottom: 5px; margin-bottom: 6px;">
            <span style="opacity: 0.7;">Architecture</span>
            <b>YOLOv8</b>
        </div>
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(128, 128, 128, 0.12); padding-bottom: 5px; margin-bottom: 6px;">
            <span style="opacity: 0.7;">Target Class</span>
            <b style="color: #ef4444;">Pothole</b>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="opacity: 0.7;">Input Mode</span>
            <b>Image (RGB)</b>
        </div>
    </div>
    """)
    
    render_html("<p style='font-size: 0.8rem; text-align: center; margin-top: 1.25rem; opacity: 0.7;'>Supported Formats: JPG, JPEG, PNG</p>")

# ----------------- HEADER BANNER -----------------
render_html("""
<div class="header-banner">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.25rem;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; line-height: 1.2; color: inherit;">Road Pothole Detection System</h1>
            <p style="margin: 6px 0 0 0; font-size: 0.95rem; opacity: 0.75; color: inherit; max-width: 620px;">
                Automated road surface inspection powered by custom YOLOv8 deep learning model.
            </p>
        </div>
        <div class="header-stats-pill">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; font-weight: 600; color: inherit;">Detection Engine</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ef4444; font-family: 'Space Grotesk', sans-serif;">YOLOv8 Active</div>
            <div style="font-size: 0.75rem; opacity: 0.65; color: inherit;">Real-time Bounding Boxes</div>
        </div>
    </div>
</div>
""")

# ----------------- UPLOAD SECTION -----------------
st.markdown("### 📤 Upload Surface Imagery")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ----------------- APPLICATION FLOW -----------------
if uploaded_file is not None:
    try:
        # Reset detection results if a new file is uploaded
        file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
        if st.session_state.get("current_file_id") != file_id:
            st.session_state["current_file_id"] = file_id
            st.session_state["detection_result"] = None

        # Load and orient PIL image correctly
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        # Convert image to RGB
        image_rgb = image.convert("RGB")
        image_array = np.array(image_rgb)
        image_corrupted = False
    except Exception as e:
        st.error(f"❌ **Invalid Image File**: The uploaded image file is corrupt or unreadable. Details: {e}")
        image_corrupted = True

    if not image_corrupted:
        st.markdown("---")
        # Layout columns: Preview on Left, Control/Trigger on Right
        col_preview, col_trigger = st.columns([5, 3])
        
        has_results = st.session_state.get("detection_result") is not None
        
        with col_preview:
            st.markdown("#### 🖼️ Image Preview")
            if has_results:
                tab_pred, tab_orig = st.tabs(["🎯 Detection Result", "🖼️ Original Image"])
                with tab_pred:
                    st.image(st.session_state["detection_result"]["annotated_image"], **stretch_kw(st.image))
                with tab_orig:
                    st.image(image_rgb, **stretch_kw(st.image))
            else:
                st.image(image_rgb, **stretch_kw(st.image))
            
        with col_trigger:
            st.markdown("#### 🚀 Detection Control")
            st.caption("Feed the input surface imagery to the YOLO model to detect and segment road hazards.")
            
            detect_button = st.button("🔎 Detect Potholes", type="primary", **stretch_kw(st.button))
            
            if has_results:
                det = st.session_state["detection_result"]
                cnt = det["pothole_count"]
                if cnt > 0:
                    st.success(f"✅ **Detection Complete!** Identified **{cnt}** pothole{'s' if cnt > 1 else ''}. Full report and comparison below 👇")
                else:
                    st.warning("⚠️ **Detection Complete**: No potholes identified with current confidence threshold. Try lowering the threshold slider in the sidebar.")
            elif not model_exists:
                st.markdown("""
                <div style="background-color: #fee2e2; border: 1px solid #fecaca; border-radius: 12px; padding: 1rem; margin-top: 1rem;">
                    <span style="color: #b91c1c; font-weight: 600; font-size: 0.95rem;">Model file best.pt not found!</span><br/>
                    <span style="color: #7f1d1d; font-size: 0.85rem;">Please place your custom-trained YOLO weights (best.pt) in the root project folder to allow detection.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                render_html(f"""
                <div class="system-ready-card">
                    <div style="color: #3b82f6; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px;">
                        ⚡ System Ready
                    </div>
                    <div style="font-size: 0.85rem; opacity: 0.85; line-height: 1.4;">
                        Confidence threshold set to <b>{confidence_threshold * 100:.0f}%</b>. Adjust sensitivity in the sidebar anytime.
                    </div>
                </div>
                """)

        if detect_button:
            if model is None:
                st.error("🛑 **Model Error**: The model `best.pt` could not be loaded. Please ensure the weights file is valid and placed in the project folder.")
            else:
                # Add spinner with progress-like animation style
                with st.spinner("⚡ Running deep-learning inference pipeline..."):
                    try:
                        # Time inference
                        start_time = time.time()
                        results = model.predict(
                            source=image_array,
                            conf=confidence_threshold,
                            verbose=False
                        )
                        inference_time = (time.time() - start_time) * 1000
                        
                        result = results[0]
                        pothole_count = len(result.boxes)
                        
                        # Generate annotated image
                        annotated_image_bgr = result.plot()
                        annotated_image_rgb = annotated_image_bgr[:, :, ::-1]
                        
                        # Extract confidences
                        confidences = []
                        if pothole_count > 0:
                            confidences = result.boxes.conf.cpu().numpy().tolist()
                        
                        # Save in session state so results persist across interactions
                        st.session_state["detection_result"] = {
                            "annotated_image": annotated_image_rgb,
                            "pothole_count": pothole_count,
                            "confidences": confidences,
                            "inference_time": inference_time,
                        }
                        st.rerun()
                            
                    except Exception as pred_err:
                        st.error("❌ **Prediction Failed**: The model encountered a processing error while predicting on this image array.")
                        with st.expander("Show detailed logs"):
                            st.code(traceback.format_exc())

        # Render Full Report & Analytics when detection has completed
        if st.session_state.get("detection_result") is not None:
            det = st.session_state["detection_result"]
            pothole_count = det["pothole_count"]
            confidences = det["confidences"]
            inference_time = det["inference_time"]
            annotated_image_rgb = det["annotated_image"]

            # Output comparison
            st.markdown("---")
            st.markdown("### 📊 Detection Results")
            
            col_orig, col_pred = st.columns(2)
            
            with col_orig:
                st.markdown('<div style="text-align: center; font-weight: 600; padding-bottom: 8px; color: inherit; opacity: 0.85;">Original Image</div>', unsafe_allow_html=True)
                st.image(image_rgb, **stretch_kw(st.image))
                
            with col_pred:
                st.markdown('<div style="text-align: center; font-weight: 600; padding-bottom: 8px; color: #ef4444;">Annotated Result (Bounding Boxes)</div>', unsafe_allow_html=True)
                st.image(annotated_image_rgb, **stretch_kw(st.image))

            # Render Premium Summary Cards
            st.markdown("---")
            st.markdown("### 📈 Session Summary")
            
            card_html_1 = f"""
            <div class="metric-card-custom metric-card-potholes">
                <div class="metric-label">Total Potholes Detected</div>
                <div class="metric-value">{pothole_count}</div>
                <div style="font-size: 0.8rem; opacity: 0.7; color: inherit; margin-top: 4px;">Surface defects identified</div>
            </div>
            """
            
            if pothole_count > 0:
                max_conf_pct = max(confidences) * 100
                max_conf_str = f"{max_conf_pct:.2f}%"
                conf_subtext = f"Across {pothole_count} detections"
            else:
                max_conf_str = "N/A"
                conf_subtext = "No detections made"
                
            card_html_2 = f"""
            <div class="metric-card-custom metric-card-confidence">
                <div class="metric-label">Highest Confidence Score</div>
                <div class="metric-value" style="color: { '#10b981' if pothole_count > 0 else 'inherit' }">{max_conf_str}</div>
                <div style="font-size: 0.8rem; opacity: 0.7; color: inherit; margin-top: 4px;">{conf_subtext}</div>
            </div>
            """
            
            # Custom card column structure
            render_html(f"""
            <div class="metric-card-container">
                {card_html_1}
                {card_html_2}
                <div class="metric-card-custom" style="border-left: 5px solid #3b82f6;">
                    <div class="metric-label">Inference Speed</div>
                    <div class="metric-value" style="color: #3b82f6;">{inference_time:.1f}<span style="font-size: 1.2rem;"> ms</span></div>
                    <div style="font-size: 0.8rem; opacity: 0.7; color: inherit; margin-top: 4px;">YOLO execution duration</div>
                </div>
            </div>
            """)
            
            # Detailed detection breakdown
            st.markdown("### 📋 Detailed Detection Breakdown")
            if pothole_count > 0:
                # Order confidences descending
                sorted_conf = sorted(confidences, reverse=True)
                
                # Render HTML Table
                table_rows = ""
                for idx, conf in enumerate(sorted_conf, 1):
                    pct = conf * 100
                    # Determine badge color
                    if pct >= 75:
                        badge_class = "badge-high"
                        badge_text = "High Confidence"
                    elif pct >= 45:
                        badge_class = "badge-mid"
                        badge_text = "Medium Confidence"
                    else:
                        badge_class = "badge-low"
                        badge_text = "Low Confidence"
                        
                    table_rows += f"""
                    <tr>
                        <td>Pothole #{idx}</td>
                        <td><b>{pct:.2f}%</b></td>
                        <td><span class="badge {badge_class}">{badge_text}</span></td>
                    </tr>
                    """
                    
                render_html(f"""
                <table class="details-table">
                    <thead>
                        <tr>
                            <th>Detection Index</th>
                            <th>Confidence Value</th>
                            <th>Reliability Tier</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """)
            else:
                st.warning("⚠️ **Detection Complete**: No pothole signatures were identified. Try adjusting the confidence slider lower in the settings if you suspect false negatives.")
else:
    # Clean adaptive empty state graphic
    render_html("""
    <div style="text-align: center; padding: 3.5rem 2rem; background: rgba(128, 128, 128, 0.05); border-radius: 16px; border: 2px dashed rgba(128, 128, 128, 0.2); margin-top: 1.5rem; color: inherit;">
        <span style="font-size: 3.5rem; display: block; margin-bottom: 0.75rem;">🛣️</span>
        <h3 style="margin: 0; color: inherit; font-size: 1.3rem;">No Image Uploaded Yet</h3>
        <p style="opacity: 0.7; font-size: 0.92rem; max-width: 420px; margin: 8px auto 0 auto; line-height: 1.5;">
            Upload surface imagery in JPG, JPEG, or PNG format using the file browser above to run the YOLO model detectors.
        </p>
    </div>
    """)
