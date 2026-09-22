import os
import time
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="Neuro-Somatic AI | CAD Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Executive CSS Styling
st.markdown("""
    <style>
    /* Dark Theme Custom Colors */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header Card Styling */
    .header-card {
        background: linear-gradient(135deg, #1E2640 0%, #0E1117 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #2E3856;
        margin-bottom: 25px;
    }
    
    .header-title {
        color: #00E5FF;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .header-subtitle {
        color: #A0AEC0;
        font-size: 15px;
    }

    /* Metric Card Styling */
    .metric-card {
        background-color: #161B26;
        padding: 18px;
        border-radius: 10px;
        border-left: 4px solid #00E5FF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    .metric-label {
        font-size: 13px;
        color: #8C9BAE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #12161F;
        border-right: 1px solid #222938;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading with Caching
@st.cache_resource
def load_yolo_model():
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        # Fallback path if inside directory
        model_path = "best.pt"
    return YOLO(model_path)

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"⚠️ Model Loading Failed: {e}")
    st.stop()

# 4. Sidebar Controls & Model Metadata
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=70)
    st.title("🧠 Diagnostics Control")
    st.caption("Neuro-Somatic AI v1.0")
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance")
    st.markdown("**Architecture:** YOLOv8 Fine-Tuned")
    st.markdown("**Accuracy:** 95.5%")
    st.markdown("**Latency:** 21 ms")
    st.markdown("**Training Epochs:** 50")
    st.markdown("---")
    
    confidence_threshold = st.slider("Min Confidence Threshold (%)", 0, 100, 50) / 100.0
    
    st.markdown("---")
    st.caption("Developed by Sarthak Shrivastava | Electronics Engineering")

# 5. Header Banner
st.markdown("""
    <div class="header-card">
        <div class="header-title">🧠 Neuro-Somatic AI Diagnostics</div>
        <div class="header-subtitle">
            Deep Learning Computer Vision System for CAD & Medical Scan Classification
        </div>
    </div>
""", unsafe_allow_html=True)

# 6. Main Layout Setup
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📸 Image Acquisition")
    uploaded_file = st.file_uploader(
        "Upload a medical scan or test image", 
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan", use_container_width=True)

with col_results:
    st.subheader("🔍 Diagnostic Results")
    
    if uploaded_file is not None:
        run_btn = st.button("⚡ Run Diagnostic Inference", type="primary", use_container_width=True)
        
        if run_btn:
            # Save temporary file for YOLO processing
            temp_path = "temp_scan.jpg"
            image.save(temp_path)
            
            start_time = time.time()
            with st.spinner("Analyzing scan features via Neural Network..."):
                results = model.predict(source=temp_path)
                latency = (time.time() - start_time) * 1000
                
                probs = results[0].probs
                classes = model.names
                top1_idx = probs.top1
                top1_class = classes[top1_idx]
                top1_conf = probs.top1conf.item() * 100
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Show Metrics
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Diagnosis</div>
                    <div class="metric-value" style="color: {'#00E5FF' if top1_conf >= confidence_threshold*100 else '#FF4B4B'};">
                        {top1_class}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Confidence Level", value=f"{top1_conf:.2f}%")
            with m_col2:
                st.metric(label="Inference Latency", value=f"{latency:.1f} ms")
            
            st.markdown("### 📈 Class Probability Distribution")
            
            # Sort and display probability breakdown
            all_probs = probs.data.tolist()
            prob_dict = {classes[i]: all_probs[i] * 100 for i in range(len(classes))}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            for cls_name, prob_val in sorted_probs:
                st.write(f"**{cls_name}** ({prob_val:.2f}%)")
                st.progress(int(prob_val))
    else:
        st.info("👈 Please upload an image in the left panel to execute diagnosis.")