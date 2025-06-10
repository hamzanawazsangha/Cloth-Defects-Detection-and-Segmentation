import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json
import time

# Enhanced Mobile-optimized CSS with better contrast and responsive design
st.markdown("""
<style>
    /* Base responsive styles with improved contrast */
    html {
        font-size: 18px;  /* Increased base font size */
    }
    
    /* Main page styling - responsive with better contrast */
    .stApp {
        background-color: #ffffff;
        padding: 0.5rem;
    }
    
    /* Responsive header with better contrast */
    .header {
        color: #1a237e;  /* Darker blue for better contrast */
        text-align: center;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3949ab;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        font-size: clamp(1.8rem, 5vw, 2.2rem);
        margin: 0.2rem 0;
    }
    
    /* Uploader with better visibility */
    .uploader-container {
        border: 2px dashed #303f9f;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        background-color: rgba(48, 63, 159, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .uploader-container p {
        color: #1a237e;
        font-size: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Cards with better contrast */
    .card {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        background-color: white;
        border: 1px solid #e0e0e0;
    }
    
    .card h3, .card h4 {
        color: #1a237e;
        margin-top: 0.2rem;
    }
    
    /* Buttons with better contrast */
    .stButton > button {
        background-color: #303f9f;
        color: white;
        border-radius: 6px;
        padding: 0.6rem;
        border: none;
        width: 100%;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #303f9f !important;
    }
    
    /* Tab styling for mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem !important;
        padding: 8px 10px !important;
    }
    
    /* Confidence meter styling */
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-level {
        height: 100%;
        background: linear-gradient(90deg, #5c6bc0, #303f9f);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
        color: white;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Image captions */
    .stImage > div > div > div > p {
        font-size: 0.9rem !important;
        color: #424242 !important;
        text-align: center;
    }
    
    /* Mobile-specific optimizations */
    @media (max-width: 480px) {
        html {
            font-size: 16px;
        }
        
        .header h1 {
            font-size: 1.5rem;
        }
        
        .uploader-container {
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem !important;
            padding: 6px 8px !important;
        }
        
        /* Force single column layout on mobile */
        .st-c5 {
            width: 100% !important;
            flex: 0 0 100% !important;
        }
        
        /* Adjust spacing for mobile */
        .stImage {
            margin-bottom: 0.5rem;
        }
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #121212;
        }
        
        .card {
            background-color: #1e1e1e;
            border-color: #333;
        }
        
        .header {
            color: #bbdefb;
            background: linear-gradient(135deg, #1a237e 0%, #303f9f 100%);
        }
        
        .uploader-container {
            background-color: rgba(30, 30, 30, 0.5);
            border-color: #5c6bc0;
        }
        
        .uploader-container p {
            color: #e0e0e0;
        }
        
        .card h3, .card h4 {
            color: #bbdefb;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_interpreters():
    # Load models (replace with your actual model paths)
    cls_int = tf.lite.Interpreter(model_path="classifier.tflite")
    seg_int = tf.lite.Interpreter(model_path="segmentation_model.tflite")
    cls_int.allocate_tensors()
    seg_int.allocate_tensors()

    # Get segmentation input details
    seg_input_details = seg_int.get_input_details()[0]
    seg_input_dtype = seg_input_details['dtype']
    seg_input_scale, seg_input_zero_point = 1.0, 0
    if 'quantization' in seg_input_details:
        seg_input_scale, seg_input_zero_point = seg_input_details['quantization']
    return cls_int, seg_int, seg_input_dtype, seg_input_scale, seg_input_zero_point

@st.cache_resource
def load_labels():
    # Load class labels (replace with your actual label file)
    with open("class_labels.json") as f:
        return json.load(f)

cls_int, seg_int, seg_dtype, seg_scale, seg_zero_point = load_interpreters()
class_labels = load_labels()

def preprocess(img, size, dtype=tf.float32, scale=1.0, zero_point=0):
    img = cv2.resize(img, size)

    if dtype == np.uint8 or dtype == np.int8:
        img = img.astype(np.float32)
        img = img / 255.0
        img = img / scale + zero_point
        img = np.round(img).astype(dtype)
    else:
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

def run_inference(img):
    # Classification
    cls_input_details = cls_int.get_input_details()
    cls_dtype = cls_input_details[0]['dtype']
    cls_input = preprocess(img, (224, 224), dtype=cls_dtype)
    cls_int.set_tensor(cls_input_details[0]['index'], cls_input)
    cls_int.invoke()
    cls_pred = cls_int.get_tensor(cls_int.get_output_details()[0]['index'])[0]
    cls_idx = int(np.argmax(cls_pred))
    cls_label = class_labels[cls_idx]
    cls_conf = float(np.max(cls_pred))

    # Segmentation
    seg_input_details = seg_int.get_input_details()
    seg_input_dtype = seg_input_details[0]['dtype']
    seg_scale, seg_zero_point = seg_input_details[0]['quantization']
    seg_input = preprocess(img, (256, 256), dtype=seg_input_dtype, scale=seg_scale, zero_point=seg_zero_point)
    seg_int.set_tensor(seg_input_details[0]['index'], seg_input)
    seg_int.invoke()
    seg_pred = seg_int.get_tensor(seg_int.get_output_details()[0]['index'])[0]

    # Dequantize if needed
    seg_output_details = seg_int.get_output_details()[0]
    seg_out_scale, seg_out_zero_point = seg_output_details['quantization']
    if seg_output_details['dtype'] in [np.uint8, np.int8]:
        seg_pred = seg_out_scale * (seg_pred.astype(np.float32) - seg_out_zero_point)

    if seg_pred.shape[-1] == 1:
        seg_mask = seg_pred[:, :, 0]
    else:
        seg_mask = np.argmax(seg_pred, axis=-1)

    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return cls_label, cls_conf, seg_mask

def overlay_mask(image, mask, alpha=0.5):
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [255, 0, 0]
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return overlay

def create_segmented_output(original_img, mask):
    segmented = np.zeros((original_img.shape[0], original_img.shape[1], 4), dtype=np.uint8)
    segmented[mask == 1, :3] = original_img[mask == 1]
    segmented[mask == 1, 3] = 255
    segmented[mask == 0, 3] = 0
    return segmented

# Mobile-responsive app layout
st.markdown('<div class="header"><h1>🧵 Cloth Defect Detection</h1></div>', unsafe_allow_html=True)

# Sidebar with improved mobile layout
with st.sidebar:
    st.markdown('<div style="font-size:1.1rem; font-weight:bold; margin-bottom:0.5rem;">Settings</div>', unsafe_allow_html=True)
    alpha = st.slider("Mask Opacity", 0.1, 1.0, 0.5, 0.05, help="Adjust the transparency of the defect overlay")
    show_segmented = st.checkbox("Show Segmented", True)
    show_original = st.checkbox("Show Original", False)
    
    st.divider()
    
    st.markdown('<div style="font-size:1.1rem; font-weight:bold; margin-bottom:0.5rem;">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.9rem;">
    This app detects defects in cloth images using deep learning.
    <br><br>
    <b>Models:</b><br>
    - Classifier: Identifies defect type<br>
    - Segmenter: Locates defect areas
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown('<div style="font-size:1.1rem; font-weight:bold; margin-bottom:0.5rem;">Performance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "92.5%", delta="+2.3%")
    with col2:
        st.metric("Speed", "1.2s", delta="-0.3s")

# Main content with improved uploader section
uploader_container = st.container()
with uploader_container:
    st.markdown("""
    <div class="uploader-container">
        <p style="font-weight:bold; font-size:1.1rem;">Upload Cloth Image</p>
        <p style="font-size:0.9rem;">JPG, JPEG, PNG • Max 200MB</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload cloth image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded:
    with st.spinner("Analyzing image..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        label, confidence, mask = run_inference(img_np)
        overlay = overlay_mask(img_np, mask, alpha)
        segmented_output = create_segmented_output(img_np, mask)
    
    # Results section with better mobile layout
    st.markdown(f'<div class="card"><h3>🔍 Detection Results</h3></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f'<div class="card"><h4>Defect Type: {label}</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin:0.5rem 0;">
            <div>Confidence:</div>
            <div class="confidence-meter">
                <div class="confidence-level" style="width:{confidence*100}%">
                    {int(confidence*100)}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Mobile-optimized tabs with clearer labels
    tabs = st.tabs(["📊 Overview", "🔍 Details", "⚙ Advanced"])
    
    with tabs[0]:  # Overview tab
        st.image(overlay, use_column_width=True, caption=f"Detected Defect: {label}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Defect Area", f"{np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100:.1f}%")
        with col2:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            st.metric("Defect Count", len(contours))
    
    with tabs[1]:  # Details tab
        if show_segmented:
            st.image(segmented_output, use_column_width=True, caption="Segmented defect areas")
        st.image(mask * 255, use_column_width=True, caption="Defect mask visualization", clamp=True)
    
    with tabs[2]:  # Advanced tab
        if show_original:
            st.image(img_np, use_column_width=True, caption="Original image")
        
        # Additional defect metrics
        defect_area = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100
        st.markdown("**Defect Statistics**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Width", f"{img_np.shape[1]}px")
            st.metric("Height", f"{img_np.shape[0]}px")
        with col2:
            st.metric("Area", f"{defect_area:.1f}%")
            st.metric("Confidence", f"{confidence*100:.1f}%")
    
    # Enhanced download buttons for mobile
    st.divider()
    st.markdown('<div style="font-size:1rem; font-weight:bold; margin-bottom:0.5rem;">Export Results</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "📥 Download Overlay",
            cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes(),
            "defect_overlay.png",
            "image/png",
            help="Download the defect overlay image"
        )
    with cols[1]:
        st.download_button(
            "📥 Download Report",
            f"Defect Analysis Report\n\nDefect Type: {label}\nConfidence: {confidence:.1%}\nDefect Area: {defect_area:.1f}%\nImage Dimensions: {img_np.shape[1]}x{img_np.shape[0]}",
            "defect_report.txt",
            "text/plain",
            help="Download the defect analysis report"
        )

else:
    # Improved empty state with better mobile layout
    st.markdown("""
    <div class="card">
        <h3>How to Use</h3>
        <div style="font-size:0.95rem;">
            <ol>
                <li style="margin-bottom:0.5rem;">📤 Upload a cloth image</li>
                <li style="margin-bottom:0.5rem;">🔄 Wait for analysis</li>
                <li style="margin-bottom:0.5rem;">🔍 View defect details</li>
                <li>💾 Export results if needed</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample images with better mobile layout
    st.markdown('<div class="card"><h4>Example Defects</h4></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    sample_images = [
        ("Hole", "https://via.placeholder.com/300x200/5c6bc0/ffffff?text=Hole"),
        ("Stain", "https://via.placeholder.com/300x200/303f9f/ffffff?text=Stain"),
        ("Tear", "https://via.placeholder.com/300x200/7986cb/ffffff?text=Tear"),
        ("Fraying", "https://via.placeholder.com/300x200/3949ab/ffffff?text=Fraying")
    ]
    
    for i, (caption, url) in enumerate(sample_images):
        if i % 2 == 0:
            with cols[0]:
                st.image(url, caption=caption, use_column_width=True)
        else:
            with cols[1]:
                st.image(url, caption=caption, use_column_width=True)
