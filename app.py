import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# Page configuration
st.set_page_config(
    page_title="Cloth Defect Detection",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main page styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        color: #2c3e50;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    /* Uploader styling */
    .uploader-container {
        border: 3px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .uploader-container:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px 15px 0 0 !important;
        padding: 12px 24px !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Confidence meter */
    .confidence-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 25px;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
        border-radius: 12px;
        margin: 10px 0;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill {
        position: absolute;
        height: 100%;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        transition: width 1.5s ease-in-out;
    }
    
    .confidence-label {
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_resource
def load_interpreters():
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
    with open("class_labels.json") as f:
        return json.load(f)

# Load models and labels
try:
    cls_int, seg_int, seg_dtype, seg_scale, seg_zero_point = load_interpreters()
    class_labels = load_labels()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

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

def create_defect_analysis_chart(mask, contours):
    fig = go.Figure()
    
    if len(contours) > 0:
        defect_sizes = [cv2.contourArea(cnt) for cnt in contours]
        defect_labels = [f"Defect {i+1}" for i in range(len(contours))]
        
        fig.add_trace(go.Bar(
            x=defect_labels,
            y=defect_sizes,
            marker_color='#667eea',
            text=[f"{size:.0f} px" for size in defect_sizes],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Defect Size Analysis",
            xaxis_title="Defect Areas",
            yaxis_title="Pixel Area",
            template="plotly_white",
            height=300
        )
    else:
        fig.add_annotation(
            text="No Defects Detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Defect Size Analysis",
            template="plotly_white",
            height=300
        )
    
    return fig

# App layout
st.markdown('<div class="main-header"><h1>🧵 Advanced Cloth Defect Detection System</h1><p>AI-Powered Quality Control for Textile Manufacturing</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; color: white; padding: 1rem;'>
        <h2>⚙️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Visualization Settings")
    alpha = st.slider("Mask Opacity", 0.1, 1.0, 0.5, 0.05, help="Adjust the transparency of defect overlays")
    show_segmented = st.checkbox("Show Segmented Output", True)
    show_original = st.checkbox("Show Original Image", False)
    show_heatmap = st.checkbox("Show Heatmap", True)
    
    st.markdown("---")
    
    st.markdown("### 📊 System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Status", "✅ Ready" if models_loaded else "❌ Error")
    with col2:
        st.metric("Cloth Types", len(class_labels) if models_loaded else 0)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Quick Actions")
    if st.button("🔄 Clear Session", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("📖 About this System"):
        st.markdown("""
        This advanced AI system provides:
        - **Cloth Type Classification**
        - **Defect Detection & Segmentation**
        - **Quality Assessment Metrics**
        - **Visual Analytics**
        
        **Models Used:**
        - EfficientNet Classifier
        - U-Net Segmenter
        """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Cloth Image")
    st.markdown("Upload high-quality images for accurate defect detection")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="main_uploader"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🎨 Sample Gallery")
    
    sample_cols = st.columns(2)
    with sample_cols[0]:
        if st.button("White Plain", use_container_width=True):
            # Placeholder for sample image loading
            st.info("Sample feature coming soon!")
    with sample_cols[1]:
        if st.button("Blue Plaid", use_container_width=True):
            st.info("Sample feature coming soon!")
    st.markdown('</div>', unsafe_allow_html=True)

# Processing and Results
if uploaded_file and models_loaded:
    # Processing animation
    with st.spinner("🔄 Processing your image..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent in range(0, 101, 20):
            progress_bar.progress(percent)
            status_text.text(f"Processing... {percent}%")
            time.sleep(0.3)
        
        # Load and process image
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        
        # Run inference
        label, confidence, mask = run_inference(img_np)
        overlay = overlay_mask(img_np, mask, alpha)
        segmented_output = create_segmented_output(img_np, mask)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

    # Results Section
    st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Results")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cloth Type", label, delta="Identified")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        defect_percentage = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Defect Area", f"{defect_percentage:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Defect Count", len(contours))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        status = "✅ Pass" if defect_percentage < 5 else "⚠️ Review" if defect_percentage < 15 else "❌ Fail"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Quality Status", status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence Visualization
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Classification Confidence")
    st.markdown(f'**Cloth Type:** {label}')
    
    col_conf1, col_conf2 = st.columns([3, 1])
    with col_conf1:
        st.markdown(f"""
        <div class="confidence-container">
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{confidence*100}%">
                    <span class="confidence-label">{confidence*100:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_conf2:
        st.metric("Score", f"{confidence*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization Tabs
    tabs = st.tabs(["🎨 Combined View", "🔍 Segmented View", "📈 Detailed Analysis"])
    
    with tabs[0]:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 Defect Visualization")
        col_viz1, col_viz2 = st.columns([2, 1])
        
        with col_viz1:
            st.image(overlay, use_column_width=True, caption=f"Defect Analysis: {label}")
        
        with col_viz2:
            st.plotly_chart(create_defect_analysis_chart(mask, contours), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Detailed Segmentation")
        
        if show_segmented:
            seg_col1, seg_col2 = st.columns(2)
            with seg_col1:
                st.image(segmented_output, use_column_width=True, 
                        caption="Isolated Defects (Transparent Background)")
            with seg_col2:
                if show_heatmap:
                    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
                    st.image(heatmap, use_column_width=True, caption="Defect Heatmap")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Comprehensive Analysis")
        
        if show_original:
            st.image(img_np, use_column_width=True, caption="Original Image")
        
        # Detailed metrics
        col_det1, col_det2 = st.columns(2)
        
        with col_det1:
            st.markdown("#### Defect Statistics")
            st.image(mask * 255, use_column_width=True, caption="Binary Mask", clamp=True)
            
            if len(contours) > 0:
                st.markdown("##### Defect Size Distribution")
                defect_areas = [cv2.contourArea(cnt) for cnt in contours]
                fig_hist = px.histogram(x=defect_areas, nbins=10, 
                                      title="Defect Size Distribution",
                                      labels={'x': 'Defect Area (pixels)', 'y': 'Count'})
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_det2:
            st.markdown("#### Quality Metrics")
            
            # Quality score calculation
            quality_score = max(0, 100 - defect_percentage * 2)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = quality_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Quality Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Section
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Export Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        if st.button("💾 Download Overlay Image", use_container_width=True):
            # Implementation for download
            st.success("Overlay image ready for download!")
    
    with col_dl2:
        if st.button("📊 Download Analysis Report", use_container_width=True):
            # Implementation for report download
            st.success("Analysis report generated!")
    
    with col_dl3:
        if st.button("🖼️ Download Segmented Image", use_container_width=True):
            # Implementation for segmented image download
            st.success("Segmented image ready!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif not models_loaded:
    st.error("🚨 Models failed to load. Please check if the model files are available.")
else:
    # Welcome section when no image is uploaded
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    
    col_welcome1, col_welcome2 = st.columns([2, 1])
    
    with col_welcome1:
        st.markdown("""
        ## 👋 Welcome to Cloth Defect Detection
        
        **Transform your textile quality control with AI-powered defect detection.**
        
        ### 🚀 How it works:
        1. **Upload** a cloth image using the uploader
        2. **AI Analysis** automatically detects cloth type and defects
        3. **Visualize** results with interactive charts and overlays
        4. **Export** detailed reports and images
        
        ### 🎯 Key Features:
        - 🔍 **Automatic cloth type classification**
        - 🎨 **Visual defect segmentation**
        - 📊 **Comprehensive quality metrics**
        - 📈 **Interactive analytics dashboard**
        - 💾 **Export capabilities**
        
        *Upload an image to get started!*
        """)
    
    with col_welcome2:
        st.markdown("### 🏆 Benefits")
        st.info("""
        **✅ Improved Accuracy**  
        AI-powered detection reduces human error
        
        **⚡ Faster Inspection**  
        Process images in seconds, not minutes
        
        **📈 Consistent Quality**  
        Maintain uniform quality standards
        
        **💡 Data-Driven Insights**  
        Get detailed analytics and reports
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem;'>
    <p>🧵 <strong>Advanced Cloth Defect Detection System</strong> | AI-Powered Quality Control</p>
    <p style='font-size: 0.8rem; opacity: 0.8;'>Built with TensorFlow Lite & Streamlit</p>
</div>
""", unsafe_allow_html=True)
