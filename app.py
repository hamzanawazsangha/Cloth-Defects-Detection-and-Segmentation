import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json

# --- Load TFLite Models ---
@st.cache_resource
def load_interpreters():
    cls_int = tf.lite.Interpreter(model_path="classifier.tflite")
    seg_int = tf.lite.Interpreter(model_path="segmenter.tflite")
    cls_int.allocate_tensors()
    seg_int.allocate_tensors()
    return cls_int, seg_int

# --- Load Class Labels ---
@st.cache_resource
def load_labels():
    with open("class_labels.json") as f:
        return json.load(f)

cls_int, seg_int = load_interpreters()
class_labels = load_labels()

# --- Preprocessing Helper ---
def preprocess(img, size):
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- Inference Function ---
def run_inference(img):
    # Classify
    cls_input = preprocess(img, (224, 224))
    cls_in = cls_int.get_input_details()
    cls_out = cls_int.get_output_details()
    cls_int.set_tensor(cls_in[0]['index'], cls_input)
    cls_int.invoke()
    cls_pred = cls_int.get_tensor(cls_out[0]['index'])[0]
    cls_idx = int(np.argmax(cls_pred))
    cls_label = class_labels[str(cls_idx)]
    cls_conf = float(np.max(cls_pred))

    # Segment
    seg_input = preprocess(img, (256, 256))
    seg_in = seg_int.get_input_details()
    seg_out = seg_int.get_output_details()
    seg_int.set_tensor(seg_in[0]['index'], seg_input)
    seg_int.invoke()
    seg_pred = seg_int.get_tensor(seg_out[0]['index'])[0]
    
    if seg_pred.shape[-1] == 1:
        seg_mask = seg_pred[:, :, 0]
    else:
        seg_mask = np.argmax(seg_pred, axis=-1)
    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    return cls_label, cls_conf, seg_mask

# --- Overlay Bounding Boxes ---
def overlay_mask(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return overlay

# --- Streamlit UI ---
st.title("🧵 Cloth Defects Detection and Segmentation")
uploaded = st.file_uploader("Upload a cloth image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    with st.spinner("Analyzing..."):
        label, confidence, mask = run_inference(img_np)
        overlay = overlay_mask(img_np, mask)

    st.subheader(f"Defect Class: {label} ({confidence:.2f})")
    st.image(overlay, caption="Detected Defects", use_column_width=True)
