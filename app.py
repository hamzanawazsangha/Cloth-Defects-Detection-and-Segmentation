import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json

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

cls_int, seg_int, seg_dtype, seg_scale, seg_zero_point = load_interpreters()
class_labels = load_labels()

def preprocess(img, size, dtype=tf.float32, scale=1.0, zero_point=0):
    img = cv2.resize(img, size)

    if dtype == np.uint8 or dtype == np.int8:
        # Normalize img from 0-255 uint8 to quantized int8/uint8 based on scale and zero_point
        img = img.astype(np.float32)
        img = img / 255.0  # normalize to 0-1 float

        # Quantize
        img = img / scale + zero_point
        img = np.round(img).astype(dtype)
    else:
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

def run_inference(img):
    # Classification (assuming float32 model)
    cls_input_details = cls_int.get_input_details()
    cls_dtype = cls_input_details[0]['dtype']
    cls_input = preprocess(img, (224, 224), dtype=cls_dtype)
    cls_int.set_tensor(cls_input_details[0]['index'], cls_input)
    cls_int.invoke()
    cls_pred = cls_int.get_tensor(cls_int.get_output_details()[0]['index'])[0]
    cls_idx = int(np.argmax(cls_pred))
    cls_label = class_labels[cls_idx]
    cls_conf = float(np.max(cls_pred))

    # Segmentation (quantized model)
    seg_input_details = seg_int.get_input_details()
    seg_input_dtype = seg_input_details[0]['dtype']
    seg_scale, seg_zero_point = seg_input_details[0]['quantization']
    seg_input = preprocess(img, (256, 256), dtype=seg_input_dtype, scale=seg_scale, zero_point=seg_zero_point)
    seg_int.set_tensor(seg_input_details[0]['index'], seg_input)
    seg_int.invoke()
    seg_pred = seg_int.get_tensor(seg_int.get_output_details()[0]['index'])[0]

    # For segmentation output, if output is quantized, dequantize it:
    seg_output_details = seg_int.get_output_details()[0]
    seg_out_scale, seg_out_zero_point = seg_output_details['quantization']
    if seg_output_details['dtype'] in [np.uint8, np.int8]:
        seg_pred = seg_out_scale * (seg_pred.astype(np.float32) - seg_out_zero_point)

    if seg_pred.shape[-1] == 1:
        seg_mask = seg_pred[:, :, 0]
    else:
        seg_mask = np.argmax(seg_pred, axis=-1)

    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    return cls_label, cls_conf, seg_mask

def overlay_mask(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return overlay

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
