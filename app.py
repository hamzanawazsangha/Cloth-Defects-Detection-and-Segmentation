import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json

# --- Load class names ---
:contentReference[oaicite:4]{index=4}
    :contentReference[oaicite:5]{index=5}

# --- Load TFLite interpreters ---
:contentReference[oaicite:6]{index=6}
cls_int.allocate_tensors()
:contentReference[oaicite:7]{index=7}
:contentReference[oaicite:8]{index=8}

:contentReference[oaicite:9]{index=9}
seg_int.allocate_tensors()
:contentReference[oaicite:10]{index=10}
:contentReference[oaicite:11]{index=11}

# --- Helpers ---
:contentReference[oaicite:12]{index=12}
    :contentReference[oaicite:13]{index=13}
    :contentReference[oaicite:14]{index=14}

:contentReference[oaicite:15]{index=15}
    # Classify
    :contentReference[oaicite:16]{index=16}
    :contentReference[oaicite:17]{index=17}
    cls_int.invoke()
    :contentReference[oaicite:18]{index=18}
    :contentReference[oaicite:19]{index=19}
    :contentReference[oaicite:20]{index=20}

    # Segment
    inp2 = preprocess(img, (256,256))
    seg_int.set_tensor(seg_in[0]['index'], inp2)
    seg_int.invoke()
    seg_pred = seg_int.get_tensor(seg_out[0]['index'])[0]
    if seg_pred.shape[-1] == 1:
        seg_mask = seg_pred[:,:,0]
    else:
        seg_mask = np.argmax(seg_pred, axis=-1)
    seg_mask = (seg_mask > 0.5).astype(np.uint8)
    return cls_label, cls_conf, seg_mask

:contentReference[oaicite:21]{index=21}
    :contentReference[oaicite:22]{index=22}
    :contentReference[oaicite:23]{index=23}
    :contentReference[oaicite:24]{index=24}
    :contentReference[oaicite:25]{index=25}
    # bounding box
    :contentReference[oaicite:26]{index=26}
    if cnts:
        :contentReference[oaicite:27]{index=27}
        :contentReference[oaicite:28]{index=28}
        :contentReference[oaicite:29]{index=29}
                    :contentReference[oaicite:30]{index=30}
    return overlay

# --- Streamlit UI ---
:contentReference[oaicite:31]{index=31}
:contentReference[oaicite:32]{index=32}

:contentReference[oaicite:33]{index=33}
    :contentReference[oaicite:34]{index=34}
    if uploaded:
        :contentReference[oaicite:35]{index=35}
        :contentReference[oaicite:36]{index=36}
        :contentReference[oaicite:37]{index=37}
        :contentReference[oaicite:38]{index=38}
        :contentReference[oaicite:39]{index=39}

else:
    :contentReference[oaicite:40]{index=40}
    :contentReference[oaicite:41]{index=41}
    :contentReference[oaicite:42]{index=42}
    :contentReference[oaicite:43]{index=43}
    while run:
        :contentReference[oaicite:44]{index=44}
        :contentReference[oaicite:45]{index=45}
            :contentReference[oaicite:46]{index=46}
            break
        :contentReference[oaicite:47]{index=47}
        :contentReference[oaicite:48]{index=48}
        :contentReference[oaicite:49]{index=49}
        :contentReference[oaicite:50]{index=50}
        :contentReference[oaicite:51]{index=51}
    else:
        camera.release()
