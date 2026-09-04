import os
import io
import datetime

import numpy as np
import requests
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "oilspill_unet_best.h5"
IMG_SIZE = (256, 256)
API_URL = os.environ.get("SPILLGUARD_API_URL", "http://127.0.0.1:8000")


# -------------------------------
# Custom Loss & Metrics
# -------------------------------
def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# -------------------------------
# Load Pretrained Model (cached so it loads only once)
# -------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(
        MODEL_PATH,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_loss": dice_loss,
            "dice_coefficient": dice_coefficient,
            "iou_metric": iou_metric,
        },
    )


model = get_model()

# -------------------------------
# Streamlit UI Setup
# -------------------------------
#st.set_page_config(page_title="Oil Spill Detection Dashboard", layout="wide")
st.title("🌊 Oil Spill Detection using Deep Learning")

if model is None:
    st.error(
        f"Model file '{MODEL_PATH}' not found. "
        "Please place the trained model in the project root before running predictions."
    )
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01)
predict_button = st.sidebar.button("🔍 Predict")


# -------------------------------
# API helper functions
# -------------------------------
def api_save_record(filename, label, spill_percentage, confidence, threshold_used):
    payload = {
        "filename": filename,
        "prediction": label,
        "spill_percentage": float(spill_percentage),
        "confidence": float(confidence),
        "threshold": float(threshold_used),
    }
    try:
        r = requests.post(f"{API_URL}/predictions", json=payload, timeout=3)
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)


def api_get_records():
    try:
        r = requests.get(f"{API_URL}/predictions", timeout=3)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return [], str(e)


def api_clear_records():
    try:
        r = requests.delete(f"{API_URL}/predictions", timeout=3)
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)


# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(img: Image.Image, threshold: float = 0.5):
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    pred = model.predict(img_array, verbose=0)   # shape (1, H, W, 1)
    pred_mask = pred[0, :, :, 0]                  # probability mask
    pred_bin = (pred_mask > threshold).astype(np.uint8)

    has_spill = np.sum(pred_bin) > 0
    label = "Oil Spill" if has_spill else "No Oil Spill"

    spill_percentage = (np.sum(pred_bin) / pred_bin.size) * 100

    # Mean confidence over predicted-positive pixels (falls back to overall mean)
    if has_spill:
        confidence = float(np.mean(pred_mask[pred_bin == 1]))
    else:
        confidence = float(np.mean(pred_mask))

    return img_resized, pred_mask, pred_bin, label, spill_percentage, confidence


def make_overlay(base_img: Image.Image, pred_bin: np.ndarray, color=(255, 0, 0), alpha=0.45):
    """Overlay the binary mask in red on top of the base image."""
    base_rgba = base_img.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((pred_bin * 255).astype(np.uint8)).resize(base_rgba.size)

    color_layer = Image.new("RGBA", base_rgba.size, color + (0,))
    alpha_mask = mask_img.point(lambda p: int(p * alpha))
    color_layer.putalpha(alpha_mask)

    combined = Image.alpha_composite(base_rgba, color_layer)
    return combined.convert("RGB")


# -------------------------------
# Main Prediction Workflow
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if predict_button:
        resized_img, pred_mask, pred_bin, label, spill_percentage, confidence = predict_image(image, threshold)
        overlay_img = make_overlay(resized_img, pred_bin)

        # Results section
        st.subheader("Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(resized_img, caption="Input (resized)", use_container_width=True)
        with col2:
            st.image(pred_mask, caption="Predicted Probability Mask", use_container_width=True, clamp=True)
        with col3:
            st.image(overlay_img, caption="Oil Spill Overlay", use_container_width=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", label)
        m2.metric("Oil Spill Area", f"{spill_percentage:.2f}%")
        m3.metric("Confidence", f"{confidence:.2f}")

        st.progress(min(max(confidence, 0.0), 1.0))

        # Report text
        report = f"""Prediction Report
-----------------
File: {uploaded_file.name}
Prediction: {label}
Oil Spill Percentage: {spill_percentage:.2f}%
Confidence Level: {confidence:.2f}
Threshold Used: {threshold:.2f}
Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="📄 Download Prediction Report",
                data=report,
                file_name="prediction_report.txt",
                mime="text/plain",
            )
        with col_b:
            mask_img = Image.fromarray((pred_bin * 255).astype(np.uint8))
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            st.download_button("🖼️ Download Mask Image", buf.getvalue(), "predicted_mask.png", "image/png")

        # Save record via FastAPI
        ok, err = api_save_record(uploaded_file.name, label, spill_percentage, confidence, threshold)
        if ok:
            st.success("Prediction record saved.")
        else:
            st.warning(f"Could not save record to API ({err}). Is the FastAPI server running?")


# -------------------------------
# History Section (from FastAPI / SQLite)
# -------------------------------
st.subheader("📜 Prediction History")

records, err = api_get_records()
if err:
    st.info(
        "History unavailable — start the API server with:\n\n"
        "`uvicorn api:app --reload --port 8000`"
    )
elif records:
    df = pd.DataFrame(records)
    df = df[["id", "timestamp", "filename", "prediction", "spill_percentage", "confidence", "threshold"]]
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Clear History"):
        ok, err = api_clear_records()
        if ok:
            st.rerun()
        else:
            st.warning(f"Could not clear history: {err}")
else:
    st.caption("No predictions recorded yet.")


# -------------------------------
# Model Info Section
# -------------------------------
with st.expander("ℹ️ Model Information"):
    st.write("Model: U-Net (custom encoder-decoder)")
    st.write("Loss: BCE + Dice")
    st.write(f"Input size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    st.write("Trained on: Oil spill detection dataset (binary: oil spill vs. background)")
    st.write(f"API backend: {API_URL}")
