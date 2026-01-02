import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from PIL import Image

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
# Load Pretrained Model
# -------------------------------
MODEL_PATH = "oilspill_unet_best.h5"
model = load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric
    }
)

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Oil Spill Detection Dashboard", layout="wide")
st.title("🌊 Oil Spill Detection using Deep Learning")

# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
threshold = st.sidebar.slider("Segmentation Threshold", 0.0, 1.0, 0.5)
predict_button = st.sidebar.button("🔍 Predict")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(img, threshold=0.5):
    # Preprocess image
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    pred = model.predict(img_array)           # shape (1,256,256,1)
    pred_mask = pred[0, :, :, 0]              # smooth probability mask
    pred_bin = (pred_mask > threshold).astype(np.uint8)  # binary mask

    # Oil spill classification
    has_spill = np.sum(pred_bin) > 0
    label = "Oil Spill" if has_spill else "No Oil Spill"

    # Oil spill percentage
    spill_percentage = (np.sum(pred_bin) / pred_bin.size) * 100

    # Confidence level (mean probability of predicted mask)
    confidence = float(np.mean(pred_mask))

    # Accuracy placeholder
    accuracy = "N/A (dataset-wide evaluation required)"

    return pred_mask, pred_bin, label, spill_percentage, confidence, accuracy

# -------------------------------
# Main Prediction Workflow
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if predict_button:
        pred_mask, pred_bin, label, spill_percentage, confidence, accuracy = predict_image(image, threshold)

        # Results section
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(pred_mask, caption="Predicted Probability Mask", use_column_width=True, clamp=True)
            st.image(pred_bin*255, caption="Thresholded Binary Mask", use_column_width=True)

        # Metrics
        st.markdown(f"**Prediction:** {label}")
        st.metric("Oil Spill %", f"{spill_percentage:.2f}%")
        st.metric("Confidence", f"{confidence:.2f}")
        st.markdown(f"**Accuracy:** {accuracy}")

        # Confidence progress bar
        st.progress(confidence)

        # Report text
        report = f"""
        Prediction Report
        -----------------
        Prediction: {label}
        Oil Spill Percentage: {spill_percentage:.2f}%
        Confidence Level: {confidence:.2f}
        Accuracy: {accuracy}
        Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """

        # Download report
        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name="prediction_report.txt",
            mime="text/plain"
        )

        # Download mask image
        mask_img = Image.fromarray((pred_bin*255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        st.download_button("🖼️ Download Mask Image", buf.getvalue(), "predicted_mask.png", "image/png")

        # Save history
        st.session_state.history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": label
        })

# -------------------------------
# History Section
# -------------------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)

# -------------------------------
# Model Info Section
# -------------------------------
with st.expander("ℹ️ Model Information"):
    st.write("Model: U-Net with ResNet50 backbone")
    st.write("Loss: BCE + Dice")
    st.write("Input size: 256x256")
    st.write("Trained on: Oil spill dataset v1")