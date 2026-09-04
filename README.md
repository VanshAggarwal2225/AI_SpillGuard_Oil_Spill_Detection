# 🌊 Oil Spill Detection using Deep Learning

## 📌 Overview
This repository hosts my work on oil spill detection using semantic segmentation with a U‑Net model (ResNet50 backbone). The project leverages deep learning to identify and segment oil spill regions from aerial or satellite images, providing a practical tool for environmental monitoring and decision support.

## 🎯 Features
- Streamlit Dashboard for interactive predictions
- Image Upload (JPG/PNG) with real‑time segmentation
- Threshold Control to adjust sensitivity of detection
- Prediction Results including:
- Original image
- Probability mask
- Thresholded binary mask
- Metrics Panel: Oil spill percentage, confidence score, classification label
- Download Options: Export prediction report and mask image
- Prediction History: Track past runs with timestamps
  
## 🛠️ Tech Stack
- Python (TensorFlow, Keras, NumPy, Pandas, Matplotlib, FastApi, SqlModel)
- Streamlit for user interface
- U‑Net with ResNet50 backbone for segmentation
- Custom Loss Functions: BCE + Dice, IoU metric

## 📂 Dataset
We used the **Oil Spill Detection Dataset** for training and evaluation.  
You can download it here:[Dataset Link](https://zenodo.org/records/10555314)

- Contains aerial/satellite images
- Includes pixel-level annotations for oil spill regions
- Suitable for semantic segmentation tasks

## 🚀 Getting Started
- Clone the repository:
git clone https://github.com/VanshAggarwal2225/AI_SpillGuard_Oil_Spill_Detection
  
- Install dependencies:
pip install -r requirements.txt

- Run the Streamlit app:
streamlit run app.py

## 📊 Results
- Accurate segmentation of oil spill regions
- Interactive threshold tuning for precision/recall balance
- Downloadable reports for documentation and analysis



