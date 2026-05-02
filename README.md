# 🎯 Real-Time Object Detection using YOLO and Webcam with Model Evaluation

> **AI Academic Project** — Full-stack web application demonstrating real-time object detection using YOLOv8, Flask, and React.js, with comprehensive model evaluation and visualisation.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [How to Run](#how-to-run)  
5. [API Reference](#api-reference)  
6. [YOLO Explanation](#yolo-explanation)  
7. [Evaluation Metrics](#evaluation-metrics)  
8. [AI Concepts](#ai-concepts)  
9. [Experiments](#experiments)  

---

## 📌 Project Overview

This project implements a **Real-Time Object Detection system** using:

- **YOLOv8** (Ultralytics) pretrained on the MS COCO dataset (80 classes)
- **OpenCV** for webcam capture and image processing
- **Flask** REST API backend with MJPEG streaming
- **React.js** frontend with a modern dark-theme dashboard
- **Recharts** for interactive graphs (training curves, PR curve, bias-variance, etc.)

### Key Features
| Feature | Description |
|---|---|
| 🎥 Live Detection | Real-time webcam feed with bounding boxes |
| 📊 Metrics | IoU, Precision, Recall, F1, mAP |
| 📈 Graphs | 5 interactive visualisation charts |
| 🧪 Experiments | Underfitting / Good Fit / Overfitting comparison |
| 📘 Theory | YOLO explanation + AI concept cards |

---

## 📁 Project Structure

```
project-root/
│
├── run.py                          ← Start Flask server (entry point)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.js           ← Sidebar navigation
│   │   │   ├── VideoFeed.js        ← Webcam stream + controls
│   │   │   ├── Charts.js           ← All Recharts visualisations
│   │   ├── pages/
│   │   │   ├── Home.js             ← Landing page
│   │   │   ├── LiveDetection.js    ← Webcam detection page
│   │   │   ├── Dashboard.js        ← Metrics + graphs
│   │   │   ├── Experiments.js      ← Scenario comparisons
│   │   │   ├── About.js            ← Theory & concepts
│   │   ├── App.js                  ← Router setup
│   │   ├── index.js                ← React entry point
│   │   ├── index.css               ← Global design system
│   ├── package.json
│
├── backend/
│   ├── app.py                      ← Flask app + all API routes
│   ├── detection.py                ← Webcam capture + MJPEG stream
│   ├── model/
│   │   ├── yolo_model.py           ← YOLOv8 wrapper class
│   │   ├── weights/                ← Auto-downloaded model weights
│   ├── utils/
│   │   ├── metrics.py              ← IoU, Precision, Recall, mAP + graph data
│   │   ├── visualization.py        ← OpenCV drawing utilities
│   ├── requirements.txt
│
├── data/
│   ├── sample_images/              ← Test images for /api/detect
│   ├── annotations/                ← Ground-truth annotation files
│
├── notebooks/
│   ├── training_analysis.ipynb     ← Jupyter analysis notebook
│
├── results/
│   ├── graphs/                     ← Saved graph exports
│   ├── logs/                       ← Detection logs
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam (for live detection)

### Step 1 — Clone / Open Project
```bash
cd "C:\Users\HP\OneDrive\Desktop\AAI"
```

### Step 2 — Backend Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install Python dependencies
pip install -r backend/requirements.txt
```

> **Note:** The `ultralytics` package will automatically download the YOLOv8 nano weights (~6 MB) on first run.

### Step 3 — Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## 🚀 How to Run

Open **two terminals** in the project root.

### Terminal 1 — Start Backend (Flask)
```bash
# Activate virtual environment first
venv\Scripts\activate

python run.py
# OR
python -m backend.app
```
Backend will start at: **http://localhost:5000**

### Terminal 2 — Start Frontend (React)
```bash
cd frontend
npm start
```
Frontend will open at: **http://localhost:3000**

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/`                  | Health check |
| `GET`  | `/video_feed`        | MJPEG webcam stream |
| `POST` | `/api/start`         | Start detection `{model_name, confidence, camera_index}` |
| `POST` | `/api/stop`          | Stop detection |
| `GET`  | `/api/status`        | Detector stats (FPS, frame count) |
| `POST` | `/api/detect`        | Detect on uploaded image (form field: `image`) |
| `GET`  | `/api/metrics`       | Live precision, recall, F1, IoU |
| `GET`  | `/api/graphs-data`   | All graph datasets (training curves, PR curve, etc.) |
| `GET`  | `/api/history`       | Last 50 detection log entries |
| `GET`  | `/api/concepts`      | AI concept explanations |
| `GET`  | `/api/experiments`   | Experiment comparison data |

---

## 🤖 YOLO Explanation

**YOLO** (*You Only Look Once*) is a real-time object detection algorithm that treats detection as a **single regression problem**.

### How It Works
1. The image is divided into an **S × S grid** (e.g., 13×13 for small YOLO).
2. Each cell predicts **B bounding boxes**: `[x, y, w, h, confidence]`.
3. Each cell also predicts **C class probabilities**.
4. **Non-Maximum Suppression (NMS)** removes overlapping boxes.
5. Final output: detected objects with labels and confidence scores.

### Why YOLOv8?
- ✅ Anchor-free architecture (faster, simpler)
- ✅ State-of-the-art speed/accuracy balance
- ✅ Easy deployment via Ultralytics Python API
- ✅ Pretrained on COCO (80 classes, 330K images)

---

## 📊 Evaluation Metrics

### Intersection over Union (IoU)
```
IoU = Area(Pred ∩ GT) / Area(Pred ∪ GT)
```
Measures overlap between predicted and ground-truth bounding boxes.  
**IoU ≥ 0.5** is typically considered a True Positive.

### Precision
```
Precision = TP / (TP + FP)
```
"Of all detections made, how many were correct?"

### Recall
```
Recall = TP / (TP + FN)
```
"Of all real objects, how many did the model find?"

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean — balances precision and recall.

### Average Precision (AP)
Area under the Precision-Recall curve for a single class.

### mean Average Precision (mAP)
```
mAP = (1/N) × Σ AP_i
```
Standard benchmark metric for object detectors.

---

## 🧠 AI Concepts

### Bias
Error from oversimplified models. **High bias → Underfitting.**  
Both training and validation errors are high.

### Variance
Sensitivity to training data noise. **High variance → Overfitting.**  
Low training error, high validation error.

### Overfitting
Model memorises training data → fails on new data.  
**Remedies:** Dropout, L2 Regularisation, Early Stopping, Data Augmentation.

### Underfitting
Model too simple → poor performance everywhere.  
**Remedies:** More complex model, more epochs, feature engineering.

### Bias–Variance Tradeoff
As model complexity increases:
- Bias decreases (model learns more)
- Variance increases (model overfits)
- **Optimal point** = minimum total error

---

## 🧪 Experiments

| Scenario | Dataset Size | Accuracy | Behaviour |
|---|---|---|---|
| Underfitting | Small (100 samples)   | ~52% | High bias, fails to learn |
| Good Fit     | Medium (5K samples)   | ~87% | Balanced — generalises well |
| Overfitting  | Large (50K, overtrained) | Train 99% / Val 88% | Memorises training set |

---

## 👨‍💻 Authors & Acknowledgements

- **Framework:** Ultralytics YOLOv8
- **Dataset:** MS COCO 2017
- **UI Library:** Recharts, React Router v6
- **Fonts:** Inter, JetBrains Mono (Google Fonts)

---

*This project was created for academic evaluation. All code is original and well-commented for demonstration purposes.*
