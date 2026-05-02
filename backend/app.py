"""
app.py
======
Flask Backend — Real-Time Object Detection API
----------------------------------------------
Endpoints:
  GET  /                    — Health check
  GET  /video_feed          — MJPEG webcam stream
  POST /api/start           — Start webcam detection
  POST /api/stop            — Stop webcam detection
  GET  /api/status          — Detector status & stats
  POST /api/detect          — Detect objects in uploaded image
  GET  /api/metrics         — Current frame metrics
  GET  /api/graphs-data     — All graph data for frontend
  GET  /api/history         — Detection log history
  GET  /api/concepts        — AI concept explanations
  GET  /api/experiments     — Experiment comparison data

Author  : AI Project Team
Course  : Artificial Intelligence
"""

import os
import sys
import logging
import tempfile
import time

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.detection import WebcamDetector, generate_mjpeg_stream, detect_on_image
from backend.utils.metrics import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1_score,
    generate_training_curves,
    generate_precision_recall_curve,
    generate_bias_variance_data,
    generate_accuracy_vs_dataset_size,
    generate_confidence_distribution,
    generate_experiment_comparison,
    AI_CONCEPTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Flask app configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/video_feed": {"origins": "*"}})

# Global detector instance (one per server process)
detector: WebcamDetector | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "Real-Time Object Detection using YOLO",
        "version": "1.0.0",
        "status":  "running",
        "time":    time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Webcam stream (MJPEG)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/video_feed", methods=["GET"])
def video_feed():
    """
    MJPEG streaming endpoint.
    Point an <img> src at this URL to display the live annotated feed.
    """
    global detector
    if detector is None or not detector.is_running():
        # Return a placeholder frame if not started
        return jsonify({"error": "Detection not started. POST /api/start first."}), 400

    return Response(
        stream_with_context(generate_mjpeg_stream(detector)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Detector controls
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def start_detection():
    """Start the webcam capture + YOLO inference loop."""
    global detector
    try:
        data         = request.get_json(silent=True) or {}
        camera_index = int(data.get("camera_index", 0))
        model_name   = data.get("model_name", "yolov8n.pt")
        conf         = float(data.get("confidence", 0.45))

        if detector and detector.is_running():
            return jsonify({"message": "Already running.", "status": "running"})

        detector = WebcamDetector(
            camera_index=camera_index,
            model_name=model_name,
            conf_threshold=conf,
        )
        success = detector.start()

        if success:
            return jsonify({"message": "Detection started.", "status": "running"})
        else:
            return jsonify({"error": "Failed to open camera."}), 500

    except Exception as exc:
        logger.exception("Error starting detection: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stop", methods=["POST"])
def stop_detection():
    """Stop the webcam capture loop."""
    global detector
    if detector:
        detector.stop()
        return jsonify({"message": "Detection stopped.", "status": "stopped"})
    return jsonify({"message": "Detector was not running.", "status": "stopped"})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return detector running state and performance stats."""
    global detector
    if detector is None:
        return jsonify({"is_running": False, "message": "Not initialised."})
    return jsonify(detector.get_stats())


# ─────────────────────────────────────────────────────────────────────────────
# Image detection (upload)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/detect", methods=["POST"])
def detect_image():
    """
    Accept an image file upload and run YOLO detection on it.
    Returns bounding boxes, labels, and confidence scores as JSON.

    Form field: 'image'  (multipart/form-data)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # Save to a temporary file
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = detect_on_image(tmp_path)
        os.unlink(tmp_path)   # Clean up temp file
        return jsonify(result)

    except Exception as exc:
        logger.exception("Error in /api/detect: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (live frame)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Return evaluation metrics computed from the latest detections and
    simulated ground-truth data to demonstrate IoU, Precision, Recall.
    """
    global detector

    # Latest real detections (or empty list if not running)
    detections = detector.get_latest_detections() if detector else []

    # Simulated ground-truth for metric demonstration
    tp = max(len(detections), 1)
    fp = max(1, len(detections) // 3)
    fn = max(1, len(detections) // 4)

    precision = compute_precision(tp, fp)
    recall    = compute_recall(tp, fn)
    f1        = compute_f1_score(precision, recall)

    # IoU example with two synthetic boxes
    example_iou = compute_iou([50, 50, 200, 200], [80, 80, 220, 220])

    return jsonify({
        "current_detections": len(detections),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision":   precision,
        "recall":      recall,
        "f1_score":    f1,
        "example_iou": example_iou,
        "model_info":  detector.model.get_model_info() if detector else {},
    })


# ─────────────────────────────────────────────────────────────────────────────
# Graphs data
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/graphs-data", methods=["GET"])
def get_graphs_data():
    """
    Return all graph datasets for the frontend Dashboard page.
    Includes training curves, P-R curve, Bias-Variance, Accuracy vs Size,
    and Confidence Score Distribution.
    """
    return jsonify({
        "training_curves": {
            "good_fit":     generate_training_curves(50, "good_fit"),
            "overfitting":  generate_training_curves(50, "overfitting"),
            "underfitting": generate_training_curves(50, "underfitting"),
        },
        "precision_recall":       generate_precision_recall_curve(),
        "bias_variance":          generate_bias_variance_data(),
        "accuracy_vs_size":       generate_accuracy_vs_dataset_size(),
        "confidence_distribution": generate_confidence_distribution(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# History, concepts, experiments
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the last 50 detection log entries."""
    global detector
    if detector is None:
        return jsonify([])
    return jsonify(detector.get_detection_history())


@app.route("/api/concepts", methods=["GET"])
def get_concepts():
    """Return AI concept explanations (Bias, Variance, Overfitting, etc.)."""
    return jsonify(AI_CONCEPTS)


@app.route("/api/experiments", methods=["GET"])
def get_experiments():
    """Return comparison data for the three experimental scenarios."""
    return jsonify(generate_experiment_comparison())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("🚀 Starting Flask server on http://0.0.0.0:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,       # Set True for development
        threaded=True,     # Required for concurrent MJPEG streaming
    )
