"""
yolo_model.py
=============
YOLO Model Integration for Real-Time Object Detection
-----------------------------------------------------
This module handles loading a pretrained YOLOv8 model and running
inference on video frames. It returns bounding boxes, class labels,
and confidence scores for each detected object.

Author  : AI Project Team
Course  : Artificial Intelligence
"""

import numpy as np
import cv2
import time
import logging

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COCO class names (80 classes) – used by YOLOv5/v8 pretrained on COCO
# ─────────────────────────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Assign a unique BGR colour to each class for bounding-box drawing
np.random.seed(42)
CLASS_COLORS = {cls: tuple(int(c) for c in np.random.randint(50, 255, 3))
                for cls in COCO_CLASSES}


class YOLOModel:
    """
    Wrapper around YOLOv8 (Ultralytics) for real-time object detection.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier, e.g. 'yolov8n.pt', 'yolov8s.pt'.
    confidence_threshold : float
        Minimum confidence score to keep a detection (0–1).
    iou_threshold : float
        NMS IoU threshold (0–1).
    device : str
        Inference device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None          # will be populated in load_model()
        self.is_loaded = False
        self.inference_times: list[float] = []   # rolling latency log

    # ─────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        """
        Load the YOLO model from Ultralytics.

        Tries to import the *ultralytics* package and download the weights
        automatically. Falls back to a simulated mode if the package is not
        installed, so the rest of the application still runs.

        Returns
        -------
        bool
            True if the real model was loaded; False if running in
            simulation mode.
        """
        try:
            from ultralytics import YOLO  # type: ignore
            logger.info("Loading YOLO model: %s on device=%s", self.model_name, self.device)
            self.model = YOLO(self.model_name)
            self.is_loaded = True
            logger.info("✅ YOLO model loaded successfully.")
            return True
        except ImportError:
            logger.warning(
                "⚠️  'ultralytics' package not found. "
                "Running in SIMULATION mode (random bounding boxes). "
                "Install with: pip install ultralytics"
            )
            self.is_loaded = False
            return False
        except Exception as exc:
            logger.error("❌ Failed to load YOLO model: %s", exc)
            self.is_loaded = False
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Run object detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3) as returned by cv2.VideoCapture.read().

        Returns
        -------
        list[dict]
            Each dict contains:
              - bbox        : [x1, y1, x2, y2]  (pixel coords)
              - label       : str  (class name)
              - confidence  : float (0–1)
              - class_id    : int
              - color       : (B, G, R) tuple
        """
        if frame is None or frame.size == 0:
            return []

        t_start = time.perf_counter()

        if self.is_loaded:
            detections = self._real_inference(frame)
        else:
            detections = self._simulated_inference(frame)

        elapsed = time.perf_counter() - t_start
        self.inference_times.append(elapsed)
        # Keep only the last 100 measurements
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        return detections

    def _real_inference(self, frame: np.ndarray) -> list[dict]:
        """Run actual YOLOv8 inference via the ultralytics API."""
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=320,         # smaller input = much faster on CPU
            verbose=False,
            device=self.device,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = (
                    result.names[cls_id]
                    if cls_id < len(result.names)
                    else f"class_{cls_id}"
                )
                detections.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "label":      label,
                    "confidence": round(conf, 3),
                    "class_id":   cls_id,
                    "color":      CLASS_COLORS.get(label, (0, 255, 0)),
                })

        return detections

    def _simulated_inference(self, frame: np.ndarray) -> list[dict]:
        """
        Generate realistic-looking random detections for demonstration
        when the ultralytics package is unavailable.
        """
        h, w = frame.shape[:2]
        num_detections = np.random.randint(0, 4)
        detections = []

        demo_classes = ["person", "car", "dog", "laptop", "cell phone", "chair"]

        for _ in range(num_detections):
            label   = np.random.choice(demo_classes)
            conf    = round(float(np.random.uniform(0.50, 0.95)), 3)
            x1 = np.random.randint(0, w - 100)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(60, min(200, w - x1))
            y2 = y1 + np.random.randint(60, min(200, h - y1))

            detections.append({
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                "label":      label,
                "confidence": conf,
                "class_id":   COCO_CLASSES.index(label) if label in COCO_CLASSES else 0,
                "color":      CLASS_COLORS.get(label, (0, 255, 0)),
            })

        return detections

    # ─────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ─────────────────────────────────────────────────────────────────────

    def get_average_inference_time(self) -> float:
        """Return average inference latency (seconds) over recent frames."""
        if not self.inference_times:
            return 0.0
        return round(sum(self.inference_times) / len(self.inference_times), 4)

    def get_fps(self) -> float:
        """Estimate frames-per-second from average inference time."""
        avg = self.get_average_inference_time()
        return round(1.0 / avg, 2) if avg > 0 else 0.0

    def update_thresholds(
        self,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> None:
        """Dynamically update confidence / IoU thresholds at runtime."""
        if confidence is not None:
            self.confidence_threshold = max(0.01, min(1.0, confidence))
        if iou is not None:
            self.iou_threshold = max(0.01, min(1.0, iou))
        logger.info(
            "Thresholds updated → conf=%.2f, iou=%.2f",
            self.confidence_threshold,
            self.iou_threshold,
        )

    def get_model_info(self) -> dict:
        """Return a summary dict of the current model configuration."""
        return {
            "model_name":           self.model_name,
            "is_loaded":            self.is_loaded,
            "device":               self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold":        self.iou_threshold,
            "avg_inference_ms":     round(self.get_average_inference_time() * 1000, 2),
            "estimated_fps":        self.get_fps(),
            "num_classes":          len(COCO_CLASSES),
        }
