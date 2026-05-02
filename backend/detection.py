"""
detection.py
============
Webcam Frame Capture + YOLO Inference Pipeline
-----------------------------------------------
This module manages the OpenCV webcam capture loop, calls the
YOLO model for inference, applies visualization helpers, and
exposes a generator that yields MJPEG-encoded frames for
Flask's streaming response.

Author  : AI Project Team
Course  : Artificial Intelligence
"""

import cv2
import time
import logging
import threading
import numpy as np
from backend.model.yolo_model import YOLOModel
from backend.utils.visualization import (
    draw_detections,
    draw_hud,
    encode_frame_to_jpeg,
    resize_frame,
)

logger = logging.getLogger(__name__)


class WebcamDetector:
    """
    Manages the webcam capture loop and YOLO inference in a separate
    background thread. Provides:
      - A `get_latest_frame()` method for the Flask streaming endpoint.
      - `start()` / `stop()` controls for the webcam session.
      - Access to the last set of detections for the /metrics API.

    Parameters
    ----------
    camera_index : int
        OpenCV camera index (0 = default webcam).
    model_name   : str
        YOLO model file name, e.g. 'yolov8n.pt'.
    conf_threshold : float
        Confidence threshold for detections.
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
    ):
        self.camera_index    = camera_index
        self.conf_threshold  = conf_threshold

        # ── Model ──────────────────────────────────────────────────────────
        self.model = YOLOModel(
            model_name=model_name,
            confidence_threshold=conf_threshold,
        )
        self.model.load_model()

        # ── State ──────────────────────────────────────────────────────────
        self._cap:           cv2.VideoCapture | None = None
        self._is_running:    bool  = False
        self._thread:        threading.Thread | None = None
        self._lock:          threading.Lock = threading.Lock()
        self._frame_event:   threading.Event = threading.Event()

        self._latest_frame:      np.ndarray | None = None
        self._latest_jpeg:       bytes | None = None  # pre-encoded JPEG
        self._latest_detections: list[dict]        = []
        self._frame_count:       int               = 0
        self._detection_history: list[dict]        = []   # last 50 frames

        # Performance tuning
        self._infer_every_n:     int = 2   # run YOLO every N frames
        self._target_width:      int = 640
        self._target_height:     int = 480

    # ─────────────────────────────────────────────────────────────────────
    # Start / Stop
    # ─────────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open the webcam and start the background capture thread.

        Returns
        -------
        bool : True if camera opened successfully.
        """
        if self._is_running:
            logger.info("Detector is already running.")
            return True

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("❌ Cannot open camera index %d.", self.camera_index)
            return False

        # Use a smaller resolution for faster YOLO inference on CPU
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        # Minimize internal OpenCV buffer to reduce lag
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("✅ Webcam capture started on camera %d.", self.camera_index)
        return True

    def stop(self) -> None:
        """Stop the capture loop and release the camera."""
        self._is_running = False
        self._frame_event.set()  # wake up any waiting generators
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("⛔ Webcam capture stopped.")

    # ─────────────────────────────────────────────────────────────────────
    # Background capture loop (runs in its own thread)
    # ─────────────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Continuously reads frames from the webcam, runs YOLO inference,
        draws annotations, and stores the result in `_latest_frame`.

        Optimization: YOLO inference only runs every `_infer_every_n` frames.
        In between, the last set of detections is re-drawn on the new frame
        for smooth video while keeping CPU usage low.
        """
        frame_idx = 0
        cached_detections = []

        while self._is_running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera.")
                time.sleep(0.01)
                continue

            # Resize to reduce inference time
            frame = resize_frame(frame, max_width=self._target_width, max_height=self._target_height)

            # ── YOLO Inference (skip frames for speed) ────────────────────
            if frame_idx % self._infer_every_n == 0:
                cached_detections = self.model.predict(frame)

            frame_idx += 1

            # ── Draw annotations using cached detections ──────────────────
            annotated = draw_detections(frame, cached_detections)
            annotated = draw_hud(
                annotated,
                detection_count=len(cached_detections),
                fps=self.model.get_fps(),
                inference_ms=self.model.get_average_inference_time() * 1000,
                model_name=self.model.model_name.replace(".pt", "").upper(),
            )

            # ── Encode JPEG once in the capture thread ────────────────────
            jpeg = encode_frame_to_jpeg(annotated, quality=65)

            # ── Thread-safe state update ──────────────────────────────────
            with self._lock:
                self._latest_frame      = annotated
                self._latest_jpeg       = jpeg
                self._latest_detections = cached_detections
                self._frame_count      += 1

                # Log detection summary for history (keep last 50)
                if cached_detections and frame_idx % self._infer_every_n == 1:
                    entry = {
                        "frame":      self._frame_count,
                        "count":      len(cached_detections),
                        "labels":     [d["label"] for d in cached_detections],
                        "avg_conf":   round(
                            sum(d["confidence"] for d in cached_detections) / len(cached_detections), 3
                        ),
                        "timestamp":  time.strftime("%H:%M:%S"),
                    }
                    self._detection_history.append(entry)
                    if len(self._detection_history) > 50:
                        self._detection_history.pop(0)

            # Signal that a new frame is available
            self._frame_event.set()

    # ─────────────────────────────────────────────────────────────────────
    # Public accessors
    # ─────────────────────────────────────────────────────────────────────

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recently annotated frame (thread-safe)."""
        with self._lock:
            return self._latest_frame

    def get_latest_jpeg(self) -> bytes | None:
        """Return the most recently encoded JPEG bytes (thread-safe)."""
        with self._lock:
            return self._latest_jpeg

    def get_latest_detections(self) -> list[dict]:
        """Return the detections from the most recent frame."""
        with self._lock:
            return list(self._latest_detections)

    def get_stats(self) -> dict:
        """Return a summary of detector performance statistics."""
        with self._lock:
            return {
                "is_running":        self._is_running,
                "frame_count":       self._frame_count,
                "fps":               self.model.get_fps(),
                "avg_inference_ms":  round(
                    self.model.get_average_inference_time() * 1000, 2
                ),
                "total_detections":  sum(
                    e["count"] for e in self._detection_history
                ),
                "model_info":        self.model.get_model_info(),
            }

    def get_detection_history(self) -> list[dict]:
        """Return the last 50 detection log entries."""
        with self._lock:
            return list(self._detection_history)

    def is_running(self) -> bool:
        """Return True while the capture loop is active."""
        return self._is_running

    def wait_for_frame(self, timeout: float = 0.1) -> bool:
        """Block until a new frame is available, or timeout."""
        result = self._frame_event.wait(timeout=timeout)
        self._frame_event.clear()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG Generator for Flask streaming
# ─────────────────────────────────────────────────────────────────────────────

def generate_mjpeg_stream(detector: WebcamDetector):
    """
    Generator that yields multipart JPEG frames for Flask's
    `Response(stream_with_context(...), mimetype='multipart/x-mixed-replace')`.

    Optimised: uses pre-encoded JPEG from the capture thread and event-based
    waiting instead of busy-polling with sleep().
    """
    BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while detector.is_running():
        # Wait for a new frame (event-driven, not busy-polling)
        detector.wait_for_frame(timeout=0.1)

        jpeg = detector.get_latest_jpeg()
        if jpeg is None:
            continue

        yield BOUNDARY + jpeg + b"\r\n"


def detect_on_image(image_path: str, model_name: str = "yolov8n.pt") -> dict:
    """
    Run single-image detection (for the /detect REST endpoint).

    Parameters
    ----------
    image_path : str  — path to image file
    model_name : str  — YOLO model to use

    Returns
    -------
    dict with 'detections', 'count', 'inference_ms', 'image_shape'
    """
    model = YOLOModel(model_name=model_name)
    model.load_model()

    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": f"Cannot read image: {image_path}"}

    t0 = time.perf_counter()
    detections = model.predict(frame)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Detections need JSON-serializable colours (tuple → list)
    for det in detections:
        det["color"] = list(det["color"])

    return {
        "detections":   detections,
        "count":        len(detections),
        "inference_ms": elapsed_ms,
        "image_shape":  list(frame.shape),
    }
