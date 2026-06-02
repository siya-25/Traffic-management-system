"""
models/detector.py

Wraps Ultralytics YOLO for both PyTorch (.pt) and TensorRT (.engine) backends.
Returns structured DetectionResult objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single detected vehicle bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class VehicleDetector:
    """
    Thin wrapper around Ultralytics YOLO that filters results to
    vehicle classes and returns DetectionResult.

    Supports:
      - Standard PyTorch model  (.pt)
      - TensorRT engine         (.engine)  — loaded via Ultralytics TRT backend
    """

    COCO_NAMES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, cfg):
        from config.settings import ModelConfig
        self.cfg: ModelConfig = cfg
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from ultralytics import YOLO

        model_path = Path(self.cfg.path)
        if not model_path.exists():
            log.warning(
                f"Model file '{model_path}' not found. "
                "Ultralytics will attempt to download it."
            )

        log.info(f"Loading model: {model_path}  device={self.cfg.device}")
        self._model = YOLO(str(model_path))
        # Warm-up
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        self._model(
            dummy,
            device=self.cfg.device,
            verbose=False,
            classes=self.cfg.vehicle_classes,
        )
        log.info("Model warm-up complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single BGR frame (H, W, 3 uint8).
        Returns DetectionResult with normalised pixel-space boxes.
        """
        import time

        t0 = time.perf_counter()
        results = self._model(
            frame,
            device=self.cfg.device,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            classes=self.cfg.vehicle_classes,
            imgsz=self.cfg.imgsz,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                detections.append(
                    Detection(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.COCO_NAMES.get(cls_id, str(cls_id)),
                    )
                )

        return DetectionResult(detections=detections, inference_ms=inference_ms)
