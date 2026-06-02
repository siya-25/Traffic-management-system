"""
core/zone_manager.py

Manages polygonal road zones.
For each frame, counts how many detected vehicles fall inside each zone.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import cv2

from config.settings import ZoneConfig
from models.detector import Detection

log = logging.getLogger(__name__)


class Zone:
    """Represents a single lane / approach zone."""

    def __init__(self, cfg: ZoneConfig, frame_w: int, frame_h: int):
        self.name = cfg.name
        self.signal_id = cfg.signal_id
        self.norm_polygon: List[List[float]] = cfg.polygon

        # Convert normalised polygon to pixel coords
        self.pixel_polygon: np.ndarray = np.array(
            [[int(x * frame_w), int(y * frame_h)] for x, y in cfg.polygon],
            dtype=np.int32,
        )
        self.vehicle_count: int = 0
        self.smoothed_count: float = 0.0     # EMA

    def contains_point(self, px: float, py: float) -> bool:
        """Test if pixel point (px, py) lies within the zone polygon."""
        return (
            cv2.pointPolygonTest(
                self.pixel_polygon, (float(px), float(py)), False
            )
            >= 0
        )

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int]) -> None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.pixel_polygon], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(
            frame, [self.pixel_polygon], True, color, 2, cv2.LINE_AA
        )
        # Zone label
        cx = int(self.pixel_polygon[:, 0].mean())
        cy = int(self.pixel_polygon[:, 1].mean())
        label = f"{self.name}: {self.vehicle_count}"
        cv2.putText(
            frame, label, (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )


class ZoneManager:
    """
    Owns all Zone objects.
    On each frame update, assigns detections to zones and updates counts.
    """

    # Colour palette for zones (BGR)
    ZONE_COLORS = [
        (0, 200, 80),    # Green
        (0, 120, 255),   # Orange-blue
        (180, 0, 200),   # Purple
        (255, 200, 0),   # Cyan-ish
        (0, 60, 200),    # Red-ish
        (200, 160, 0),   # Teal
    ]

    def __init__(self, zone_cfgs: List[ZoneConfig], frame_w: int, frame_h: int, ema_alpha: float = 0.15):
        self.ema_alpha = ema_alpha
        self.zones: List[Zone] = [
            Zone(cfg, frame_w, frame_h) for cfg in zone_cfgs
        ]
        log.info(
            f"ZoneManager: {len(self.zones)} zones initialised "
            f"({frame_w}x{frame_h})"
        )

    # ------------------------------------------------------------------

    def update(self, detections: List[Detection]) -> Dict[str, int]:
        """
        Assign each detection to the first matching zone.
        Returns mapping {signal_id: vehicle_count}.
        """
        # Reset raw counts
        for z in self.zones:
            z.vehicle_count = 0

        for det in detections:
            cx, cy = det.cx, det.cy
            for z in self.zones:
                if z.contains_point(cx, cy):
                    z.vehicle_count += 1
                    break  # one vehicle → one zone

        # Update EMA
        for z in self.zones:
            z.smoothed_count = (
                self.ema_alpha * z.vehicle_count
                + (1 - self.ema_alpha) * z.smoothed_count
            )

        return {z.signal_id: z.vehicle_count for z in self.zones}

    def smoothed_counts(self) -> Dict[str, float]:
        return {z.signal_id: z.smoothed_count for z in self.zones}

    # ------------------------------------------------------------------

    def draw_zones(self, frame: np.ndarray) -> None:
        for i, zone in enumerate(self.zones):
            color = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]
            zone.draw(frame, color)
