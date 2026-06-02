"""
core/annotator.py

Draws bounding boxes, zone overlays, congestion levels, and signal states
onto frames for display / output video.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from core.congestion_analyser import CongestionLevel, ZoneCongestionReport
from core.signal_controller import SignalState, SignalStatus
from models.detector import Detection


# Signal-state colours (BGR)
STATE_COLOR = {
    SignalState.GREEN:  (0, 220, 50),
    SignalState.YELLOW: (0, 220, 220),
    SignalState.RED:    (50, 50, 220),
}

LEVEL_COLOR = {
    CongestionLevel.FREE_FLOW: (50, 200, 50),
    CongestionLevel.MODERATE:  (30, 170, 230),
    CongestionLevel.CONGESTED: (30, 90, 240),
    CongestionLevel.GRIDLOCK:  (20, 20, 200),
}


class Annotator:
    """Stateless annotator — all methods modify frame in-place."""

    # ------------------------------------------------------------------
    # Bounding boxes
    # ------------------------------------------------------------------

    @staticmethod
    def draw_detections(
        frame: np.ndarray, detections: List[Detection]
    ) -> None:
        for det in detections:
            x1, y1, x2, y2 = (
                int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 100), -1
            )
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # HUD: signal states
    # ------------------------------------------------------------------

    @staticmethod
    def draw_signal_hud(
        frame: np.ndarray,
        statuses: Dict[str, SignalStatus],
        reports: Dict[str, ZoneCongestionReport],
    ) -> None:
        """Draws a compact signal panel in the top-right corner."""
        h, w = frame.shape[:2]
        panel_w = 220
        panel_h = 30 + len(statuses) * 56
        x0 = w - panel_w - 12
        y0 = 12

        # Panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x0 - 6, y0 - 6),
            (x0 + panel_w + 6, y0 + panel_h + 6),
            (20, 20, 20), -1
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(
            frame, "SIGNALS", (x0 + 60, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
        )

        for i, (sid, status) in enumerate(statuses.items()):
            yy = y0 + 30 + i * 56
            color = STATE_COLOR[status.state]
            report = reports.get(sid)

            # Circle indicator
            cv2.circle(frame, (x0 + 18, yy + 12), 12, color, -1)
            cv2.circle(frame, (x0 + 18, yy + 12), 12, (255, 255, 255), 1)

            # Zone name + state
            cv2.putText(
                frame,
                f"{report.zone_name if report else sid}",
                (x0 + 38, yy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"{status.state.name}  {status.remaining_s:.0f}s",
                (x0 + 38, yy + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )

            # Congestion level + vehicle count
            if report:
                lvl_color = LEVEL_COLOR.get(
                    CongestionLevel[status.congestion_level],
                    (200, 200, 200)
                ) if status.congestion_level in CongestionLevel.__members__ else (200, 200, 200)
                cv2.putText(
                    frame,
                    f"{report.raw_count} veh  {status.congestion_level}",
                    (x0 + 38, yy + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lvl_color, 1, cv2.LINE_AA
                )

    # ------------------------------------------------------------------
    # Stats bar at bottom
    # ------------------------------------------------------------------

    @staticmethod
    def draw_stats_bar(
        frame: np.ndarray,
        total_vehicles: int,
        inference_ms: float,
        frame_idx: int,
    ) -> None:
        h, w = frame.shape[:2]
        bar_h = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        text = (
            f"Frame: {frame_idx}  |  "
            f"Total vehicles: {total_vehicles}  |  "
            f"Inference: {inference_ms:.1f} ms"
        )
        cv2.putText(
            frame, text, (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
        )
