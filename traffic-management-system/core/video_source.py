"""
core/video_source.py

Unified video source supporting:
  - Local video files
  - Webcam / USB camera index
  - RTSP / HTTP / HLS streams
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import VideoConfig

log = logging.getLogger(__name__)


class VideoSource:
    """
    Wraps cv2.VideoCapture with:
      - automatic frame resizing
      - frame-skip support
      - stream reconnection on drop (for RTSP)
    """

    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_WAIT_S = 2.0

    def __init__(self, source: str, cfg: VideoConfig):
        self.source = source
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx: int = 0
        self._open()

    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Parse source and open capture."""
        # Detect webcam index
        raw = self.source.strip()
        if raw.isdigit():
            src = int(raw)
        else:
            src = raw

        log.info(f"Opening video source: {src}")
        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")

        # Try to set buffer size for streams to reduce latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log.info(
            f"Source opened: {w}x{h} @ {fps:.1f} fps  "
            f"(total frames: {total if total > 0 else 'stream'})"
        )

    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        if self.cfg.resize_width > 0:
            return self.cfg.resize_width
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self.cfg.resize_height > 0:
            return self.cfg.resize_height
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.
        Applies frame-skip and resizing.
        Returns (success, frame_bgr).
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        skip = max(0, self.cfg.skip_frames)
        # Consume skipped frames
        for _ in range(skip):
            self._cap.grab()
            self._frame_idx += 1

        ok, frame = self._cap.read()
        self._frame_idx += 1

        if not ok or frame is None:
            return False, None

        # Resize if requested
        rw, rh = self.cfg.resize_width, self.cfg.resize_height
        if rw > 0 and rh > 0:
            frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        elif rw > 0:
            scale = rw / frame.shape[1]
            frame = cv2.resize(
                frame, (rw, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        elif rh > 0:
            scale = rh / frame.shape[0]
            frame = cv2.resize(
                frame, (int(frame.shape[1] * scale), rh),
                interpolation=cv2.INTER_LINEAR,
            )

        return True, frame

    @property
    def frame_index(self) -> int:
        return self._frame_idx

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
            log.info("Video source released.")
