"""
config/settings.py
Typed configuration loaded from YAML (or defaults).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    path: str = "yolov8n.pt"
    trt_engine_dir: str = "models/engines"
    trt_precision: str = "fp16"          # fp32 | fp16 | int8
    trt_workspace_gb: int = 4
    imgsz: int = 640
    conf_threshold: float = 0.40
    iou_threshold: float = 0.45
    device: str = "cuda"                 # cuda | cpu
    # Classes to track (COCO IDs): car=2, bus=5, truck=7, motorbike=3
    vehicle_classes: List[int] = field(
        default_factory=lambda: [2, 3, 5, 7]
    )


@dataclass
class ZoneConfig:
    """
    Defines one lane / approach zone as a polygon (list of [x, y] pairs).
    Coordinates are normalised 0-1 relative to frame size.
    """
    name: str = "zone"
    polygon: List[List[float]] = field(
        default_factory=lambda: [[0.0, 0.5], [0.5, 0.5],
                                  [0.5, 1.0], [0.0, 1.0]]
    )
    signal_id: str = "S1"


@dataclass
class SignalConfig:
    # Base green time in seconds
    min_green: float = 10.0
    max_green: float = 60.0
    yellow_time: float = 3.0
    red_time: float = 5.0        # Minimum all-red before switching
    # Vehicle density thresholds for scaling green time
    low_density_threshold: int = 3
    high_density_threshold: int = 15


@dataclass
class CongestionConfig:
    # Smoothing window (number of frames) for density EMA
    ema_alpha: float = 0.15
    # Density level boundaries (vehicles per zone)
    free_flow_max: int = 3
    moderate_max: int = 8
    congested_max: int = 15
    # >= congested_max + 1 → gridlock


@dataclass
class VideoConfig:
    skip_frames: int = 0          # 0 = process every frame
    resize_width: int = 0         # 0 = original width
    resize_height: int = 0        # 0 = original height
    output_fps: float = 25.0


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    zones: List[ZoneConfig] = field(default_factory=lambda: [
        ZoneConfig(name="North", polygon=[[0.0, 0.0], [0.5, 0.0],
                                           [0.5, 0.5], [0.0, 0.5]],
                   signal_id="S_NORTH"),
        ZoneConfig(name="South", polygon=[[0.5, 0.0], [1.0, 0.0],
                                           [1.0, 0.5], [0.5, 0.5]],
                   signal_id="S_SOUTH"),
        ZoneConfig(name="East",  polygon=[[0.0, 0.5], [0.5, 0.5],
                                           [0.5, 1.0], [0.0, 1.0]],
                   signal_id="S_EAST"),
        ZoneConfig(name="West",  polygon=[[0.5, 0.5], [1.0, 0.5],
                                           [1.0, 1.0], [0.5, 1.0]],
                   signal_id="S_WEST"),
    ])
    signal: SignalConfig = field(default_factory=SignalConfig)
    congestion: CongestionConfig = field(default_factory=CongestionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        p = Path(path)
        if not p.exists():
            log.warning(
                f"Config file '{path}' not found — using defaults."
            )
            return cls()

        with open(p) as f:
            data = yaml.safe_load(f) or {}

        inst = cls()

        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(inst.model, k):
                    setattr(inst.model, k, v)

        if "signal" in data:
            for k, v in data["signal"].items():
                if hasattr(inst.signal, k):
                    setattr(inst.signal, k, v)

        if "congestion" in data:
            for k, v in data["congestion"].items():
                if hasattr(inst.congestion, k):
                    setattr(inst.congestion, k, v)

        if "video" in data:
            for k, v in data["video"].items():
                if hasattr(inst.video, k):
                    setattr(inst.video, k, v)

        if "zones" in data:
            inst.zones = [
                ZoneConfig(
                    name=z.get("name", "zone"),
                    polygon=z.get("polygon", []),
                    signal_id=z.get("signal_id", "S"),
                )
                for z in data["zones"]
            ]

        log.info(f"Configuration loaded from '{path}'.")
        return inst
