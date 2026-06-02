"""
core/congestion_analyser.py

Translates per-zone vehicle counts into congestion levels and
recommends green-time durations for each signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict

from config.settings import CongestionConfig, SignalConfig

log = logging.getLogger(__name__)


class CongestionLevel(Enum):
    FREE_FLOW  = auto()   # Few / no vehicles
    MODERATE   = auto()   # Building up
    CONGESTED  = auto()   # Heavy traffic
    GRIDLOCK   = auto()   # Severe — maximum green needed


@dataclass
class ZoneCongestionReport:
    signal_id: str
    zone_name: str
    raw_count: int
    smoothed_count: float
    level: CongestionLevel
    recommended_green_s: float   # seconds


class CongestionAnalyser:
    """
    Maps smoothed vehicle density to CongestionLevel and recommended
    green-time using a piecewise-linear interpolation between
    min_green and max_green.
    """

    LEVEL_COLORS = {
        CongestionLevel.FREE_FLOW:  "\033[92m",   # bright green
        CongestionLevel.MODERATE:   "\033[93m",   # yellow
        CongestionLevel.CONGESTED:  "\033[91m",   # red
        CongestionLevel.GRIDLOCK:   "\033[95m",   # magenta
    }
    RESET = "\033[0m"

    def __init__(self, cong_cfg: CongestionConfig, sig_cfg: SignalConfig):
        self.c = cong_cfg
        self.s = sig_cfg

    # ------------------------------------------------------------------

    def classify(self, count: float) -> CongestionLevel:
        if count <= self.c.free_flow_max:
            return CongestionLevel.FREE_FLOW
        if count <= self.c.moderate_max:
            return CongestionLevel.MODERATE
        if count <= self.c.congested_max:
            return CongestionLevel.CONGESTED
        return CongestionLevel.GRIDLOCK

    def recommend_green(self, count: float) -> float:
        """
        Piecewise linear: maps vehicle count → green duration.
        0 vehicles    → min_green
        high_density  → max_green (clamped)
        """
        high = float(self.c.congested_max)
        ratio = min(max(count / high, 0.0), 1.0)
        return self.s.min_green + ratio * (self.s.max_green - self.s.min_green)

    # ------------------------------------------------------------------

    def analyse(
        self,
        raw_counts: Dict[str, int],
        smoothed_counts: Dict[str, float],
        zone_names: Dict[str, str],   # signal_id → zone name
    ) -> Dict[str, ZoneCongestionReport]:
        """
        Returns a report for every signal.
        """
        reports: Dict[str, ZoneCongestionReport] = {}
        for sig_id, smooth in smoothed_counts.items():
            raw = raw_counts.get(sig_id, 0)
            level = self.classify(smooth)
            green = self.recommend_green(smooth)
            reports[sig_id] = ZoneCongestionReport(
                signal_id=sig_id,
                zone_name=zone_names.get(sig_id, sig_id),
                raw_count=raw,
                smoothed_count=round(smooth, 2),
                level=level,
                recommended_green_s=round(green, 1),
            )

        self._log_reports(reports)
        return reports

    def _log_reports(self, reports: Dict[str, ZoneCongestionReport]) -> None:
        for r in reports.values():
            color = self.LEVEL_COLORS.get(r.level, "")
            log.debug(
                f"  {color}[{r.zone_name}]{self.RESET} "
                f"raw={r.raw_count} smooth={r.smoothed_count:.1f} "
                f"level={r.level.name} green={r.recommended_green_s}s"
            )
