"""
core/signal_controller.py

Adaptive traffic signal controller.

State machine per intersection:
  GREEN (one signal at a time) → YELLOW → ALL_RED → GREEN (next)

Signal selection: signal with highest congestion / recommended green time
goes first; ties broken by round-robin.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from config.settings import SignalConfig
from core.congestion_analyser import ZoneCongestionReport

log = logging.getLogger(__name__)


class SignalState(Enum):
    RED    = auto()
    YELLOW = auto()
    GREEN  = auto()


@dataclass
class SignalStatus:
    signal_id: str
    state: SignalState
    remaining_s: float          # seconds left in current state
    recommended_green_s: float
    congestion_level: str


class SignalController:
    """
    Controls the traffic light cycle for all signals at one intersection.

    Update loop:
      1. Receive congestion reports.
      2. If no signal is currently green, select the most congested signal.
      3. Tick timers and advance state machine.
    """

    def __init__(
        self,
        signal_ids: List[str],
        sig_cfg: SignalConfig,
    ):
        self.cfg = sig_cfg
        self.signal_ids = list(signal_ids)

        # Current state per signal
        self._states: Dict[str, SignalState] = {
            sid: SignalState.RED for sid in signal_ids
        }
        self._timers: Dict[str, float] = {sid: 0.0 for sid in signal_ids}
        self._recommended: Dict[str, float] = {
            sid: sig_cfg.min_green for sid in signal_ids
        }

        self._active_green: Optional[str] = None   # currently-green signal
        self._last_tick: float = time.monotonic()

        # Bootstrap: give first signal a green
        if signal_ids:
            self._activate(signal_ids[0])

        log.info(
            f"SignalController: managing {len(signal_ids)} signals: "
            f"{signal_ids}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, reports: Dict[str, ZoneCongestionReport]
    ) -> Dict[str, SignalStatus]:
        """
        Should be called once per frame.
        Returns current status for all signals.
        """
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        # Update recommended green times from latest reports
        for sid, r in reports.items():
            self._recommended[sid] = r.recommended_green_s

        # Tick active signal
        if self._active_green:
            self._tick(self._active_green, dt, reports)

        return self._build_status(reports)

    def statuses(
        self, reports: Dict[str, ZoneCongestionReport]
    ) -> Dict[str, SignalStatus]:
        return self._build_status(reports)

    # ------------------------------------------------------------------
    # Internal state machine
    # ------------------------------------------------------------------

    def _tick(
        self,
        sid: str,
        dt: float,
        reports: Dict[str, ZoneCongestionReport],
    ) -> None:
        self._timers[sid] = max(0.0, self._timers[sid] - dt)

        if self._timers[sid] > 0:
            return  # still in current phase

        current = self._states[sid]

        if current == SignalState.GREEN:
            # Transition to YELLOW
            self._states[sid] = SignalState.YELLOW
            self._timers[sid] = self.cfg.yellow_time
            log.debug(f"[{sid}] GREEN → YELLOW ({self.cfg.yellow_time}s)")

        elif current == SignalState.YELLOW:
            # Transition to RED (all-red pause)
            self._states[sid] = SignalState.RED
            self._timers[sid] = self.cfg.red_time
            self._active_green = None
            log.debug(f"[{sid}] YELLOW → RED (all-red {self.cfg.red_time}s)")
            # Schedule next green
            self._schedule_next(sid, reports)

        elif current == SignalState.RED and self._active_green is None:
            # All-red expired → activate next
            next_sid = self._pick_next(sid, reports)
            self._activate(next_sid)

    def _activate(self, sid: str) -> None:
        green_time = self._recommended.get(sid, self.cfg.min_green)
        green_time = max(self.cfg.min_green, min(self.cfg.max_green, green_time))
        self._states[sid] = SignalState.GREEN
        self._timers[sid] = green_time
        self._active_green = sid
        log.info(
            f"[{sid}] → GREEN for {green_time:.1f}s"
        )

    def _schedule_next(
        self,
        current_sid: str,
        reports: Dict[str, ZoneCongestionReport],
    ) -> None:
        """Pick and set timer so we switch after all-red."""
        next_sid = self._pick_next(current_sid, reports)
        self._timers[current_sid] = self.cfg.red_time
        # Will be activated in next tick when timer expires

    def _pick_next(
        self,
        just_finished: str,
        reports: Dict[str, ZoneCongestionReport],
    ) -> str:
        """
        Choose next signal to go green.
        Strategy: highest recommended_green_s (i.e. most congested).
        """
        candidates = [sid for sid in self.signal_ids if sid != just_finished]
        if not candidates:
            return just_finished

        # Score = recommended green time
        def score(sid):
            r = reports.get(sid)
            return r.recommended_green_s if r else self.cfg.min_green

        return max(candidates, key=score)

    # ------------------------------------------------------------------

    def _build_status(
        self, reports: Dict[str, ZoneCongestionReport]
    ) -> Dict[str, SignalStatus]:
        out: Dict[str, SignalStatus] = {}
        for sid in self.signal_ids:
            r = reports.get(sid)
            out[sid] = SignalStatus(
                signal_id=sid,
                state=self._states[sid],
                remaining_s=round(self._timers[sid], 1),
                recommended_green_s=self._recommended.get(sid, self.cfg.min_green),
                congestion_level=r.level.name if r else "UNKNOWN",
            )
        return out
