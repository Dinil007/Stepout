"""
Gait Analysis Module

Performs temporal gait cycle analysis from sequential BiomechanicsResult frames.
Computes step duration, stride duration, swing/stance phases, left-right balance,
and running consistency coefficient.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.pose.biomechanics import BiomechanicsResult

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Minimum frames needed to compute a valid gait report
MIN_FRAMES_REQUIRED: int = 15


@dataclass
class GaitReport:
    """Temporal gait cycle metrics for a single player."""
    track_id: int

    # Step & Stride Timing
    avg_step_duration_s: Optional[float] = None
    avg_stride_duration_s: Optional[float] = None
    step_count: int = 0

    # Phase Percentages
    swing_phase_pct: Optional[float] = None   # % of time foot is in the air
    stance_phase_pct: Optional[float] = None  # % of time foot is grounded

    # Left/Right Balance
    left_contact_pct: Optional[float] = None  # % of frames left foot grounded
    right_contact_pct: Optional[float] = None

    # Consistency
    cadence_consistency_cv: Optional[float] = None   # Coefficient of variation of step durations
    stride_length_cv: Optional[float] = None         # CV of stride lengths

    # Summary classification
    gait_pattern: str = "Unknown"   # "Balanced", "Left-Dominant", "Right-Dominant", "Irregular"

    def to_dict(self) -> Dict:
        return asdict(self)


class GaitAnalyzer:
    """
    Analyzes temporal gait patterns from sequential biomechanics frames.
    Accumulates per-track biomechanics history and computes gait reports.
    """

    def __init__(self, fps: float):
        """
        Initializes the GaitAnalyzer.

        Args:
            fps: Video frame rate.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive. Got: {fps}")

        self.fps = fps
        self.dt = 1.0 / fps

        # Frame history per track ID
        self._history: Dict[int, List[BiomechanicsResult]] = {}

    def update(self, result: BiomechanicsResult) -> None:
        """
        Ingests a new BiomechanicsResult for a player into the gait history buffer.

        Args:
            result: BiomechanicsResult from the BiomechanicsAnalyzer.
        """
        track_id = result.track_id
        self._history.setdefault(track_id, []).append(result)

    def _detect_step_events(
        self, history: List[BiomechanicsResult]
    ) -> Tuple[List[int], List[int]]:
        """
        Detects frame indices of foot-strike events (transition from swing to stance).

        Returns:
            (left_strike_frames, right_strike_frames)
        """
        left_strikes, right_strikes = [], []
        prev_left, prev_right = False, False

        for i, frame in enumerate(history):
            if frame.left_ground_contact and not prev_left:
                left_strikes.append(i)
            if frame.right_ground_contact and not prev_right:
                right_strikes.append(i)
            prev_left = frame.left_ground_contact
            prev_right = frame.right_ground_contact

        return left_strikes, right_strikes

    def _compute_contact_percentages(
        self, history: List[BiomechanicsResult]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Computes left/right contact percentages and swing/stance percentages.

        Returns:
            (left_contact_pct, right_contact_pct, stance_pct, swing_pct)
        """
        if not history:
            return None, None, None, None

        n = len(history)
        left_count = sum(1 for f in history if f.left_ground_contact)
        right_count = sum(1 for f in history if f.right_ground_contact)
        both_count = sum(1 for f in history if f.left_ground_contact or f.right_ground_contact)

        left_pct = round(left_count / n * 100, 1)
        right_pct = round(right_count / n * 100, 1)
        stance_pct = round(both_count / n * 100, 1)
        swing_pct = round((n - both_count) / n * 100, 1)

        return left_pct, right_pct, stance_pct, swing_pct

    def _coefficient_of_variation(self, values: List[float]) -> Optional[float]:
        """Returns coefficient of variation (std/mean * 100) as a consistency measure."""
        if len(values) < 2:
            return None
        mean = np.mean(values)
        if mean < 1e-6:
            return None
        cv = (np.std(values) / mean) * 100.0
        return round(float(cv), 2)

    def _classify_gait_pattern(
        self,
        left_contact_pct: Optional[float],
        right_contact_pct: Optional[float],
        cadence_cv: Optional[float]
    ) -> str:
        """Classifies the overall gait pattern."""
        if left_contact_pct is None or right_contact_pct is None:
            return "Unknown"
        diff = abs(left_contact_pct - right_contact_pct)
        if cadence_cv is not None and cadence_cv > 25.0:
            return "Irregular"
        if diff < 5.0:
            return "Balanced"
        elif left_contact_pct > right_contact_pct:
            return "Left-Dominant"
        else:
            return "Right-Dominant"

    def generate_report(self, track_id: int) -> GaitReport:
        """
        Generates a complete GaitReport for a tracked player from accumulated frames.

        Args:
            track_id: Player track ID.

        Returns:
            GaitReport with all gait metrics.
        """
        report = GaitReport(track_id=track_id)
        history = self._history.get(track_id, [])

        if len(history) < MIN_FRAMES_REQUIRED:
            logger.debug(f"Insufficient frames for gait analysis: track_id={track_id}")
            return report

        # Step event detection
        left_strikes, right_strikes = self._detect_step_events(history)
        all_strikes = sorted(left_strikes + right_strikes)
        report.step_count = len(all_strikes)

        # Step & stride duration
        if len(all_strikes) >= 2:
            step_intervals_s = [
                (all_strikes[i + 1] - all_strikes[i]) * self.dt
                for i in range(len(all_strikes) - 1)
            ]
            report.avg_step_duration_s = round(float(np.mean(step_intervals_s)), 3)
            report.avg_stride_duration_s = round(report.avg_step_duration_s * 2.0, 3)
            report.cadence_consistency_cv = self._coefficient_of_variation(step_intervals_s)

        # Contact percentages
        left_pct, right_pct, stance_pct, swing_pct = self._compute_contact_percentages(history)
        report.left_contact_pct = left_pct
        report.right_contact_pct = right_pct
        report.stance_phase_pct = stance_pct
        report.swing_phase_pct = swing_pct

        # Stride length consistency
        stride_lengths = [f.stride_length_norm for f in history if f.stride_length_norm is not None]
        if stride_lengths:
            report.stride_length_cv = self._coefficient_of_variation(stride_lengths)

        # Gait pattern classification
        report.gait_pattern = self._classify_gait_pattern(
            left_pct, right_pct, report.cadence_consistency_cv
        )

        return report

    def generate_all_reports(self) -> List[GaitReport]:
        """Generates GaitReports for all tracked players."""
        return [self.generate_report(tid) for tid in self._history.keys()]
