"""
Ball Trajectory Interpolation Module

Post-processing step applied AFTER BallTracker output.
Fills gaps in ball trajectory using configurable linear interpolation.

Configurable parameters:
  - max_gap: Maximum number of consecutive missing frames to interpolate (default: 20)
  - method: Interpolation method (default: 'linear')

Does NOT modify BallTracker or detection pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BallInterpolator:
    """Post-hoc interpolation for ball trajectory gaps.

    Takes the per-frame ball output from BallTracker and fills missing
    positions using linear interpolation up to a configurable gap limit.
    """

    def __init__(self, max_gap: int = 20, method: str = "linear") -> None:
        """
        Args:
            max_gap: Maximum consecutive missing frames to interpolate.
                     Gaps longer than this remain as NaN / missing.
            method: Interpolation method (passed to pandas.DataFrame.interpolate).
                    Options: 'linear', 'time', 'quadratic', 'spline', etc.
        """
        self.max_gap = max_gap
        self.method = method
        self._stats: Dict = {}

    @property
    def max_gap(self) -> int:
        return self._max_gap

    @max_gap.setter
    def max_gap(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"max_gap must be >= 0, got {value}")
        self._max_gap = value

    def interpolate(
        self,
        ball_history: List[Dict],
        total_frames: int,
    ) -> List[Dict]:
        """Interpolate ball trajectory.

        Args:
            ball_history: List of per-frame ball dicts from BallTracker.
                          Each dict should have 'frame', 'center' (tuple x,y),
                          and optionally 'is_predicted', 'confidence'.
            total_frames: Total number of frames in the video.

        Returns:
            List of dicts with interpolated positions for ALL frames.
            Each dict has: frame, center_x, center_y, is_interpolated, confidence
        """
        if not ball_history:
            logger.warning("Empty ball history, nothing to interpolate")
            return []

        # Build DataFrame with all frames
        frame_map = {}
        for entry in ball_history:
            f = entry.get("frame", 0)
            center = entry.get("center", (None, None))
            frame_map[f] = {
                "cx": float(center[0]) if center[0] is not None else np.nan,
                "cy": float(center[1]) if center[1] is not None else np.nan,
                "confidence": float(entry.get("confidence", 0.0)),
                "was_predicted": bool(entry.get("is_predicted", False)),
            }

        df = pd.DataFrame(index=range(1, total_frames + 1))
        df["cx_raw"] = np.nan
        df["cy_raw"] = np.nan
        df["confidence"] = 0.0
        df["was_predicted"] = False

        for f, data in frame_map.items():
            if f <= total_frames:
                df.at[f, "cx_raw"] = data["cx"]
                df.at[f, "cy_raw"] = data["cy"]
                df.at[f, "confidence"] = data["confidence"]
                df.at[f, "was_predicted"] = data["was_predicted"]

        # Track which frames were originally detected (not predicted)
        df["was_detected"] = df["cx_raw"].notna() & ~df["was_predicted"]

        # Interpolate
        df["cx_interp"] = df["cx_raw"].interpolate(
            method=self.method, limit=self.max_gap
        )
        df["cy_interp"] = df["cy_raw"].interpolate(
            method=self.method, limit=self.max_gap
        )

        # Forward fill small gaps (1-2 frames)
        df["cx_interp"] = df["cx_interp"].fillna(method="ffill", limit=2)
        df["cy_interp"] = df["cy_interp"].fillna(method="ffill", limit=2)

        # Determine if frame is interpolated (newly filled by interpolator)
        # NOT detected by YOLO AND NOT predicted by BallTracker
        df["is_interpolated"] = (
            df["cx_interp"].notna()
            & ~df["was_detected"]
            & ~df["was_predicted"]
        )

        # Collect statistics
        self._compute_stats(df)

        # Build output
        output = []
        for idx, row in df.iterrows():
            cx = row["cx_interp"]
            cy = row["cy_interp"]
            if pd.notna(cx) and pd.notna(cy):
                output.append({
                    "frame": int(idx),
                    "center_x": round(float(cx), 1),
                    "center_y": round(float(cy), 1),
                    "is_interpolated": bool(row["is_interpolated"]),
                    "was_predicted": bool(row["was_predicted"]),
                    "was_detected": bool(row["was_detected"]),
                    "confidence": round(float(row["confidence"]), 4),
                })

        self._stats["output_frames"] = len(output)
        logger.info(
            f"BallInterpolator: {len(output)}/{total_frames} frames filled "
            f"(gap={self.max_gap})"
        )
        return output

    def _compute_stats(self, df: pd.DataFrame) -> None:
        """Compute interpolation statistics."""
        total = len(df)
        detected = int(df["was_detected"].sum())
        predicted = int(df["was_predicted"].sum())
        interpolated = int(df["is_interpolated"].sum())
        missing = int((~df["was_detected"] & ~df["is_interpolated"] & ~df["was_predicted"]).sum())

        # Track longest missing gap AFTER interpolation
        longest_gap = 0
        current_gap = 0
        for _, row in df.iterrows():
            if row["is_interpolated"] or row["was_detected"] or row["was_predicted"]:
                longest_gap = max(longest_gap, current_gap)
                current_gap = 0
            else:
                current_gap += 1
        longest_gap = max(longest_gap, current_gap)

        # Track gaps by length
        gaps = []
        current = 0
        for _, row in df.iterrows():
            if row["is_interpolated"] or row["was_detected"] or row["was_predicted"]:
                if current > 0:
                    gaps.append(current)
                current = 0
            else:
                current += 1
        if current > 0:
            gaps.append(current)

        self._stats = {
            "total_frames": total,
            "detected_frames": detected,
            "predicted_frames": predicted,
            "interpolated_frames": interpolated,
            "missing_frames": missing,
            "coverage_pct": round((detected + predicted + interpolated) / max(total, 1) * 100, 2),
            "longest_missing_gap": longest_gap,
            "average_missing_gap": round(float(np.mean(gaps)), 1) if gaps else 0.0,
            "num_gaps": len(gaps),
            "interpolation_max_gap_used": self.max_gap,
        }

    def get_stats(self) -> Dict:
        """Return interpolation statistics as dict."""
        return dict(self._stats)

    def check_unrealistic_jumps(
        self, trajectory: List[Dict], max_jump_px: float = 200.0
    ) -> List[Dict]:
        """Detect unrealistic position jumps in interpolated trajectory.

        Args:
            trajectory: List of per-frame ball dicts with center_x, center_y
            max_jump_px: Maximum plausible pixel displacement between frames.

        Returns:
            List of detected jumps with frame and displacement info.
        """
        jumps = []
        prev = None
        for entry in trajectory:
            cx, cy = entry.get("center_x"), entry.get("center_y")
            if cx is None or cy is None:
                prev = None
                continue
            if prev is not None:
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist = np.hypot(dx, dy)
                if dist > max_jump_px:
                    jumps.append({
                        "frame": entry.get("frame"),
                        "displacement_px": round(float(dist), 1),
                        "from": (round(prev[0], 1), round(prev[1], 1)),
                        "to": (round(float(cx), 1), round(float(cy), 1)),
                        "is_interpolated": entry.get("is_interpolated", False),
                    })
            prev = (cx, cy)

        self._stats["unrealistic_jumps"] = len(jumps)
        self._stats["unrealistic_jump_details"] = jumps[:20]  # first 20
        return jumps