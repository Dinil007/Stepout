from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_types import (
    PressureEvent,
    PressingDetection,
    PressingSequence,
    PressingZone,
)

logger = logging.getLogger(__name__)


@dataclass
class PressingDetector:
    """Detects pressure events and pressing sequences from tracked players.

    This module only detects pressing actions from player tracking data.
    It does not compute PPDA, aggregate metrics, or produce visualizations.

    Attributes:
        config: Configuration parameters for pressing analysis.
    """

    config: PressingConfig

    def detect_pressure_events(
        self,
        attackers: Sequence[tuple[float, float, float, float]],
        defenders: Sequence[tuple[float, float, float, float]],
        frame_number: int,
        timestamp: float,
    ) -> list[PressureEvent]:
        """Detect individual pressure events in a single frame.

        Args:
            attackers: Sequence of (x, y, vx, vy) for attacking players.
            defenders: Sequence of (x, y, vx, vy) for defending players.
            frame_number: Current frame number.
            timestamp: Current timestamp in seconds.

        Returns:
            List of PressureEvent instances.
        """
        events: list[PressureEvent] = []
        if not attackers or not defenders:
            return events

        att_array = np.array(attackers, dtype=float)
        def_array = np.array(defenders, dtype=float)

        if att_array.shape[1] < 2 or def_array.shape[1] < 2:
            return events

        att_xy = att_array[:, :2]
        def_xy = def_array[:, :2]
        att_v = att_array[:, 2:4] if att_array.shape[1] >= 4 else np.zeros((att_array.shape[0], 2))
        def_v = def_array[:, 2:4] if def_array.shape[1] >= 4 else np.zeros((def_array.shape[0], 2))

        for a_idx, (a_pos, a_vel) in enumerate(zip(att_xy, att_v)):
            for d_idx, (d_pos, d_vel) in enumerate(zip(def_xy, def_v)):
                distance = float(np.linalg.norm(a_pos - d_pos) * 100.0)
                if distance > self.config.pressure_distance_threshold:
                    continue

                rel_vel = a_vel - d_vel
                closing_speed = float(np.linalg.norm(rel_vel) * 50.0)
                if closing_speed < self.config.minimum_closing_speed:
                    continue

                dx = float(d_pos[0] - a_pos[0])
                dy = float(d_pos[1] - a_pos[1])
                angle = float(math.atan2(dy, dx))

                events.append(
                    PressureEvent(
                        attacker_id=a_idx,
                        defender_id=d_idx,
                        team_id=0,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        distance=distance,
                        closing_speed=closing_speed,
                        pressure_angle=angle,
                        successful=False,
                    )
                )
        return events

    def detect_pressing_sequences(
        self,
        events_by_frame: dict[int, list[PressureEvent]],
        frame_rate: float = 25.0,
    ) -> list[PressingSequence]:
        """Group pressure events into continuous sequences.

        Args:
            events_by_frame: Mapping from frame number to pressure events.
            frame_rate: Frames per second for duration calculation.

        Returns:
            List of PressingSequence instances.
        """
        sequences: list[PressingSequence] = []
        if not events_by_frame:
            return sequences

        sorted_frames = sorted(events_by_frame.keys())
        current_events: list[PressureEvent] = []
        start_frame = sorted_frames[0]
        end_frame = sorted_frames[0]
        seq_id = 0

        for frame in sorted_frames:
            if not current_events:
                current_events = list(events_by_frame[frame])
                start_frame = frame
                end_frame = frame
                continue
            if frame - end_frame <= 1:
                current_events.extend(events_by_frame[frame])
                end_frame = frame
            else:
                duration = (end_frame - start_frame) / frame_rate if frame_rate > 0 else 0.0
                if duration >= self.config.minimum_pressure_duration and current_events:
                    ts_start = current_events[0].timestamp
                    ts_end = current_events[-1].timestamp
                    sequences.append(
                        PressingSequence(
                            sequence_id=seq_id,
                            team_id=current_events[0].team_id,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            start_time=ts_start,
                            end_time=ts_end,
                            pressure_events=list(current_events),
                            duration_seconds=duration,
                        )
                    )
                    seq_id += 1
                current_events = list(events_by_frame[frame])
                start_frame = frame
                end_frame = frame

        if current_events:
            duration = (end_frame - start_frame) / frame_rate if frame_rate > 0 else 0.0
            if duration >= self.config.minimum_pressure_duration:
                ts_start = current_events[0].timestamp
                ts_end = current_events[-1].timestamp
                sequences.append(
                    PressingSequence(
                        sequence_id=seq_id,
                        team_id=current_events[0].team_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_time=ts_start,
                        end_time=ts_end,
                        pressure_events=list(current_events),
                        duration_seconds=duration,
                    )
                )
        return sequences

    def classify_pressing_zone(self, y: float) -> PressingZone:
        """Classify pressing zone based on pitch coordinate.

        Args:
            y: Normalized Y coordinate (0-1).

        Returns:
            PressingZone enum value.
        """
        if y <= self.config.high_press_line_y:
            return PressingZone.HIGH_PRESS
        if y <= self.config.mid_block_line_y:
            return PressingZone.MID_BLOCK
        if y <= self.config.low_block_line_y:
            return PressingZone.LOW_BLOCK
        return PressingZone.LOW_BLOCK

    def calculate_confidence(self, events: Sequence[PressureEvent], sequences: Sequence[PressingSequence]) -> float:
        """Calculate heuristic confidence for detected pressing activity.

        Args:
            events: Detected pressure events.
            sequences: Detected pressing sequences.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not events or not sequences:
            return 0.0
        event_count = len(events)
        seq_count = len(sequences)
        density = min(event_count / max(seq_count, 1), 5.0) / 5.0
        avg_speed = float(np.mean([e.closing_speed for e in events]))
        speed_factor = min(avg_speed / 2.0, 1.0)
        success_rate = sum(1 for e in events if e.successful) / event_count
        confidence = (density + speed_factor + success_rate) / 3.0
        return float(np.clip(confidence, 0.0, 1.0))

    def reset(self) -> None:
        """Reset internal detector state."""
        logger.info("PressingDetector reset.")