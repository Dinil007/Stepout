"""Debug visualization helpers for detection, tracking, and team stages."""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.detection.detection_types import Detection


class DebugVisualizer:
    """Render inspection overlays and contact sheets."""

    TEAM_COLORS = {
        "Team A": (40, 60, 255),
        "Team B": (255, 70, 30),
        "Unknown": (180, 180, 180),
    }

    def __init__(self, pitch_roi: np.ndarray) -> None:
        self.pitch_roi = pitch_roi
        self.contact_samples: List[np.ndarray] = []
        self.max_contact_samples = 50

    def draw_detections(
        self,
        frame: np.ndarray,
        accepted: List[Detection],
        rejected: List[Detection],
        balls: List[Detection],
        title: str = "",
    ) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [self.pitch_roi], True, (255, 255, 0), 2)
        for d in rejected:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = d.reject_reason or "REJECT"
            cv2.putText(out, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for d in accepted:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.circle(out, d.foot, 3, (0, 255, 255), -1)
            cv2.putText(out, f"P {d.conf:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        for d in balls:
            cv2.circle(out, d.center, max(5, int(math.sqrt(max(d.area, 1)) / 2) + 4), (0, 255, 255), 2)
            cv2.putText(out, f"BALL {d.conf:.2f}", (d.center[0] + 8, d.center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if title:
            cv2.putText(out, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return out

    def draw_tracking(
        self,
        frame: np.ndarray,
        players: List[Detection],
        ball_center: Optional[Tuple[float, float]] = None,
        ball_detected: bool = False,
        ball_history: Optional[List[Tuple[float, float]]] = None,
        new_track_ids: Optional[set] = None,
    ) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [self.pitch_roi], True, (255, 255, 0), 2)
        new_track_ids = new_track_ids or set()
        for d in players:
            x1, y1, x2, y2 = d.bbox
            color = (0, 0, 255) if d.track_id in new_track_ids else (255, 140, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"ID:{d.track_id}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if ball_center is not None and ball_history:
            for i in range(1, len(ball_history)):
                p1 = (int(ball_history[i - 1][0]), int(ball_history[i - 1][1]))
                p2 = (int(ball_history[i][0]), int(ball_history[i][1]))
                cv2.line(out, p1, p2, (0, 220, 255), 1)
            bc = (int(ball_center[0]), int(ball_center[1]))
            b_color = (0, 255, 255) if ball_detected else (0, 165, 255)
            cv2.circle(out, bc, 9, b_color, 2)
            state = "detected" if ball_detected else "predicted"
            cv2.putText(out, f"Ball [{state}]", (bc[0] - 30, max(15, bc[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 2)
        return out

    def draw_teams(
        self,
        frame: np.ndarray,
        players: List[Detection],
        team_labels: Dict[int, Tuple[str, float]],
    ) -> np.ndarray:
        out = frame.copy()
        for d in players:
            label, conf = team_labels.get(d.track_id, ("Unknown", 0.0))
            color = self.TEAM_COLORS.get(label, (180, 180, 180))
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{label} {conf:.2f} ID{d.track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        return out

    def maybe_add_contact_sample(
        self,
        frame: np.ndarray,
        det_view: np.ndarray,
        trk_view: np.ndarray,
        team_view: np.ndarray,
        frame_index: int,
        sample_stride: int,
    ) -> None:
        if len(self.contact_samples) >= self.max_contact_samples:
            return
        if frame_index != 1 and frame_index % sample_stride != 0:
            return
        h, w = frame.shape[:2]
        scale = 0.35
        sh, sw = int(h * scale), int(w * scale)
        panels = [
            cv2.resize(frame, (sw, sh)),
            cv2.resize(det_view, (sw, sh)),
            cv2.resize(trk_view, (sw, sh)),
            cv2.resize(team_view, (sw, sh)),
        ]
        labels = ["Original", "Detection", "Tracking", "Team"]
        for panel, label in zip(panels, labels):
            cv2.rectangle(panel, (0, 0), (sw, 22), (0, 0, 0), -1)
            cv2.putText(panel, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        row = np.hstack(panels)
        cv2.putText(row, f"Frame {frame_index}", (6, sh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        self.contact_samples.append(row)

    def save_contact_sheet(self, path) -> None:
        if not self.contact_samples:
            return
        cols = 2
        rows = []
        for i in range(0, len(self.contact_samples), cols):
            chunk = self.contact_samples[i : i + cols]
            if len(chunk) == 1:
                chunk.append(np.zeros_like(chunk[0]))
            rows.append(np.hstack(chunk))
        canvas = np.vstack(rows)
        cv2.imwrite(str(path), canvas)

    def render_preprocessing_comparison(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        det_view: np.ndarray,
        steps: List[str],
        path,
    ) -> None:
        h, w = original.shape[:2]
        panels = [
            cv2.resize(original, (w // 2, h // 2)),
            cv2.resize(preprocessed, (w // 2, h // 2)),
            cv2.resize(det_view, (w // 2, h // 2)),
        ]
        labels = ["Original", "Preprocessed: " + ",".join(steps or ["none"]), "Detection Result"]
        for panel, label in zip(panels, labels):
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(panel, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imwrite(str(path), np.vstack(panels))
