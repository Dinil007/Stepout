"""Crop extraction and quality processing for person classifier dataset."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CropResult:
    success: bool
    crop: Optional[np.ndarray]
    reason: str = ""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    frame: int = 0
    timestamp: float = 0.0
    crop_width: int = 0
    crop_height: int = 0


class CropExtractor:
    def __init__(
        self,
        crop_size: int = 256,
        padding: bool = True,
        min_bbox_size: int = 8,
    ) -> None:
        self.crop_size = crop_size
        self.padding = padding
        self.min_bbox_size = min_bbox_size

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float = 0.0, frame_number: int = 0, timestamp: float = 0.0) -> CropResult:
        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(w_img - 1, x2))
        y2 = int(min(h_img - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return CropResult(success=False, crop=None, reason="invalid_bbox", bbox=(x1, y1, x2, y2), confidence=confidence, frame=frame_number, timestamp=timestamp)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < self.min_bbox_size or crop_h < self.min_bbox_size:
            return CropResult(success=False, crop=None, reason="bbox_too_small", bbox=(x1, y1, x2, y2), confidence=confidence, frame=frame_number, timestamp=timestamp)

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return CropResult(success=False, crop=None, reason="empty_crop", bbox=(x1, y1, x2, y2), confidence=confidence, frame=frame_number, timestamp=timestamp)

        longest = max(crop_h, crop_w)
        target = self.crop_size
        if longest == 0:
            return CropResult(success=False, crop=None, reason="zero_size", bbox=(x1, y1, x2, y2), confidence=confidence, frame=frame_number, timestamp=timestamp)

        scale = target / float(longest)
        new_w = int(round(crop_w * scale))
        new_h = int(round(crop_h * scale))
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.padding:
            pad_top = (target - new_h) // 2
            pad_bottom = target - new_h - pad_top
            pad_left = (target - new_w) // 2
            pad_right = target - new_w - pad_left
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            out = padded
            final_w, final_h = target, target
        else:
            out = resized
            final_w, final_h = new_w, new_h

        return CropResult(success=True, crop=out, bbox=(x1, y1, x2, y2), confidence=confidence, frame=frame_number, timestamp=timestamp, crop_width=final_w, crop_height=final_h)