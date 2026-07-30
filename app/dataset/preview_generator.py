"""Contact sheet / preview grid generator for manual labeling."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.dataset.dataset_builder import TrackCropRecord


class PreviewGenerator:
    def __init__(self, output_root: Path, cols: int = 5, rows: int = 4) -> None:
        self.output_root = Path(output_root)
        self.cols = cols
        self.rows = rows
        self.max_items = cols * rows

    def generate_for_track(self, track_id: int, records: List[TrackCropRecord]) -> None:
        if not records:
            return
        selected = records[-self.max_items:] if len(records) > self.max_items else records
        thumbnails: List[np.ndarray] = []
        for rec in selected:
            path = self.output_root / rec.image_path
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            side = 256
            if h >= w:
                new_h = side
                new_w = max(1, int(w * side / h))
            else:
                new_w = side
                new_h = max(1, int(h * side / w))
            thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((side, side, 3), dtype=np.uint8)
            y0 = (side - new_h) // 2
            x0 = (side - new_w) // 2
            canvas[y0:y0+new_h, x0:x0+new_w] = thumb
            thumbnails.append(canvas)

        if not thumbnails:
            return

        rows = []
        for r in range(self.rows):
            if r * self.cols >= len(thumbnails):
                break
            row_imgs = thumbnails[r * self.cols:(r + 1) * self.cols]
            while len(row_imgs) < self.cols:
                row_imgs.append(np.zeros((256, 256, 3), dtype=np.uint8))
            rows.append(np.hstack(row_imgs))

        if not rows:
            return
        sheet = np.vstack(rows)
        track_folder = self.output_root / "raw" / f"track_{track_id:04d}"
        track_folder.mkdir(parents=True, exist_ok=True)
        out_path = track_folder / "preview.jpg"
        cv2.imwrite(str(out_path), sheet)