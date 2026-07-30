"""Track Quality Validator.
Validates each track for identity consistency before training.
Rejects corrupted tracks (identity switches, poor quality, inconsistent bbox).
"""
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.dataset.dataset_builder import TrackCropRecord, TrackSummary


@dataclass
class TrackQualityResult:
    track_id: int
    frames: int
    avg_jersey_hist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    avg_histogram_distance: float = 0.0
    max_histogram_distance: float = 0.0
    avg_embedding_similarity: float = 0.0
    min_embedding_similarity: float = 0.0
    bbox_width_std: float = 0.0
    bbox_height_std: float = 0.0
    bbox_width_mean: float = 0.0
    bbox_height_mean: float = 0.0
    aspect_ratio_std: float = 0.0
    aspect_ratio_mean: float = 0.0
    motion_smoothness: float = 0.0
    identity_score: float = 1.0
    rejected: bool = False
    reason: str = ""
    # Identity switch detection flags
    histogram_jump_detected: bool = False
    embedding_jump_detected: bool = False
    bbox_change_detected: bool = False
    appearance_change_detected: bool = False


class TrackValidator:
    """Validates track quality using multiple metrics."""

    def __init__(
        self,
        dataset_root: Path,
        hist_threshold: float = 0.5,
        embedding_threshold: float = 0.4,
        bbox_change_threshold: float = 0.5,
        appearance_threshold: float = 0.3,
        min_frames: int = 5,
        identity_score_threshold: float = 0.5,
    ):
        self.dataset_root = Path(dataset_root)
        self.raw_dir = self.dataset_root / "raw"
        self.rejected_dir = self.dataset_root / "rejected_tracks"
        self.hist_threshold = hist_threshold
        self.embedding_threshold = embedding_threshold
        self.bbox_change_threshold = bbox_change_threshold
        self.appearance_threshold = appearance_threshold
        self.min_frames = min_frames
        self.identity_score_threshold = identity_score_threshold

    # ── Feature extraction ──────────────────────────────────────────

    def _extract_jersey_histogram(self, img: np.ndarray) -> np.ndarray:
        """Extract HSV histogram focusing on the lower half (jersey area)."""
        h, w = img.shape[:2]
        # Focus on lower 60% of the crop (jersey/body area)
        lower_half = img[int(h * 0.3):, :]
        hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _histogram_distance(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Correlation-based distance between two histograms.
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        # Correlation is [-1, 1]. Convert to distance [0, 1].
        return max(0.0, 1.0 - corr) / 2.0

    def _extract_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract a simple appearance embedding using color + edge features."""
        try:
            # Color moments
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean, std = cv2.meanStdDev(hsv)
            color_feat = np.concatenate([mean.flatten(), std.flatten()])

            # Edge histogram
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            cv2.normalize(edge_hist, edge_hist)
            edge_feat = edge_hist.flatten()

            # Texture (LBP-like: local std)
            local_std = cv2.blur(gray, (5, 5))
            texture_hist = cv2.calcHist([local_std.astype(np.uint8)], [0], None, [16], [0, 256])
            cv2.normalize(texture_hist, texture_hist)
            texture_feat = texture_hist.flatten()

            emb = np.concatenate([color_feat, edge_feat, texture_feat])
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            return emb
        except Exception:
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Per-track validation ────────────────────────────────────────

    def validate_track(
        self,
        track_id: int,
        records: List[TrackCropRecord],
    ) -> TrackQualityResult:
        """Run all quality checks on a single track."""
        result = TrackQualityResult(track_id=track_id, frames=len(records))

        if len(records) < self.min_frames:
            result.rejected = True
            result.reason = f"Too few frames ({len(records)} < {self.min_frames})"
            result.identity_score = 0.0
            return result

        # Load all images
        images: List[Tuple[str, np.ndarray]] = []
        for rec in records:
            path = self.dataset_root / rec.image_path
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                images.append((rec.image_path, img))

        if len(images) < 2:
            result.rejected = True
            result.reason = "Could not load enough images"
            result.identity_score = 0.0
            return result

        # ── 1. Jersey color histogram analysis ──────────────────────
        histograms = []
        for _, img in images:
            h = self._extract_jersey_histogram(img)
            histograms.append(h)

        # Average jersey histogram
        result.avg_jersey_hist = np.mean(histograms, axis=0)

        # Frame-to-frame histogram distance
        hist_dists = []
        for i in range(1, len(histograms)):
            d = self._histogram_distance(histograms[i - 1], histograms[i])
            hist_dists.append(d)

        result.avg_histogram_distance = float(np.mean(hist_dists)) if hist_dists else 0.0
        result.max_histogram_distance = float(np.max(hist_dists)) if hist_dists else 0.0

        # Detect histogram jump (identity switch via color change)
        result.histogram_jump_detected = result.max_histogram_distance > self.hist_threshold

        # ── 2. Embedding similarity ─────────────────────────────────
        embeddings = []
        for _, img in images:
            emb = self._extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) >= 2:
            sims = []
            for i in range(1, len(embeddings)):
                s = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                sims.append(s)
            result.avg_embedding_similarity = float(np.mean(sims)) if sims else 1.0
            result.min_embedding_similarity = float(np.min(sims)) if sims else 1.0

            # Detect embedding jump
            result.embedding_jump_detected = result.min_embedding_similarity < self.embedding_threshold
        else:
            result.avg_embedding_similarity = 1.0
            result.min_embedding_similarity = 1.0

        # ── 3. Bounding box size consistency ────────────────────────
        widths = []
        heights = []
        aspect_ratios = []
        for rec in records:
            x1, y1, x2, y2 = rec.bbox
            w = x2 - x1
            h = y2 - y1
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / max(h, 1))

        result.bbox_width_mean = float(np.mean(widths)) if widths else 0.0
        result.bbox_height_mean = float(np.mean(heights)) if heights else 0.0
        result.bbox_width_std = float(np.std(widths)) if widths else 0.0
        result.bbox_height_std = float(np.std(heights)) if heights else 0.0
        result.aspect_ratio_mean = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
        result.aspect_ratio_std = float(np.std(aspect_ratios)) if aspect_ratios else 0.0

        # Detect large bbox change (normalized by mean)
        bbox_changes = []
        for i in range(1, len(records)):
            w_prev = widths[i - 1]
            h_prev = heights[i - 1]
            w_cur = widths[i]
            h_cur = heights[i]
            # Relative change
            w_change = abs(w_cur - w_prev) / max(w_prev, 1)
            h_change = abs(h_cur - h_prev) / max(h_prev, 1)
            bbox_changes.append(max(w_change, h_change))

        max_bbox_change = float(np.max(bbox_changes)) if bbox_changes else 0.0
        result.bbox_change_detected = max_bbox_change > self.bbox_change_threshold

        # ── 4. Motion smoothness ────────────────────────────────────
        # Compute center-of-bbox displacement
        centers = []
        for rec in records:
            x1, y1, x2, y2 = rec.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))

        displacements = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            displacements.append(np.sqrt(dx**2 + dy**2))

        if displacements:
            # Motion smoothness = 1 / (1 + std_of_displacements)
            # High std = jerky motion (potential ID switch)
            disp_std = float(np.std(displacements))
            disp_mean = float(np.mean(displacements))
            result.motion_smoothness = 1.0 / (1.0 + disp_std / max(disp_mean, 1e-6))
        else:
            result.motion_smoothness = 1.0

        # ── 5. Sudden appearance change ─────────────────────────────
        # Compare first vs last frame directly
        if len(images) >= 2:
            first_img = images[0][1]
            last_img = images[-1][1]
            h_first = self._extract_jersey_histogram(first_img)
            h_last = self._extract_jersey_histogram(last_img)
            first_last_hist_dist = self._histogram_distance(h_first, h_last)

            emb_first = self._extract_embedding(first_img)
            emb_last = self._extract_embedding(last_img)
            if emb_first is not None and emb_last is not None:
                first_last_emb_sim = self._cosine_similarity(emb_first, emb_last)
            else:
                first_last_emb_sim = 1.0

            result.appearance_change_detected = (
                first_last_hist_dist > self.hist_threshold * 1.5
                or first_last_emb_sim < self.embedding_threshold * 0.7
            )

        # ── 6. Composite identity score ─────────────────────────────
        # Weighted combination of all metrics
        score = 1.0
        penalties = []

        # Histogram consistency (lower distance = better)
        if hist_dists:
            hist_score = 1.0 / (1.0 + result.avg_histogram_distance)
            penalties.append(("histogram", 1.0 - hist_score))

        # Embedding consistency (higher similarity = better)
        if result.min_embedding_similarity < 1.0:
            emb_score = result.avg_embedding_similarity
            penalties.append(("embedding", 1.0 - emb_score))

        # Bbox consistency (lower std/mean = better)
        if result.bbox_width_mean > 0:
            w_cv = result.bbox_width_std / result.bbox_width_mean  # coefficient of variation
            h_cv = result.bbox_height_std / result.bbox_height_mean
            bbox_score = 1.0 / (1.0 + (w_cv + h_cv) / 2.0)
            penalties.append(("bbox", 1.0 - bbox_score))

        # Aspect ratio consistency
        if result.aspect_ratio_mean > 0:
            ar_cv = result.aspect_ratio_std / result.aspect_ratio_mean
            ar_score = 1.0 / (1.0 + ar_cv)
            penalties.append(("aspect_ratio", 1.0 - ar_score))

        # Motion smoothness
        motion_penalty = 1.0 - result.motion_smoothness
        penalties.append(("motion", motion_penalty))

        # Apply penalties (weighted)
        if penalties:
            total_penalty = sum(p[1] for p in penalties) / len(penalties)
            score = max(0.0, 1.0 - total_penalty * 2.0)  # Scale penalty

        result.identity_score = round(score, 4)

        # ── 7. Decision ─────────────────────────────────────────────
        reasons = []
        if result.histogram_jump_detected:
            reasons.append(f"Histogram jump ({result.max_histogram_distance:.3f})")
        if result.embedding_jump_detected:
            reasons.append(f"Embedding drop ({result.min_embedding_similarity:.3f})")
        if result.bbox_change_detected:
            reasons.append(f"Bbox change ({max_bbox_change:.3f})")
        if result.appearance_change_detected:
            reasons.append("Appearance change (first vs last)")
        if result.identity_score < self.identity_score_threshold:
            reasons.append(f"Low identity score ({result.identity_score:.3f})")

        if reasons:
            result.rejected = True
            result.reason = "; ".join(reasons)

        return result

    # ── Batch validation ────────────────────────────────────────────

    def validate_all(
        self,
        track_records: Dict[int, List[TrackCropRecord]],
        summaries: List[TrackSummary],
    ) -> List[TrackQualityResult]:
        """Validate all tracks and return results."""
        results: List[TrackQualityResult] = []
        for s in summaries:
            tid = s.track_id
            records = track_records.get(tid, [])
            result = self.validate_track(tid, records)
            results.append(result)
        return results

    # ── Rejection handling ──────────────────────────────────────────

    def move_rejected_tracks(
        self,
        results: List[TrackQualityResult],
    ) -> None:
        """Move rejected track folders to rejected_tracks/."""
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            if not r.rejected:
                continue
            src = self.raw_dir / f"track_{r.track_id:04d}"
            dst = self.rejected_dir / f"track_{r.track_id:04d}"
            if src.exists():
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.move(str(src), str(dst))
                print(f"[VALIDATOR] Rejected track_{r.track_id:04d}: {r.reason}")

    # ── Report generation ───────────────────────────────────────────

    def write_quality_report(
        self,
        results: List[TrackQualityResult],
        output_path: Path,
    ) -> None:
        """Write track_quality_report.csv."""
        with open(str(output_path), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Track ID",
                "Frames",
                "Identity Score",
                "Rejected",
                "Reason",
                "Avg Hist Dist",
                "Max Hist Dist",
                "Avg Emb Sim",
                "Min Emb Sim",
                "BBox W Std",
                "BBox H Std",
                "Aspect Ratio Std",
                "Motion Smoothness",
                "Hist Jump",
                "Emb Jump",
                "BBox Change",
                "Appearance Change",
            ])
            for r in sorted(results, key=lambda x: x.track_id):
                writer.writerow([
                    r.track_id,
                    r.frames,
                    f"{r.identity_score:.4f}",
                    "YES" if r.rejected else "NO",
                    r.reason,
                    f"{r.avg_histogram_distance:.4f}",
                    f"{r.max_histogram_distance:.4f}",
                    f"{r.avg_embedding_similarity:.4f}",
                    f"{r.min_embedding_similarity:.4f}",
                    f"{r.bbox_width_std:.2f}",
                    f"{r.bbox_height_std:.2f}",
                    f"{r.aspect_ratio_std:.4f}",
                    f"{r.motion_smoothness:.4f}",
                    "YES" if r.histogram_jump_detected else "NO",
                    "YES" if r.embedding_jump_detected else "NO",
                    "YES" if r.bbox_change_detected else "NO",
                    "YES" if r.appearance_change_detected else "NO",
                ])

    # ── Visual contact sheet for rejected tracks ────────────────────

    def generate_rejected_contact_sheets(
        self,
        results: List[TrackQualityResult],
        track_records: Dict[int, List[TrackCropRecord]],
    ) -> None:
        """Create contact_sheet.jpg for each rejected track showing key frames."""
        contact_dir = self.rejected_dir
        contact_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            if not r.rejected:
                continue
            records = track_records.get(r.track_id, [])
            if not records:
                continue

            # Load all images
            images: List[np.ndarray] = []
            for rec in records:
                path = self.dataset_root / rec.image_path
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)

            if len(images) < 2:
                continue

            # Select key frames
            first = images[0]
            middle = images[len(images) // 2]
            last = images[-1]

            # Find frame with largest appearance change (max histogram distance from first)
            h_first = self._extract_jersey_histogram(first)
            max_change_idx = 0
            max_change_dist = 0.0
            for i, img in enumerate(images):
                h = self._extract_jersey_histogram(img)
                d = self._histogram_distance(h_first, h)
                if d > max_change_dist:
                    max_change_dist = d
                    max_change_idx = i
            largest_change = images[max_change_idx]

            # Create contact sheet (2x2 grid)
            side = 256
            key_frames = [
                ("First", first),
                ("Middle", middle),
                ("Last", last),
                (f"MaxChange(f{max_change_idx})", largest_change),
            ]

            rows = []
            row_imgs = []
            for label, img in key_frames:
                h, w = img.shape[:2]
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
                canvas[y0:y0 + new_h, x0:x0 + new_w] = thumb

                # Add label
                cv2.putText(
                    canvas, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )
                row_imgs.append(canvas)

                if len(row_imgs) == 2:
                    rows.append(np.hstack(row_imgs))
                    row_imgs = []

            if row_imgs:
                while len(row_imgs) < 2:
                    row_imgs.append(np.zeros((side, side, 3), dtype=np.uint8))
                rows.append(np.hstack(row_imgs))

            if not rows:
                continue

            sheet = np.vstack(rows)

            # Add info header
            info_bar = np.zeros((40, sheet.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                info_bar,
                f"Track {r.track_id:04d} | Score: {r.identity_score:.3f} | {r.reason}",
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
            sheet = np.vstack([info_bar, sheet])

            out_path = contact_dir / f"track_{r.track_id:04d}_contact.jpg"
            cv2.imwrite(str(out_path), sheet)
            print(f"[VALIDATOR] Contact sheet saved: {out_path}")