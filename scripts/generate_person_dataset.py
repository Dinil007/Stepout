"""Generate person classifier dataset from the existing football analytics pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import get_config
from app.dataset.dataset_builder import DatasetBuildConfig, DatasetBuilder
from app.dataset.metadata_writer import MetadataWriter
from app.dataset.preview_generator import PreviewGenerator
from app.detection.detector import YoloDetector
from app.tracking.player_tracker import PlayerTracker
from app.classification.track_validator import TrackValidator


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_embedding(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    try:
        import torch
        from torchvision import transforms
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        model.classifier[0] = torch.nn.Identity()
        model.classifier[3] = torch.nn.Identity()
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            tensor = transform(image_rgb).unsqueeze(0)
            emb = model(tensor).squeeze(0).cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb
    except Exception:
        return None


def validate_tracks(builder: DatasetBuilder, sample_size: int = 5) -> List[Dict]:
    warnings: List[Dict] = []
    track_ids = list(builder.track_records.keys())
    random.seed(42)
    sampled = track_ids if len(track_ids) <= sample_size else random.sample(track_ids, sample_size)

    for tid in sampled:
        recs = builder.track_records[tid]
        if len(recs) < 2:
            continue
        embeddings = []
        for rec in recs:
            path = builder.output_root / rec.image_path
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = _extract_embedding(image_rgb)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) < 2:
            continue
        sims = []
        for i in range(len(embeddings) - 1):
            sims.append(_cosine_similarity(embeddings[i], embeddings[i + 1]))
        avg_sim = float(np.mean(sims)) if sims else 1.0
        min_sim = float(np.min(sims)) if sims else 1.0
        multi_identity = avg_sim < 0.6 or min_sim < 0.3
        warnings.append({
            "track_id": tid,
            "crops": len(recs),
            "embeddings_compared": len(sims),
            "avg_cosine_similarity": round(avg_sim, 4),
            "min_cosine_similarity": round(min_sim, 4),
            "multiple_identities_suspected": bool(multi_identity),
        })

    return warnings


def main() -> None:
    cfg = get_config().raw
    ds_cfg = DatasetBuildConfig(
        enabled=bool(cfg.get("dataset", {}).get("enabled", True)),
        save_every_n_frames=int(cfg.get("dataset", {}).get("save_every_n_frames", 5)),
        crop_size=int(cfg.get("dataset", {}).get("crop_size", 256)),
        padding=bool(cfg.get("dataset", {}).get("padding", True)),
        output_dir=str(cfg.get("dataset", {}).get("output_dir", "datasets/person_classifier")),
        contact_sheet_cols=int(cfg.get("dataset", {}).get("contact_sheet_cols", 5)),
        contact_sheet_rows=int(cfg.get("dataset", {}).get("contact_sheet_rows", 4)),
    )
    if not ds_cfg.enabled:
        print("[DATASET] Dataset generation disabled in config")
        return

    video_path = cfg.get("video", {}).get("input_path", "D:/stepout/videos/raw/match30.mp4")
    max_frames = cfg.get("video", {}).get("max_frames", None)
    fps = float(cfg.get("video", {}).get("fps", 25.0))

    builder = DatasetBuilder(config=ds_cfg)
    builder.setup()
    metadata_writer = MetadataWriter(builder.output_root)
    preview_generator = PreviewGenerator(
        builder.output_root,
        cols=ds_cfg.contact_sheet_cols,
        rows=ds_cfg.contact_sheet_rows,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if max_frames is None else min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    detector = YoloDetector(config=cfg)
    detector.load()
    # Use YOLO native tracking instead of separate tracker
    detector.classes = [0]  # Only detect persons

    start_time = time.perf_counter()
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps if fps > 0 else 0.0
        frame_number = frame_idx + 1
        try:
            # Use YOLO native tracking
            detections = detector.track(frame, persist=True)
        except Exception:
            detections = []
        player_dets = [d for d in detections if getattr(d, "cls_id", None) == 0]

        player_tracks: Dict[int, Dict] = {}
        for trk in player_dets:
            raw_tid = getattr(trk, "track_id", -1)
            tid = int(raw_tid) if raw_tid is not None else -1
            if tid < 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in getattr(trk, "bbox", (0, 0, 0, 0))]
            conf = float(getattr(trk, "conf", getattr(trk, "confidence", 0.0)))
            player_tracks[tid] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            }

        builder.process_frame(frame, frame_number, timestamp, player_tracks)

        print(
            f"[PIPELINE] Processing Frame {frame_number}/{total_frames} | "
            f"Detections: {len(player_dets)} | "
            f"Tracks: {len(player_tracks)}"
        )

    cap.release()
    end_time = time.perf_counter()
    processing_fps = float(total_frames) / (end_time - start_time) if end_time > start_time else 0.0

    summaries = builder.get_summaries()
    metadata_records = [r for recs in builder.track_records.values() for r in recs]
    metadata_writer.write_metadata(metadata_records)
    metadata_writer.write_track_summary(summaries)
    metadata_writer.write_labels([s.track_id for s in summaries])
    stats = builder.get_statistics(processing_fps=processing_fps)
    metadata_writer.write_report(stats, builder)

    for s in summaries:
        preview_generator.generate_for_track(s.track_id, builder.track_records[s.track_id])

    validation_warnings = validate_tracks(builder, sample_size=5)
    debug_path = builder.output_root / "debug_report.txt"
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("Track Debug Report\n")
        f.write("=" * 60 + "\n")
        for s in summaries:
            first_frame = s.first_frame
            last_frame = s.last_frame
            num_crops = s.saved_images
            val = next((w for w in validation_warnings if w["track_id"] == s.track_id), None)
            multi = val["multiple_identities_suspected"] if val else False
            f.write(
                f"Track ID: {s.track_id:04d} | Crops: {num_crops} | FirstFrame: {first_frame} | LastFrame: {last_frame} | "
                f"UniqueTrackerID: {s.track_id} | MultipleIdentitiesSuspected: {multi}\n"
            )
        if not summaries:
            f.write("No tracks generated.\n")

    # ── Track Quality Validation ────────────────────────────────────
    print("\n[VALIDATOR] Running track quality validation...")
    validator = TrackValidator(
        dataset_root=builder.output_root,
        hist_threshold=0.5,
        embedding_threshold=0.4,
        bbox_change_threshold=0.5,
        min_frames=5,
        identity_score_threshold=0.5,
    )
    validation_results = validator.validate_all(builder.track_records, summaries)

    # Write quality report
    quality_report_path = builder.output_root / "track_quality_report.csv"
    validator.write_quality_report(validation_results, quality_report_path)
    print(f"[VALIDATOR] Quality report: {quality_report_path}")

    # Generate contact sheets for rejected tracks
    validator.generate_rejected_contact_sheets(validation_results, builder.track_records)

    # Move rejected tracks out of raw/
    validator.move_rejected_tracks(validation_results)

    accepted = [r for r in validation_results if not r.rejected]
    rejected = [r for r in validation_results if r.rejected]
    print(f"[VALIDATOR] Accepted: {len(accepted)} tracks | Rejected: {len(rejected)} tracks")

    print("Dataset generation complete.")
    print(f"Tracks: {stats['total_tracks']} | Crops: {stats['total_crops']} | Rejected: {stats['rejected_crops']} | FPS: {stats['processing_fps']}")


if __name__ == "__main__":
    main()