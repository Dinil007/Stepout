"""
Comprehensive Team Classification Validation and Debugging

Generates detailed reports explaining every classification decision.
"""

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ultralytics import YOLO

from app.core.config import get_config
from app.detection.detection_filter import parse_yolo_results, inside_pitch, split_dets
from app.team_classification.jersey_classifier import JerseyClassifier
from app.team_classification.team_metrics import TeamMetricsCollector
from app.utils.roi_loader import load_pitch_roi_as_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUTPUT_DIR = ROOT / "outputs" / "team_validation"
CROPS_DIR = OUTPUT_DIR / "crops"
MODEL_WEIGHTS = ROOT / "yolov8x.pt"
TRACKER_CONFIG = ROOT / "app" / "tracking" / "bytetrack_custom.yaml"
PITCH_ROI, _ = load_pitch_roi_as_numpy(ROOT, verbose=True)


class TeamValidationAnalyzer:
    """Comprehensive team classification validator."""

    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.classifier = JerseyClassifier()
        self.metrics_writer = TeamMetricsCollector(OUTPUT_DIR / "classification.csv")
        
        self.frame_quality: Dict[int, Dict] = {}
        self.team_history: Dict[int, List[Tuple[int, str, float]]] = defaultdict(list)
        self.switch_events: List[Dict] = []
        self.suspicious_cases: List[Dict] = []
        self.feature_vectors: List[np.ndarray] = []
        self.feature_labels: List[int] = []

    def analyze(self, max_frames: int = 0) -> Dict:
        logger.info("Starting team classification validation...")
        
        cap = cv2.VideoCapture(str(INPUT_VIDEO))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {INPUT_VIDEO}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames > 0:
            total = min(total, max_frames)
        
        device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        model = YOLO(str(MODEL_WEIGHTS))
        model.to(device)
        try:
            model.fuse()
        except Exception:
            pass
        model.model.half()
        
        frame_rows = []
        frame_no = 0
        
        with __import__("torch").inference_mode():
            while frame_no < total:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1
                
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    classes=[0],
                    conf=0.25,
                    iou=0.55,
                    verbose=False,
                    device=device,
                )
                
                players, _, _ = split_dets(parse_results(results))
                
                quality = self._assess_frame_quality(frame)
                self.frame_quality[frame_no] = quality
                
                tracked = []
                for d in players:
                    if d.track_id >= 0:
                        tracked.append(d)
                        label, conf = self.classifier.classify(d.track_id, frame, d.bbox)
                        self._record_classification(frame_no, d.track_id, d, label, conf, frame)
                        self.metrics_writer.record(frame_no, d.track_id, label, conf, d.bbox[3] - d.bbox[1], conf)
                
                self._detect_team_switches(frame_no)
                
                frame_rows.append({
                    "frame": frame_no,
                    "players": len(tracked),
                    "team_a": sum(1 for d in tracked if self.classifier.track_team.get(d.track_id) == 0),
                    "team_b": sum(1 for d in tracked if self.classifier.track_team.get(d.track_id) == 1),
                    "unknown": sum(1 for d in tracked if self.classifier.track_team.get(d.track_id) is None),
                })
                
                if frame_no % 100 == 0:
                    logger.info(f"Processed {frame_no}/{total} frames")
        
        cap.release()
        
        self.metrics_writer.flush()
        
        report = self._generate_report(frame_rows, total)
        self._write_outputs(report)
        
        logger.info("Team classification validation complete.")
        return report

    def _assess_frame_quality(self, frame: np.ndarray) -> Dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        issues = []
        if lap < 100:
            issues.append("blur")
        if mean < 80:
            issues.append("underexposure")
        if mean > 200:
            issues.append("overexposure")
        
        return {
            "brightness": mean,
            "contrast": std,
            "blur": lap,
            "issues": issues,
        }

    def _record_classification(self, frame: int, track_id: int, det, label: str, conf: float, frame_img: np.ndarray) -> None:
        self.team_history[track_id].append((frame, label, conf))
        
        x1, y1, x2, y2 = det.bbox
        crop = frame_img[y1:y2, x1:x2]
        if crop.size > 0:
            crop_path = CROPS_DIR / f"frame_{frame:06d}_player_{track_id:03d}.png"
            cv2.imwrite(str(crop_path), crop)
        
        if self.classifier.centers is not None:
            color = self.classifier.extract(frame_img, det.bbox)
            if color is not None:
                dists = np.linalg.norm(self.classifier.centers - color, axis=1)
                self.feature_vectors.append(color)
                self.feature_labels.append(0 if label == "Team A" else 1 if label == "Team B" else 2)

    def _detect_team_switches(self, frame: int) -> None:
        for track_id, history in self.team_history.items():
            if len(history) < 2:
                continue
            current = history[-1]
            previous = history[-2]
            if current[1] != previous[1] and current[1] != "Unknown" and previous[1] != "Unknown":
                self.switch_events.append({
                    "track_id": track_id,
                    "frame": frame,
                    "previous_team": previous[1],
                    "new_team": current[1],
                    "confidence": current[2],
                    "reason": "Confidence drop or rapid reclassification",
                })

    def _detect_suspicious_cases(self) -> None:
        for track_id, history in self.team_history.items():
            if len(history) < 3:
                continue
            teams = [h[1] for h in history]
            confs = [h[2] for h in history]
            
            if teams.count("Unknown") > len(teams) * 0.5:
                self.suspicious_cases.append({
                    "track_id": track_id,
                    "reason": "Majority Unknown classifications",
                    "severity": "high",
                })
            
            if max(confs) - min(confs) > 0.4:
                self.suspicious_cases.append({
                    "track_id": track_id,
                    "reason": "High confidence variance",
                    "severity": "medium",
                })
            
            team_changes = sum(1 for i in range(1, len(teams)) if teams[i] != teams[i-1] and teams[i] != "Unknown" and teams[i-1] != "Unknown")
            if team_changes > 2:
                self.suspicious_cases.append({
                    "track_id": track_id,
                    "reason": f"Multiple team switches ({team_changes})",
                    "severity": "high",
                })

    def _compute_distribution(self, frame_rows: List[Dict]) -> List[Dict]:
        distribution = []
        for row in frame_rows:
            total = row["players"]
            distribution.append({
                "frame": row["frame"],
                "team_a_pct": row["team_a"] / max(total, 1) * 100,
                "team_b_pct": row["team_b"] / max(total, 1) * 100,
                "unknown_pct": row["unknown"] / max(total, 1) * 100,
            })
        return distribution

    def _generate_confidence_analysis(self) -> Dict:
        all_confs = []
        low_conf_count = 0
        for history in self.team_history.values():
            for _, _, conf in history:
                all_confs.append(conf)
                if conf < self.classifier.sticky_threshold and conf > 0:
                    low_conf_count += 1
        
        if not all_confs:
            return {}
        
        all_confs.sort()
        n = len(all_confs)
        median = all_confs[n // 2] if n % 2 == 1 else (all_confs[n // 2 - 1] + all_confs[n // 2]) / 2
        
        return {
            "min": min(all_confs),
            "max": max(all_confs),
            "mean": sum(all_confs) / n,
            "median": median,
            "low_confidence_count": low_conf_count,
        }

    def _generate_report(self, frame_rows: List[Dict], total_frames: int) -> Dict:
        self._detect_suspicious_cases()
        
        total_players = sum(r["players"] for r in frame_rows)
        classified = sum(r["team_a"] + r["team_b"] for r in frame_rows)
        unknown = sum(r["unknown"] for r in frame_rows)
        confidence = self._generate_confidence_analysis()
        
        return {
            "total_frames": total_frames,
            "total_players_detected": total_players,
            "total_classified": classified,
            "total_unknown": unknown,
            "classification_rate": classified / max(total_players, 1) * 100,
            "confidence_analysis": confidence,
            "team_switches": len(self.switch_events),
            "suspicious_cases": len(self.suspicious_cases),
            "frame_distribution": self._compute_distribution(frame_rows),
        }

    def _write_outputs(self, report: Dict) -> None:
        (OUTPUT_DIR / "team_validation_report.md").write_text(self._generate_markdown(report), encoding="utf-8")
        (OUTPUT_DIR / "team_distribution.csv").write_text(
            "frame,team_a_pct,team_b_pct,unknown_pct\n" + "\n".join(
                f"{d['frame']},{d['team_a_pct']:.1f},{d['team_b_pct']:.1f},{d['unknown_pct']:.1f}"
                for d in report["frame_distribution"]
            ),
            encoding="utf-8",
        )
        
        with (OUTPUT_DIR / "team_switches.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "previous_team", "new_team", "frame", "confidence", "reason"])
            writer.writeheader()
            writer.writerows(self.switch_events)
        
        with (OUTPUT_DIR / "unknown_players.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "frame", "confidence", "reason"])
            writer.writeheader()
            for track_id, history in self.team_history.items():
                for frame, label, conf in history:
                    if label == "Unknown":
                        writer.writerow({"track_id": track_id, "frame": frame, "confidence": conf, "reason": "Low confidence or insufficient data"})
        
        with (OUTPUT_DIR / "suspicious_cases.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "reason", "severity"])
            writer.writeheader()
            writer.writerows(self.suspicious_cases)
        
        if len(self.feature_vectors) > 2:
            self._generate_cluster_visualization()

    def _generate_cluster_visualization(self) -> None:
        X = np.array(self.feature_vectors)
        y = np.array(self.feature_labels)
        
        try:
            X_pca = PCA(n_components=2).fit_transform(X)
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ["blue" if l == 0 else "red" if l == 1 else "gray" for l in y]
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=20)
            ax.set_title("Team Classification Feature Clusters")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            fig.savefig(OUTPUT_DIR / "feature_clusters.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Cluster visualization failed: {e}")

    def _generate_markdown(self, report: Dict) -> str:
        lines = []
        lines.append("# Team Classification Validation Report\n")
        lines.append(f"- Total frames processed: {report['total_frames']}")
        lines.append(f"- Total players detected: {report['total_players_detected']}")
        lines.append(f"- Total classified: {report['total_classified']}")
        lines.append(f"- Total unknown: {report['total_unknown']}")
        lines.append(f"- Classification rate: {report['classification_rate']:.1f}%")
        
        ca = report.get("confidence_analysis", {})
        if ca:
            lines.append(f"\n## Confidence Analysis\n")
            lines.append(f"- Min: {ca.get('min', 0):.3f}")
            lines.append(f"- Max: {ca.get('max', 0):.3f}")
            lines.append(f"- Mean: {ca.get('mean', 0):.3f}")
            lines.append(f"- Median: {ca.get('median', 0):.3f}")
            lines.append(f"- Low confidence count: {ca.get('low_confidence_count', 0)}")
        
        lines.append(f"\n## Team Analysis\n")
        lines.append(f"- Team switches detected: {report['team_switches']}")
        lines.append(f"- Suspicious cases: {report['suspicious_cases']}")
        
        lines.append(f"\n## Recommendations\n")
        if report["classification_rate"] < 90:
            lines.append("- Classification rate below 90%. Review jersey crop extraction.")
        if report.get("suspicious_cases", 0) > 10:
            lines.append("- High number of suspicious cases. Review confidence thresholds.")
        
        return "\n".join(lines)


def main() -> int:
    analyzer = TeamValidationAnalyzer()
    report = analyzer.analyze()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())