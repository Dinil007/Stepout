"""
Football Analytics Evaluation Framework

Measures accuracy of every analytics module:
- Tracking (MOTA, MOTP, IDF1, ID Switches, etc.)
- Event Detection (passes, shots, goals, possession changes)
- Formation Detection (accuracy, stability, confidence)
- Player Metrics (speed, distance, heatmap consistency)

Generates:
- evaluation_report.json
- evaluation_dashboard.json
- module_scores.json
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationThresholds:
    """PASS/FAIL thresholds for each metric."""
    # Tracking thresholds
    mota_min: float = 0.6
    motp_min: float = 0.5
    idf1_min: float = 0.5
    max_id_switches: int = 20
    max_fragmentations: int = 15
    track_recall_min: float = 0.7
    track_precision_min: float = 0.7

    # Event detection thresholds
    pass_precision_min: float = 0.7
    pass_recall_min: float = 0.7
    shot_precision_min: float = 0.6
    shot_recall_min: float = 0.6
    goal_precision_min: float = 0.8
    goal_recall_min: float = 0.8
    possession_precision_min: float = 0.8
    possession_recall_min: float = 0.8

    # Formation detection thresholds
    formation_accuracy_min: float = 0.6
    formation_stability_min: float = 0.6
    confidence_mean_min: float = 0.6
    change_detection_precision_min: float = 0.5
    change_detection_recall_min: float = 0.5

    # Player metrics thresholds
    speed_error_max: float = 5.0  # km/h
    distance_error_max: float = 500.0  # meters
    heatmap_iou_min: float = 0.5


class EvaluationFramework:
    """Evaluate accuracy of all analytics modules."""

    def __init__(self, output_dir: Path, thresholds: Optional[EvaluationThresholds] = None):
        self.output_dir = output_dir
        self.thresholds = thresholds or EvaluationThresholds()
        self.results: Dict[str, Any] = {}

    def _safe_div(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value."""
        return numerator / denominator if denominator != 0 else default

    # ============================================================
    # TRACKING EVALUATION
    # ============================================================
    def evaluate_tracking(
        self,
        gt_tracks: List[Dict],
        pred_tracks: List[Dict],
        total_gt_objects: int,
    ) -> Dict[str, Any]:
        """Evaluate tracking performance.

        Args:
            gt_tracks: Ground truth tracks [{frame, track_id, bbox}]
            pred_tracks: Predicted tracks [{frame, track_id, bbox}]
            total_gt_objects: Total ground truth objects in video

        Returns:
            Dictionary with tracking metrics.
        """
        # Build frame-by-frame associations
        gt_by_frame: Dict[int, List[Dict]] = {}
        pred_by_frame: Dict[int, List[Dict]] = {}

        for t in gt_tracks:
            gt_by_frame.setdefault(t["frame"], []).append(t)
        for t in pred_tracks:
            pred_by_frame.setdefault(t["frame"], []).append(t)

        all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

        total_matches = 0
        total_gt = 0
        total_pred = 0
        iou_sum = 0.0
        id_switches = 0
        fragmentations = 0
        track_assignments: Dict[int, int] = {}  # gt_id -> pred_id

        for frame in all_frames:
            gt_boxes = gt_by_frame.get(frame, [])
            pred_boxes = pred_by_frame.get(frame, [])
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            # Greedy matching by IoU
            matches = self._match_boxes(gt_boxes, pred_boxes)
            total_matches += len(matches)

            for gt_id, pred_id, iou in matches:
                iou_sum += iou
                # Check for ID switch
                if gt_id in track_assignments:
                    if track_assignments[gt_id] != pred_id:
                        id_switches += 1
                track_assignments[gt_id] = pred_id

        # Calculate metrics
        mota = 1.0 - (id_switches + fragmentations + abs(total_pred - total_matches)) / max(total_gt_objects, 1)
        motp = iou_sum / max(total_matches, 1)
        idf1 = self._safe_div(2 * total_matches, total_gt + total_pred)

        # Track recall and precision
        track_recall = self._safe_div(total_matches, total_gt_objects)
        track_precision = self._safe_div(total_matches, max(total_pred, 1))

        # Pass/Fail assessment
        passed = (
            mota >= self.thresholds.mota_min and
            motp >= self.thresholds.motp_min and
            idf1 >= self.thresholds.idf1_min and
            id_switches <= self.thresholds.max_id_switches and
            fragmentations <= self.thresholds.max_fragmentations and
            track_recall >= self.thresholds.track_recall_min and
            track_precision >= self.thresholds.track_precision_min
        )

        return {
            "mota": round(mota, 4),
            "motp": round(motp, 4),
            "idf1": round(idf1, 4),
            "id_switches": id_switches,
            "fragmentations": fragmentations,
            "track_recall": round(track_recall, 4),
            "track_precision": round(track_precision, 4),
            "total_gt": total_gt,
            "total_pred": total_pred,
            "matches": total_matches,
            "passed": passed,
        }

    def _match_boxes(self, gt_boxes: List[Dict], pred_boxes: List[Dict], iou_threshold: float = 0.5) -> List[Tuple]:
        """Match GT and predicted boxes by IoU."""
        matches = []
        if not gt_boxes or not pred_boxes:
            return matches

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                iou_matrix[i, j] = self._bbox_iou(gt["bbox"], pred["bbox"])

        # Greedy matching
        used_pred = set()
        for i in range(len(gt_boxes)):
            if iou_matrix[i, :].max() > 0:
                j = int(iou_matrix[i, :].argmax())
                if j not in used_pred and iou_matrix[i, j] >= iou_threshold:
                    matches.append((gt_boxes[i]["track_id"], pred_boxes[j]["track_id"], float(iou_matrix[i, j])))
                    used_pred.add(j)

        return matches

    def _bbox_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    # ============================================================
    # EVENT DETECTION EVALUATION
    # ============================================================
    def evaluate_event_detection(
        self,
        gt_events: Dict[str, List[Dict]],
        pred_events: Dict[str, List[Dict]],
        tolerance_frames: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate event detection (passes, shots, goals, possession changes).

        Args:
            gt_events: Ground truth events by type
            pred_events: Predicted events by type
            tolerance_frames: Frame tolerance for matching events

        Returns:
            Dictionary with precision, recall, F1 for each event type.
        """
        event_types = ["passes", "shots", "goals", "possession_changes"]
        results = {}

        for event_type in event_types:
            gt_list = gt_events.get(event_type, [])
            pred_list = pred_events.get(event_type, [])

            # Match events by frame proximity
            tp, fp, fn = self._match_events(gt_list, pred_list, tolerance_frames)

            precision = self._safe_div(tp, tp + fp)
            recall = self._safe_div(tp, tp + fn)
            f1 = self._safe_div(2 * precision * recall, precision + recall)

            results[event_type] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        # Overall assessment
        overall_f1 = np.mean([v["f1"] for v in results.values()])
        passed = all(
            results[et]["precision"] >= getattr(self.thresholds, f"{et[:-1]}_precision_min", 0.5) and
            results[et]["recall"] >= getattr(self.thresholds, f"{et[:-1]}_recall_min", 0.5)
            for et in event_types if hasattr(self.thresholds, f"{et[:-1]}_precision_min")
        )

        return {
            "events": results,
            "overall_f1": round(overall_f1, 4),
            "passed": passed,
        }

    def _match_events(self, gt_events: List[Dict], pred_events: List[Dict], tolerance: int) -> Tuple[int, int, int]:
        """Match ground truth and predicted events."""
        if not gt_events:
            return 0, len(pred_events), 0
        if not pred_events:
            return 0, 0, len(gt_events)

        gt_frames = [e["frame"] for e in gt_events]
        pred_frames = [e["frame"] for e in pred_events]

        matched_pred = set()
        tp = 0

        for gt_frame in gt_frames:
            for i, pred_frame in enumerate(pred_frames):
                if i in matched_pred:
                    continue
                if abs(gt_frame - pred_frame) <= tolerance:
                    tp += 1
                    matched_pred.add(i)
                    break

        fp = len(pred_events) - tp
        fn = len(gt_events) - tp
        return tp, fp, fn

    # ============================================================
    # FORMATION DETECTION EVALUATION
    # ============================================================
    def evaluate_formation_detection(
        self,
        gt_formations: List[Dict],
        pred_formations: List[Dict],
    ) -> Dict[str, Any]:
        """Evaluate formation detection accuracy.

        Args:
            gt_formations: Ground truth formations [{frame, formation, team_id}]
            pred_formations: Predicted formations [{frame, formation, team_id, confidence}]

        Returns:
            Dictionary with formation metrics.
        """
        if not gt_formations or not pred_formations:
            return {"accuracy": 0.0, "passed": False}

        # Match by frame and team
        gt_by_key = {(g["frame"], g["team_id"]): g["formation"] for g in gt_formations}
        pred_by_key = {(p["frame"], p["team_id"]): p for p in pred_formations}

        common_keys = set(gt_by_key.keys()) & set(pred_by_key.keys())
        if not common_keys:
            return {"accuracy": 0.0, "passed": False}

        correct = sum(1 for k in common_keys if gt_by_key[k] == pred_by_key[k]["formation"])
        accuracy = correct / len(common_keys)

        # Confidence distribution
        confidences = [pred_by_key[k]["confidence"] for k in common_keys]
        mean_confidence = np.mean(confidences) if confidences else 0.0

        # Formation stability (consecutive same formations)
        stability = self._calculate_stability(pred_formations)

        # Change detection accuracy
        gt_changes = set()
        pred_changes = set()
        for team_id in set(g["team_id"] for g in gt_formations):
            team_gt = [g for g in gt_formations if g["team_id"] == team_id]
            team_pred = [p for p in pred_formations if p["team_id"] == team_id]
            gt_changes |= self._detect_changes(team_gt)
            pred_changes |= self._detect_changes(team_pred)

        if gt_changes:
            change_tp = len(gt_changes & pred_changes)
            change_precision = change_tp / len(pred_changes) if pred_changes else 0.0
            change_recall = change_tp / len(gt_changes)
        else:
            change_precision = 1.0 if not pred_changes else 0.0
            change_recall = 1.0

        passed = (
            accuracy >= self.thresholds.formation_accuracy_min and
            stability >= self.thresholds.formation_stability_min and
            mean_confidence >= self.thresholds.confidence_mean_min
        )

        return {
            "accuracy": round(accuracy, 4),
            "stability": round(stability, 4),
            "mean_confidence": round(mean_confidence, 4),
            "change_detection_precision": round(change_precision, 4),
            "change_detection_recall": round(change_recall, 4),
            "total_detections": len(pred_formations),
            "correct_detections": correct,
            "passed": passed,
        }

    def _calculate_stability(self, formations: List[Dict]) -> float:
        """Calculate formation stability (fraction of consecutive same formations)."""
        if len(formations) < 2:
            return 1.0

        team_formations: Dict[int, List[str]] = {}
        for f in formations:
            team_formations.setdefault(f["team_id"], []).append(f["formation"])

        stable_count = 0
        total_transitions = 0
        for team_id, seq in team_formations.items():
            for i in range(1, len(seq)):
                total_transitions += 1
                if seq[i] == seq[i - 1]:
                    stable_count += 1

        return self._safe_div(stable_count, total_transitions, default=1.0)

    def _detect_changes(self, formations: List[Dict]) -> set:
        """Detect frame numbers where formation changes occur."""
        changes = set()
        if len(formations) < 2:
            return changes

        sorted_f = sorted(formations, key=lambda x: x["frame"])
        for i in range(1, len(sorted_f)):
            if sorted_f[i]["formation"] != sorted_f[i - 1]["formation"]:
                changes.add(sorted_f[i]["frame"])
        return changes

    # ============================================================
    # PLAYER METRICS EVALUATION
    # ============================================================
    def evaluate_player_metrics(
        self,
        gt_player_metrics: List[Dict],
        pred_player_metrics: List[Dict],
        gt_heatmaps: Dict[int, np.ndarray],
        pred_heatmaps: Dict[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Evaluate player metric accuracy.

        Args:
            gt_player_metrics: Ground truth player metrics
            pred_player_metrics: Predicted player metrics
            gt_heatmaps: Ground truth heatmaps per player
            pred_heatmaps: Predicted heatmaps per player

        Returns:
            Dictionary with player metric errors.
        """
        # Match players by track_id
        gt_by_id = {p["track_id"]: p for p in gt_player_metrics}
        pred_by_id = {p["track_id"]: p for p in pred_player_metrics}

        common_ids = set(gt_by_id.keys()) & set(pred_by_id.keys())
        if not common_ids:
            return {"speed_error": 0.0, "distance_error": 0.0, "heatmap_iou": 0.0, "passed": False}

        # Speed and distance errors
        speed_errors = []
        distance_errors = []
        for tid in common_ids:
            gt = gt_by_id[tid]
            pred = pred_by_id[tid]
            speed_errors.append(abs(gt.get("max_speed_kmh", 0) - pred.get("max_speed_kmh", 0)))
            distance_errors.append(abs(gt.get("total_distance_m", 0) - pred.get("total_distance_m", 0)))

        mean_speed_error = np.mean(speed_errors) if speed_errors else 0.0
        mean_distance_error = np.mean(distance_errors) if distance_errors else 0.0

        # Heatmap IoU
        heatmap_ious = []
        for tid in common_ids:
            if tid in gt_heatmaps and tid in pred_heatmaps:
                iou = self._heatmap_iou(gt_heatmaps[tid], pred_heatmaps[tid])
                heatmap_ious.append(iou)
        mean_heatmap_iou = np.mean(heatmap_ious) if heatmap_ious else 0.0

        passed = (
            mean_speed_error <= self.thresholds.speed_error_max and
            mean_distance_error <= self.thresholds.distance_error_max and
            mean_heatmap_iou >= self.thresholds.heatmap_iou_min
        )

        return {
            "speed_error_kmh": round(mean_speed_error, 4),
            "distance_error_m": round(mean_distance_error, 4),
            "heatmap_iou": round(mean_heatmap_iou, 4),
            "num_players_evaluated": len(common_ids),
            "passed": passed,
        }

    def _heatmap_iou(self, gt: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> float:
        """Compute IoU between two heatmaps."""
        gt_bin = gt > threshold
        pred_bin = pred > threshold
        intersection = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()
        return intersection / union if union > 0 else 0.0

    # ============================================================
    # MASTER EVALUATION
    # ============================================================
    def evaluate_all(
        self,
        gt_data: Dict[str, Any],
        pred_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all evaluations.

        Args:
            gt_data: Ground truth data dictionary
            pred_data: Predicted data dictionary

        Returns:
            Complete evaluation report.
        """
        logger.info("Starting comprehensive evaluation...")

        # Tracking
        tracking_metrics = self.evaluate_tracking(
            gt_tracks=gt_data.get("tracks", []),
            pred_tracks=pred_data.get("tracks", []),
            total_gt_objects=gt_data.get("total_objects", 0),
        )

        # Event detection
        event_metrics = self.evaluate_event_detection(
            gt_events=gt_data.get("events", {}),
            pred_events=pred_data.get("events", {}),
        )

        # Formation detection
        formation_metrics = self.evaluate_formation_detection(
            gt_formations=gt_data.get("formations", []),
            pred_formations=pred_data.get("formations", []),
        )

        # Player metrics
        player_metrics = self.evaluate_player_metrics(
            gt_player_metrics=gt_data.get("player_metrics", []),
            pred_player_metrics=pred_data.get("player_metrics", []),
            gt_heatmaps=gt_data.get("heatmaps", {}),
            pred_heatmaps=pred_data.get("heatmaps", {}),
        )

        # Compute module scores
        module_scores = self._compute_module_scores({
            "tracking": tracking_metrics,
            "event_detection": event_metrics,
            "formation_detection": formation_metrics,
            "player_metrics": player_metrics,
        })

        # Overall pass/fail
        overall_passed = all(m.get("passed", False) for m in [tracking_metrics, event_metrics, formation_metrics, player_metrics])

        report = {
            "tracking": tracking_metrics,
            "event_detection": event_metrics,
            "formation_detection": formation_metrics,
            "player_metrics": player_metrics,
            "module_scores": module_scores,
            "overall_passed": overall_passed,
        }

        logger.info(f"Evaluation complete. Overall: {'PASS' if overall_passed else 'FAIL'}")
        return report

    def _compute_module_scores(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute 0-100 scores for each module."""
        scores = {}

        # Tracking score (weighted average)
        tracking = metrics["tracking"]
        tracking_score = (
            0.3 * (tracking.get("mota", 0) * 100) +
            0.2 * (tracking.get("motp", 0) * 100) +
            0.3 * (tracking.get("idf1", 0) * 100) +
            0.2 * (tracking.get("track_recall", 0) * 100)
        )
        scores["tracking"] = round(tracking_score, 2)

        # Event detection score
        events = metrics["event_detection"]
        event_scores = [v["f1"] * 100 for v in events.get("events", {}).values()]
        scores["event_detection"] = round(np.mean(event_scores) if event_scores else 0, 2)

        # Formation detection score
        formation = metrics["formation_detection"]
        formation_score = (
            0.4 * (formation.get("accuracy", 0) * 100) +
            0.3 * (formation.get("stability", 0) * 100) +
            0.3 * (formation.get("mean_confidence", 0) * 100)
        )
        scores["formation_detection"] = round(formation_score, 2)

        # Player metrics score
        player = metrics["player_metrics"]
        speed_score = max(0, 100 - (player.get("speed_error_kmh", 0) / self.thresholds.speed_error_max * 100))
        distance_score = max(0, 100 - (player.get("distance_error_m", 0) / self.thresholds.distance_error_max * 100))
        heatmap_score = player.get("heatmap_iou", 0) * 100
        scores["player_metrics"] = round(np.mean([speed_score, distance_score, heatmap_score]), 2)

        # Overall score
        scores["overall"] = round(np.mean(list(scores.values())), 2)

        return scores

    # ============================================================
    # REPORT GENERATION
    # ============================================================
    def generate_reports(self, evaluation: Dict[str, Any]) -> None:
        """Generate all evaluation reports."""
        # evaluation_report.json - detailed metrics
        with open(self.output_dir / "evaluation_report.json", "w") as f:
            json.dump(evaluation, f, indent=4)

        # module_scores.json - 0-100 scores
        with open(self.output_dir / "module_scores.json", "w") as f:
            json.dump(evaluation.get("module_scores", {}), f, indent=4)

        # evaluation_dashboard.json - summary for dashboard
        dashboard = {
            "overall_passed": evaluation.get("overall_passed", False),
            "overall_score": evaluation.get("module_scores", {}).get("overall", 0),
            "modules": {
                "tracking": {
                    "score": evaluation.get("module_scores", {}).get("tracking", 0),
                    "passed": evaluation.get("tracking", {}).get("passed", False),
                    "key_metric": f"MOTA={evaluation.get('tracking', {}).get('mota', 0):.2f}",
                },
                "event_detection": {
                    "score": evaluation.get("module_scores", {}).get("event_detection", 0),
                    "passed": evaluation.get("event_detection", {}).get("passed", False),
                    "key_metric": f"F1={evaluation.get('event_detection', {}).get('overall_f1', 0):.2f}",
                },
                "formation_detection": {
                    "score": evaluation.get("module_scores", {}).get("formation_detection", 0),
                    "passed": evaluation.get("formation_detection", {}).get("passed", False),
                    "key_metric": f"Acc={evaluation.get('formation_detection', {}).get('accuracy', 0):.2f}",
                },
                "player_metrics": {
                    "score": evaluation.get("module_scores", {}).get("player_metrics", 0),
                    "passed": evaluation.get("player_metrics", {}).get("passed", False),
                    "key_metric": f"SpeedErr={evaluation.get('player_metrics', {}).get('speed_error_kmh', 0):.1f}km/h",
                },
            },
            "thresholds": {
                "tracking": "MOTA>=0.6, MOTP>=0.5, IDF1>=0.5",
                "event_detection": "F1>=0.6 all events",
                "formation_detection": "Accuracy>=0.6, Stability>=0.6",
                "player_metrics": "SpeedError<=5km/h, DistanceError<=500m",
            },
        }
        with open(self.output_dir / "evaluation_dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=4)

        logger.info("Evaluation reports generated.")