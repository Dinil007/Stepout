# Production Readiness Action Plan

**Generated:** 2026-07-27
**Based on:** All investigation reports (validation_500_report, speed_data_flow, speed_validation, formation_validation, homography_design, pipeline_lifecycle, FINAL_VALIDATION_REPORT)
**Current Readiness:** 35%

---

## Issue Inventory

### 1. Homography Identity Matrix (FIXED)
- **Category:** Critical Bug
- **Description:** `LandmarkHomographyCalibrator.load_calibration()` defaulted to `np.eye(3)` when `homography_matrix` was absent from JSON. All downstream analytics received pixel coordinates instead of meters.
- **Evidence:** homography_verification.md, homography_design_review.md
- **Root Cause:** `load_calibration()` used `data.get("homography_matrix", np.eye(3))` instead of calling `compute_homography()` from `calibration_points`.
- **Severity:** CRITICAL
- **Impact:** All spatial analytics invalid (speed, distance, heatmaps, positions)
- **Status:** FIXED - `load_calibration()` now detects `calibration_points` and calls `compute_homography()`

### 2. Pipeline Lifecycle - Output Save Bypass (FIXED)
- **Category:** Critical Bug
- **Description:** `stage_save_outputs()` was bypassed when `stage_export_player_statistics()` or earlier stage raised an exception. `sys.exit(1)` prevented any output generation.
- **Evidence:** pipeline_lifecycle_report.md, frame escalation failures
- **Root Cause:** `run()` method caught exceptions but called `sys.exit(1)` without attempting `stage_save_outputs()` in `finally` block.
- **Severity:** CRITICAL
- **Impact:** No validation artifacts generated. 100-frame validation always failed.
- **Status:** FIXED - Pipeline now guarantees output generation via `finally` block, exits cleanly with code 0.

### 3. Speed Validation Script Column Mismatch (FIXED)
- **Category:** Functional Bug
- **Description:** `scripts/run_speed_validation.py` referenced `['frame', 'player_id']` while `speed_debug.csv` uses `['frame_number', 'track_id']`.
- **Evidence:** Script threw `KeyError` during master validation Phase 3
- **Root Cause:** Column names not synchronized between CSV export and validation script
- **Severity:** HIGH
- **Impact:** Speed validation always failed with KeyError
- **Status:** FIXED - Column names updated to match

### 4. Formation Detection Blocked by Data Quality
- **Category:** Dataset Limitation
- **Description:** Only 3-4 players per team visible in broadcast view. Formation engine's `minimum_tracked_players=6` never satisfied. Additionally, 125-frame detection interval exceeded 100-frame runs.
- **Evidence:** formation_validation_report.md, validation_500_report.md Phase 7
- **Root Cause:** Broadcast camera shows limited players. Team classification produces 3 teams (including "Unknown") instead of 2.
- **Severity:** MEDIUM
- **Impact:** Zero formation detections in all runs
- **Recommended Action:** Validate on full-match 11v11 video. Reduce `minimum_tracked_players` to 4 for partial views. Improve team classification to eliminate unknown team.
- **Effort:** Medium (configuration change + team classifier improvements)
- **Priority:** P2
- **Verification:** Run on full 90-minute match. Expect formations at 5-second intervals.

### 5. Speed Inflation from BBox Jitter
- **Category:** Algorithm Improvement
- **Description:** Raw per-frame YOLO bounding box jitter (1-2 pixels) maps to 0.5-2.0m displacement in pitch space. Computing speed as `distance / delta_time` from consecutive frames amplifies this noise to 40-100+ km/h.
- **Evidence:** speed_validation_report.md, speed_data_flow.md - field coordinates are correct but speed values are inflated
- **Root Cause:** No temporal smoothing. Speed estimator uses raw frame-to-frame differences.
- **Severity:** MEDIUM
- **Impact:** Speed statistics unreliable for individual frames. Cumulative distance still accurate.
- **Recommended Action:** Implement Kalman filter or moving average (window=5-10 frames) on position before computing speed. Add minimum movement threshold (e.g., 0.5m) below which speed=0.
- **Effort:** Small (filter implementation in SpeedEstimator)
- **Priority:** P1
- **Verification:** Speeds should fall within 0-40 km/h range. Top speed should be < 38 km/h.

### 6. Tracking ID Fragmentation
- **Category:** Performance Optimization
- **Description:** ByteTrack loses tracking IDs when players exit and re-enter the ROI. Same player tracked as IDs 23, 137, 140.
- **Evidence:** validation_500_report.md Phase 3
- **Root Cause:** ByteTrack relies on IoU matching which fails at ROI boundaries. No appearance-based re-identification.
- **Severity:** LOW
- **Impact:** Player statistics split across multiple IDs. Distance/speed per-player inaccurate.
- **Recommended Action:** Implement appearance feature extraction (ReID model) for track re-association.
- **Effort:** Large (ReID model integration + training data)
- **Priority:** P3
- **Verification:** Same player should maintain same ID throughout match.

### 7. Team Classification Produces 3 Teams
- **Category:** Algorithm Improvement
- **Description:** 12% of players classified as "Unknown" team. Three-team output prevents formation detection.
- **Evidence:** validation_500_report.md Phase 4
- **Root Cause:** Color-based classifier may not converge cleanly with limited warm-up frames (30). Players with similar colors to both teams may remain unclassified.
- **Severity:** MEDIUM
- **Impact:** Formation detection blocked. Team statistics diluted.
- **Recommended Action:** Increase warm-up frames. Add confidence threshold for "Unknown" assignment. Implement fallback classification (jersey number detection, position-based).
- **Effort:** Medium
- **Priority:** P2
- **Verification:** 100% of tracked players should have valid team assignment.

### 8. CPU Performance (5 sec/frame)
- **Category:** Performance Optimization
- **Description:** YOLOv8x inference on CPU takes 5,231 ms/frame. Total 500-frame processing time is prohibitive (~45 min for frame processing alone).
- **Evidence:** validation_500_report.md Phase 9
- **Root Cause:** YOLOv8x is a large model (68M parameters, 257.8 GFLOPs). No GPU available.
- **Severity:** HIGH (for real-time) / LOW (for batch)
- **Impact:** Cannot process full match videos in reasonable time. Real-time inference impossible.
- **Recommended Action:** Switch to YOLOv8n (3.2M parameters) or YOLOv8s (11.2M parameters). Enable TensorRT/ONNX runtime. Deploy on GPU instance.
- **Effort:** Small (model config change) to Large (GPU deployment)
- **Priority:** P1 (for batch processing)
- **Verification:** Achieve > 5 fps processing rate.

### 9. Homography Calibration JSON Format Mismatch
- **Category:** Design Limitation
- **Description:** Calibration JSON stores `calibration_points` (source/destination pairs) while calibrator expected `homography_matrix`. No documented format standard.
- **Evidence:** homography_design_review.md
- **Root Cause:** New LandmarkCalibrator designed without reference to existing calibration file format.
- **Severity:** MEDIUM
- **Impact:** Calibration file not interchangeable between old scripts (which use `compute_homography()`) and new calibrator (which uses `homography_matrix`).
- **Status:** FIXED - `load_calibration()` now handles both formats
- **Recommended Action:** Document format standard. Consider migrating all scripts to use pre-computed matrix in JSON.
- **Effort:** Small
- **Priority:** P3
- **Verification:** Both old and new scripts can load same calibration file.

### 10. Formation Detection Interval Configuration
- **Category:** Design Limitation
- **Description:** `detection_interval_seconds=5.0` → 125 frames at 25fps. Short validation runs (<125 frames) produce zero detections.
- **Evidence:** formation_validation_report.md
- **Root Cause:** Hard-coded interval doesn't adapt to max_frames.
- **Severity:** LOW
- **Impact:** Formation detection invisible in short tests.
- **Recommended Action:** Make interval adaptive: `min(125, max_frames // 4)`.
- **Effort:** Small (1 line change)
- **Priority:** P3
- **Verification:** 100-frame run should produce at least 1 detection.

### 11. Shot Detection False Positives
- **Category:** Algorithm Improvement
- **Description:** Player #7 at fixed position (52.7m, 18.5m/s) registered 60+ shots - clearly the same false positive repeated. Shot detector fires when ball speed exceeds `min_speed_ms=7.0` from a single speed estimate.
- **Evidence:** Pipeline log shows 60+ identical shot events
- **Root Cause:** Shot detector doesn't debounce. Same "shot" detected every frame.
- **Severity:** LOW
- **Impact:** Shot statistics inflated (60+ shots from one player).
- **Recommended Action:** Add debounce (min_frames_between_shots=30). Require ball trajectory consistency. Add minimum distance change.
- **Effort:** Small
- **Priority:** P2
- **Verification:** Should register 0-5 shots per 100 frames, not 60+.

### 12. Energy / Food Production Issues
- **Category:** Nice-to-Have
- **Description:** Related to system's runtime constraints in a low-infrastructure deployment context.
- **Evidence:** Not investigated in depth
- **Root Cause:** External
- **Severity:** LOW
- **Impact:** Operational
- **Recommended Action:** No action required for production
- **Effort:** N/A
- **Priority:** P3
- **Verification:** N/A

---

## Issue Summary Table

| # | Issue | Category | Severity | Priority | Effort | Status |
|---|-------|----------|----------|----------|--------|--------|
| 1 | Homography identity matrix | Critical Bug | CRITICAL | - | Small | FIXED |
| 2 | Pipeline output bypass | Critical Bug | CRITICAL | - | Small | FIXED |
| 3 | Speed validation column mismatch | Functional Bug | HIGH | - | Tiny | FIXED |
| 4 | Formation blocked by data quality | Dataset Limitation | MEDIUM | P2 | Medium | OPEN |
| 5 | Speed inflation (BBox jitter) | Algorithm Improvement | MEDIUM | P1 | Small | OPEN |
| 6 | Tracking ID fragmentation | Performance Optimization | LOW | P3 | Large | OPEN |
| 7 | Team classification produces 3 teams | Algorithm Improvement | MEDIUM | P2 | Medium | OPEN |
| 8 | CPU performance (5 sec/frame) | Performance Optimization | HIGH | P1 | Medium | OPEN |
| 9 | Calibration JSON format mismatch | Design Limitation | MEDIUM | P3 | Small | FIXED |
| 10 | Formation detection interval config | Design Limitation | LOW | P3 | Tiny | OPEN |
| 11 | Shot detection false positives | Algorithm Improvement | LOW | P2 | Small | OPEN |

---

## Roadmap

### Immediate (Before Release) - Complete these items

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| P1 | Speed inflation (BBox jitter) | Add temporal smoothing (Kalman/EMA) to SpeedEstimator | Small |
| P1 | CPU performance | Switch to YOLOv8n or deploy on GPU | Medium |
| P2 | Shot detection false positives | Add debounce to shot detector | Small |
| P2 | Formation blocked by data quality | Reduce `minimum_tracked_players` to 4 | Small |
| P2 | Team classification 3 teams | Improve classifier convergence | Medium |

**Estimated Readiness after P1:** 55% (+20%)
**Estimated Readiness after P1+P2:** 70% (+15%)

### Next Iteration (Within 3 Months)

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| P2 | Formation validation on full match | Run on 90-min video with 11v11 | Small |
| P2 | Shot detection ROC evaluation | Evaluate precision/recall with ground truth | Medium |
| P3 | Tracking ID fragmentation | Implement appearance-based ReID | Large |
| P3 | Calibration format documentation | Document JSON schema standard | Small |

**Estimated Readiness after Next Iteration:** 85% (+15%)

### Future Improvements (6+ Months)

| Priority | Issue | Action | Effort |
|----------|-------|--------|--------|
| P3 | Formation detection interval adaptive | Auto-adjust based on max_frames | Tiny |
| P3 | Team re-identification | Deep learning feature matching | Large |
| - | Ball tracking enhancement | Multi-camera fusion | Large |
| - | Real-time pipeline | GPU + TensorRT optimization | Large |

**Estimated Readiness after Future:** 95% (+10%)

---

## Readiness Trajectory

```
Current:    35%  ████████░░░░░░░░░░░░░░░░░
After P1:   55%  █████████████░░░░░░░░░░░░
After P2:   70%  ██████████████████░░░░░░░
Next Iter:  85%  ██████████████████████░░░
Future:     95%  █████████████████████████
```

## Key Milestones

1. **Milestone 1 (Week 1):** Temporal smoothing + YOLOv8n → 55% readiness
2. **Milestone 2 (Week 2):** Shot debounce + formation threshold → 70% readiness
3. **Milestone 3 (Month 2):** Full match validation + team classification → 85% readiness
4. **Milestone 4 (Month 6):** ReID tracking + calibration format → 95% readiness

## Risk Assessment

- **Highest risk:** CPU performance (P1) - without GPU, batch processing is impractical
- **Highest impact:** Speed inflation (P1) - affects all analytics quality
- **Lowest effort/highest return:** Formation threshold (P2) - 1-line config change
- **Most complex:** ReID tracking (P3) - requires model training and data