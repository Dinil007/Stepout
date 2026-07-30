# Performance and Homography Review

**Date:** 2026-07-27
**Status:** Evidence-based investigation only - NO CODE CHANGES

---

## 1. CPU Performance Analysis

### Measured Performance (100-frame run, YOLOv8x CPU)

| Module | ms/frame | % of total |
|--------|----------|------------|
| YOLO Detection | 5,231.46 | 62.3% |
| Tactical Save | 1,168.04 | 13.9% |
| Intelligence Engine | 1,360.62 | 16.2% |
| Pass Network Viz | 317.77 | 3.8% |
| Tactical Analytics | 140.01 | 1.7% |
| MediaPipe Pose | 50.48 | 0.6% |
| Pass Network Analysis | 109.18 | 1.3% |
| Team Classification | 1.76 | 0.02% |
| Homography | 0.14 | 0.002% |
| Speed Estimator | 0.12 | 0.001% |
| **Total** | **8,387.72** | **100%** |

**Average FPS:** 0.1 (10 seconds per frame)

### Bottleneck Identification

**Primary:** YOLOv8x inference on CPU (5.2 sec/frame)
- Model: 68M parameters, 257.8 GFLOPs
- Device: CPU (no CUDA)
- Impact: 62% of total runtime

**Secondary:** Output generation (3,936 ms/frame)
- Tactical save: 1,168 ms
- Intelligence engine: 1,361 ms
- Pass network viz: 318 ms
- Combined: 46.9% of runtime

**Tertiary:** MediaPipe Pose (50 ms/frame) - negligible

### Runtime Estimates

| Scenario | Frames | Time (current) | Notes |
|----------|--------|----------------|-------|
| 500 frames | 500 | ~70 min | Currently running |
| 1000 frames | 1000 | ~140 min | 2.3 hours |
| Full match (~18k frames) | 18,000 | ~50 hours | ~2 days |
| Full match (90 min @ 25fps) | 135,000 | ~375 hours | ~15.6 days |

### Model Comparison (CPU inference estimates)

| Model | Params | GFLOPs | Est. ms/frame | Est. FPS | 500-frame time |
|-------|--------|--------|---------------|----------|----------------|
| YOLOv8x | 68M | 257.8 | 5,200 | 0.19 | 43 min |
| YOLOv8m | 26M | 98.5 | 2,100 | 0.48 | 17 min |
| YOLOv8n | 3.2M | 10.7 | 200 | 5.0 | 1.7 min |
| YOLOv8s | 11.2M | 28.6 | 600 | 1.7 | 4.9 min |

**GPU comparison (estimated):**
| Model | Device | Est. ms/frame | Est. FPS | 500-frame time |
|-------|--------|---------------|----------|----------------|
| YOLOv8x | GPU | 15-30 | 33-67 | 7.5-15 sec |
| YOLOv8m | GPU | 8-15 | 67-125 | 4-7 sec |
| YOLOv8n | GPU | 2-5 | 200-500 | 1-2.5 sec |

### Conclusion

**Bottleneck is YOLOv8x on CPU.** Switching to YOLOv8n would improve speed by ~26x. Switching to GPU would improve speed by ~170x.

**Recommendation:** JUSTIFIED to switch model OR deploy GPU. Both are evidence-based improvements.

---

## 2. Homography Validation

### Observed Coordinates

Debug output shows field positions:
- Track ID 2, Frame 3: **124.4 m** (expected max: 105 m)
- Track ID 2, Frame 4: **125.2 m**
- Track ID 2, Frame 5: **125.9 m**
- Track ID 2, Frame 6: **126.7 m** (continued)
- Track ID 3, Frame 10: **93.3 m** (within bounds)
- Track ID 2, Frame 10: **130.3 m** (further extrapolation)

### Calibration Points

```json
"calibration_points": {
  "source": [
    [50.0, 300.0],
    [1000.0, 300.0],
    [1050.0, 600.0],
    [0.0, 680.0]
  ],
  "destination": [
    [0.0, 0.0],
    [105.0, 0.0],
    [105.0, 68.0],
    [0.0, 68.0]
  ]
}
```

### Root Cause Analysis

**Question:** Is the homography mathematically correct?

**Answer:** YES. A perspective transform (3x3 matrix) maps the calibrated quadrilateral to the rectangular pitch. The transform is mathematically valid.

**Question:** Are detections outside the pitch actual players?

**Answer:** LIKELY YES. Players running along the touchline or in goal areas can be legitimately near the edge. However, coordinates >105m indicate extrapolation beyond the pitch boundary.

**Question:** Is extrapolation occurring?

**Answer:** YES. `cv2.perspectiveTransform()` extrapolates for any point in the image, even if outside the calibration polygon. The calibration source points form a quadrilateral that doesn't cover the entire frame. Points outside this quadrilateral get transformed to coordinates outside [0, 105] x [0, 68].

### Why coordinates exceed 105m

The calibration source points:
- [50, 300] → [0, 0]
- [1000, 300] → [105, 0]
- [1050, 600] → [105, 68]
- [0, 680] → [0, 68]

This defines a quadrilateral that covers most of the pitch but NOT the entire frame. Players detected near the image edges (x > 1050 or y < 300) get mapped to coordinates outside the pitch.

### Downstream Impact Assessment

| Module | Impact | Severity |
|--------|--------|----------|
| Speed estimation | Coordinates >105m are valid inputs; speed computed from displacement | LOW |
| Distance tracker | Cumulative distance may include out-of-bounds segments | LOW |
| Heatmaps | Out-of-bounds positions render outside pitch canvas (clipped or off-canvas) | MEDIUM |
| Tactical analytics | Average positions, territory control, shape analysis may include out-of-bounds | MEDIUM |
| Formation detection | Positions >105m may skew formation centroids | MEDIUM |
| Pass detection | Passes to/from out-of-bounds players may be invalid | MEDIUM |
| Shot detection | Ball position may be out-of-bounds; shot location invalid | MEDIUM |

### Expected Behavior of cv2.perspectiveTransform()

`cv2.perspectiveTransform()` is a continuous transformation. For any point in the source image, it computes a destination point. There is NO built-in clipping or rejection. Out-of-bounds results are mathematically valid but physically meaningless for pitch analytics.

### Recommendations (Evidence-Based)

**Option A: Leave unchanged**
- Pro: Simplicity; preserves all data
- Con: Analytics modules must handle out-of-bounds gracefully
- Risk: MEDIUM - tactical outputs may be skewed

**Option B: Clip coordinates to [0, 105] x [0, 68]**
- Pro: Ensures all positions are physically possible
- Con: Loses information about proximity to pitch edge; may distort speed/distance near boundaries
- Risk: LOW - but introduces edge artifacts

**Option C: Flag/reject out-of-bounds positions**
- Pro: Cleanest solution; downstream modules can skip invalid positions
- Con: Loses data for players at touchlines/goal areas
- Risk: LOW - but may reduce valid detections

**Recommended: Option A with defensive coding**
- Leave homography unchanged (it's mathematically correct)
- Add bounds checking in downstream modules that produce aggregate statistics (heatmaps, tactical, formation)
- Log out-of-bounds events for diagnostics

---

## 3. Speed Validation

### Current Status

The 500-frame pipeline is still running. `speed_debug.csv` has not yet been generated with the new smoothing parameters.

### Baseline Data (Previous run, no smoothing)

From `speed_validation_report.md`:
```json
{
  "min_kmh": 0.0,
  "avg_kmh": 49.98,
  "median_kmh": 45.28,
  "max_kmh": 109.01,
  "over_30_kmh": 157,
  "over_35_kmh": 138,
  "over_40_kmh": 119
}
```

### Expected After Smoothing

Based on `speed_smoothing_report.md`:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max speed | 109.01 km/h | ~35-38 km/h | -65% |
| Mean speed | 49.98 km/h | ~18 km/h | -60% |
| Median speed | 45.28 km/h | ~15 km/h | -67% |
| 95th percentile | 93.72 km/h | ~32 km/h | -66% |
| Spikes >35 km/h | 157 | ~5-10 | -95% |
| Spikes >40 km/h | 138 | ~0-2 | -99% |
| Spikes >50 km/h | 119 | 0 | -100% |

### Verification Method

Once `speed_debug.csv` is generated from the current 500-frame run:

```python
import pandas as pd
df = pd.read_csv('outputs/speed_debug.csv')
print(df['speed_kmh'].describe())
print(f"Count >35: {(df['speed_kmh'] > 35).sum()}")
print(f"Count >40: {(df['speed_kmh'] > 40).sum()}")
```

### Current Evidence

From the running pipeline log (frames 1-10):
- Track ID 2: 74-83 km/h (before smoothing: same range)
- Track ID 3: 45-84 km/h
- No visible change in early frames

**Note:** EMA smoothing requires multiple frames to converge. The first few frames will show similar values to before. Smoothing effects become apparent after 10+ frames per track.

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CPU performance prevents production use | HIGH | CRITICAL | Deploy GPU or switch to YOLOv8n |
| Out-of-bounds homography coordinates skew analytics | MEDIUM | MEDIUM | Add bounds checking in tactical/heatmap modules |
| Speed smoothing not effective enough | LOW | MEDIUM | Tune alpha after seeing actual CSV data |
| Shot detection false positives | MEDIUM | LOW | Already documented in PRODUCTION_ACTION_PLAN.md |
| Formation detection blocked by insufficient players | HIGH | MEDIUM | Dataset limitation; requires full 11v11 view |

---

## 5. Recommendations

### Immediate (P1)

1. **Performance:** Deploy on GPU OR switch to YOLOv8n
   - Evidence: 5.2 sec/frame on CPU; 0.1 FPS
   - Impact: 500-frame run takes 70+ minutes
   - Effort: Small (model swap) or Medium (GPU deployment)

2. **Homography bounds checking:** Add optional clipping/rejection in `PitchMapper` or downstream consumers
   - Evidence: Coordinates >130m observed; pitch is 105x68m
   - Impact: Tactical analytics and heatmaps may include invalid positions
   - Effort: Small (optional flag in config)

### Next Iteration (P2)

3. **Speed validation:** Analyze actual `speed_debug.csv` from completed 500-frame run
   - Evidence: Baseline stats documented; smoothing parameters configured
   - Impact: Confirm +10% readiness improvement
   - Effort: Small (data analysis)

4. **Shot detection debounce:** Add cooldown period to reduce false positives
   - Evidence: 60+ shots from single player in 100 frames
   - Impact: Shot statistics reliability
   - Effort: Small

### Future (P3)

5. **Adaptive smoothing:** Tune EMA alpha based on camera angle/zoom
   - Evidence: Current alpha=0.15 may be too aggressive for some views
   - Impact: Speed accuracy
   - Effort: Medium

---

## Conclusion

**Performance:** CPU is the primary blocker. Evidence strongly supports model downgrade or GPU deployment.

**Homography:** Mathematically correct per `cv2.perspectiveTransform()`. Out-of-bounds coordinates are expected behavior for points outside the calibration quadrilateral. Impact on downstream analytics is manageable with defensive coding.

**Speed:** Smoothing code is in place; awaiting final CSV validation from running pipeline.

**No source code modifications recommended until data is verified.**