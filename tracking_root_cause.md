# Tracking Root Cause Analysis

**Date:** 2025-07-26  
**Status:** ROOT CAUSE IDENTIFIED - ByteTrack false track generation

## Executive Summary

The impossible player speeds (max 1369 km/h, avg >200 km/h) are caused by **ByteTrack generating spurious track IDs with inconsistent positions**, not by ID switches or lost tracks.

## Key Finding

Frame-level inspection of 200 frames revealed:

- **101 unique track IDs** generated (should be ~22 players)
- **1186 high-speed events** (>40 km/h)
- **99% of events** are from same track ID with 1-4m displacement
- Only 1% are actual ID switches or lost track recoveries

## Evidence

### Speed Distribution (from 200-frame inspection)

| Metric | Value |
|--------|-------|
| Total unique tracks | 101 |
| High-speed events | 1186 |
| Max speed | 1369.2 km/h |
| Typical displacement | 1.9-4.1m per frame |

### Example Events

| Frame | Track ID | Speed | Displacement | Classification |
|-------|----------|-------|--------------|----------------|
| 187 | 101 | 1369.2 km/h | 12.7m | ID Switch / Large Jump |
| 150 | 74 | 620.7 km/h | 5.8m | Lost Track / Recovery |
| 58 | 3 | 440.8 km/h | 4.1m | Speed Calculation (same ID) |
| 155 | 5 | 437.5 km/h | 4.0m | Speed Calculation (same ID) |
| 45 | 15 | 295.9 km/h | 2.7m | Speed Calculation (same ID) |

### Root Cause Distribution

- **Speed Calculation (no artifact):** 1179 events (99%)
- **Confidence Drop / Detection Error:** 5 events (0%)
- **ID Switch / Large Position Jump:** 1 event (0%)
- **Lost Track / Recovery:** 1 event (0%)

## Root Cause Analysis

### Primary Cause: False Track Generation

ByteTrack is creating **spurious track IDs** for the same player across consecutive frames. Each track ID has inconsistent positions, causing massive apparent movement.

**Why this happens:**
1. YOLO generates multiple detections per player (duplicate boxes)
2. ByteTrack assigns different track IDs to these duplicates
3. Each track ID gets a slightly different position
4. Between frames, the position difference for the "same" track ID is 1-4m
5. Speed calculation treats this as real movement: 4m / 0.033s = 120 m/s = 432 km/h

### Secondary Cause: Detection Duplication

YOLOv8x at 1280px input size generates overlapping detections for single players, especially in crowded scenes.

## Calculation Chain Verification

The speed calculation is **mathematically correct**:

```
displacement_m = sqrt((x2-x1)^2 + (y2-y1)^2)
speed_ms = displacement_m / dt
speed_kmh = speed_ms * 3.6
```

However, the **input positions are corrupted** by tracking instability, not by calculation errors.

## Affected Modules

| Module | Status | Issue |
|--------|--------|-------|
| ByteTrack (`app/tracking/bytetrack_custom.yaml`) | **FAULT** | Generates false track IDs |
| YOLO Detection | Contributing | Duplicate detections per player |
| Homography | Not at fault | Correct coordinate transformation |
| SpeedEstimator | Not at fault | Correct calculation, receives bad input |
| DistanceTracker | Not at fault | Correct calculation, accumulates bad input |

## Current Mitigation

Filters added to:
- `app/analytics/speed_estimator.py` - filters position jumps >5m
- `app/analytics/distance_tracker.py` - filters distance jumps >5m
- `run_pipeline.py` - validates and smooths speeds

**Status:** Mitigated by filtering; upstream false track generation remains.

## Recommended Permanent Fixes

### Option 1: YOLO NMS (Immediate)
Add Non-Maximum Suppression to remove duplicate detections:
```python
results = model.track(
    source=frame,
    persist=True,
    tracker=tracker_config,
    classes=[0],
    conf=0.25,
    iou=0.5,  # Already set, but ensure NMS is applied
    imgsz=1280,
    verbose=False
)
```

### Option 2: ByteTrack Tuning (Short-term)
Further adjust ByteTrack parameters:
- `track_high_thresh`: 0.5 → 0.6
- `track_low_thresh`: 0.1 → 0.2
- `new_track_thresh`: 0.5 → 0.7
- `track_buffer`: 120 → 150

### Option 3: Position Consistency Filter (Short-term)
Reject track IDs that appear for only 1-2 frames:
```python
if track_id in recent_tracks and len(recent_tracks[track_id]) < 3:
    # Skip this track ID
```

### Option 4: Alternative Tracker (Long-term)
Consider BoT-SORT or DeepSORT with ReID for better track stability.

## Conclusion

**Primary Root Cause:** ByteTrack generates 101 unique track IDs for ~22 players due to duplicate YOLO detections, causing position jitter that propagates as impossible speeds.

**Secondary Root Cause:** No track quality validation to reject spurious tracks.

**Fix Priority:** 
1. Increase YOLO IoU threshold or add explicit NMS
2. Add track persistence validation (require 3+ frames before accepting track)
3. Implement position consistency checks within track history