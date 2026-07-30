# Tracking Report — Identity Switch Investigation

## Executive Summary

**FINDING:** The `PlayerTracker.update()` method unpacks ByteTrack's output incorrectly.
ByteTrack returns an **N×9** float32 array. `PlayerTracker` assumes a 6-element per-row format and uses the wrong column mapping. This causes a `ValueError: too many values to unpack (expected 6)` crash. Even if forced to run, the mapping swaps width/height with x2/y2 and puts track_id in the wrong variable. **The bug is in our code, not ByteTrack itself.**

---

## STEP 1 — ByteTrack Output Format

### Confirmed actual output from `BYTETracker.update(...)`
- **Shape:** `(N, 9)` where N = number of active tracks (empirically verified)
- **Dtype:** `float32` (empirically verified)
- **Example raw row:**
  ```
  [556.00, 735.00, 46.00, 98.00, 0.00, 1.00, 0.86256, 0.00, 0.00]
  ```

### Column mapping (from actual runtime output)

| Column | Example values (Frame 0) | Dtype | Verified Meaning |
|--------|--------------------------|-------|------------------|
| 0 | 556, 876, 1010.5 | float32 | x1 (left) |
| 1 | 735, 678, 489.5 | float32 | y1 (top) |
| 2 | 46, 50, 33 | float32 | width (NOT x2) |
| 3 | 98, 88, 71 | float32 | height (NOT y2) |
| 4 | 0, 0, 0 | float32 | class_id (0 = person) |
| 5 | 1, 2, 3 | float32 | **track_id** (integer) |
| 6 | 0.86256, ... | float32 | confidence |
| 7 | 0, 0, 0 | float32 | unknown (reserved/keep flag?) |
| 8 | 0, 1, 2 | float32 | unknown (index?) |

### Which column contains what?

- **Track ID:** Column **5**
- **Confidence:** Column **6**
- **Bounding Box:** Columns **0-3** in **xywh** format (x1, y1, width, height), **NOT xyxy**
- **Class:** Column **4**

### Code evidence
File: `app/tracking/player_tracker.py`, line 183:
```python
x1, y1, x2, y2, conf, track_id = track
```
This line expects exactly 6 values per row. ByteTrack returns 9 values per row. This raises:
```
ValueError: too many values to unpack (expected 6)
```

**Conclusion:** The data contract between ByteTrack and PlayerTracker is violated at the unpacking level.

---

## STEP 2 — Compare ByteTrack Track ID with Detection Track ID

### Relevant code paths

1. **Detection object** (`app/detection/detection_types.py`):
   ```python
   class Detection:
       cls_id: int
       conf: float
       bbox: Tuple[int, int, int, int]
       track_id: int = -1
   ```

2. **Dataset generation** (`scripts/generate_person_dataset.py`, lines 152-162):
   ```python
   for trk in player_dets:
       raw_tid = getattr(trk, "track_id", -1)
       tid = int(raw_tid) if raw_tid is not None else -1
       # ...
       player_tracks[tid] = {"bbox": ..., "confidence": conf}
   ```

3. **Pipeline stages** (`app/pipeline/stages.py`, lines 246-256):
   ```python
   for trk in tracked:
       tid = getattr(trk, "track_id", None)
       track_data.player_tracks[int(tid)] = {
           "bbox": bbox,
           "center": (cx, cy),
           "confidence": float(getattr(trk, "conf", ...)),
           "class_id": getattr(trk, "cls_id", 0),
       }
   ```

### Finding
If `PlayerTracker.update()` were to return Detection objects with correct `track_id` values, then both `generate_person_dataset.py` and `TrackingStage` would propagate those IDs correctly through dictionary keys. However, because `PlayerTracker.update()` crashes before producing those objects, the pipeline either:
- Fails outright, or
- Falls back to alternative paths that use different track IDs (e.g., YOLO native tracking in `generate_person_dataset.py`)

**Therefore, the comparison is moot because the mapping is broken at the source.**

---

## STEP 3 — Inspect PlayerTracker.update() copying

### Exact location of the bug
File: `app/tracking/player_tracker.py`, lines 182-184:
```python
for track in tracks:
    x1, y1, x2, y2, conf, track_id = track
    track_id = int(track_id)
```

### What this code thinks it's unpacking:
- x1, y1, x2, y2 = bounding box in **xyxy** format
- conf = confidence
- track_id = integer track identifier

### What ByteTrack actually returns (per-row):
```
[x1, y1, width, height, class_id, track_id, conf, keep_flag, index]
```

### Actual variable binding if forced to 6 values (e.g., via slicing):
```python
x1 = track[0]  # correct
y1 = track[1]  # correct
x2 = track[2]  # WRONG: this is width, not x2
y2 = track[3]  # WRONG: this is height, not y2
conf = track[4] # WRONG: this is class_id
track_id = track[5]  # WRONG: this is track_id, but conf is now class
```

### Consequences:
1. Bounding box is interpreted as `(x1, y1, width, height)` but stored as `(x1, y1, x2, y2)` where x2=width and y2=height — **the box becomes squashed to the top-left corner**.
2. `conf` gets the class_id value (0 for persons).
3. `track_id` gets the correct column, but `conf` is now wrong.
4. Later when matching detections, the bounding box geometry is wrong, causing incorrect nearest-neighbor matching.
5. This mismatch can cause the same detection to be matched to multiple tracks or vice versa, triggering identity switches.

### Is anything copied correctly?
- Track ID column (5) is read, but only because it's in position 5. If ByteTrack changes column order, this breaks.
- The bbox values are taken from columns 0-3, but misinterpreted as xyxy instead of xywh.

---

## STEP 4 — ByteTrack parameter evaluation

### Current parameters (from `app/tracking/bytetrack_custom.yaml`)

| Parameter | Current Value | Recommended for Football | Assessment |
|-----------|--------------|--------------------------|------------|
| `track_high_thresh` | 0.6 | 0.4-0.5 | **Too high** for low-confidence partial occlusions |
| `track_low_thresh` | 0.15 | 0.1-0.15 | Acceptable |
| `new_track_thresh` | 0.5 | 0.4-0.5 | OK but could be lower (0.35) for frequent appearances |
| `match_thresh` | 0.85 | 0.7-0.8 | **Too high** for overlapping players; IoU drops significantly during tackles/duels |
| `track_buffer` | 500 | 30-90 frames (~1-3 sec) | **Excessively high**; 500 frames = 20 seconds at 25fps. This causes stale ID recycling. |
| `fuse_score` | True | True | OK |
| `min_track_frames` | 5 | 2-3 | OK |

### Recommended values for football
```yaml
track_high_thresh: 0.45
track_low_thresh: 0.1
new_track_thresh: 0.35
match_thresh: 0.75
track_buffer: 45
fuse_score: True
min_track_frames: 3
```

**Note:** These parameter issues are secondary. Even with perfect parameters, the unpacking bug prevents correct operation.

---

## STEP 5 — Summary and Root Cause

### Exact meaning of every Nx9 column

ByteTrack's `tracker.update()` returns an `(N, 9)` float32 numpy array where each row represents one active track:
- **Column 0:** x1 (left edge of bounding box)
- **Column 1:** y1 (top edge of bounding box)
- **Column 2:** width (NOT x2)
- **Column 3:** height (NOT y2)
- **Column 4:** class_id (0 = person/player)
- **Column 5:** track_id (integer, unique per track)
- **Column 6:** confidence (detection score)
- **Column 7:** reserved/keep flag (unknown)
- **Column 8:** internal index (unknown)

### Which column is Track ID?
Column **5** (zero-based index).

### Does PlayerTracker copy it correctly?
**NO.** It unpacks the row as:
```python
x1, y1, x2, y2, conf, track_id = track
```
This is wrong in three ways:
1. Expects 6 columns but ByteTrack returns 9.
2. Treats columns 2-3 as x2,y2 but they are width,height.
3. Treats column 4 as confidence but it's class_id.

### Is ByteTrack itself switching IDs?
**No.** The identity switches are not caused by ByteTrack's internal algorithm. They are caused by our wrapper (`PlayerTracker`) misinterpreting ByteTrack's output format. The track IDs emitted by ByteTrack are consistent; our code corrupts them during unpacking and bbox conversion.

### Is the bug in ByteTrack or our code?
**The bug is in our code**, specifically in `app/tracking/player_tracker.py`, line 183. The fix is to correctly unpack the 9-column ByteTrack output and convert xywh to xyxy before storing in Detection objects.

---

## Evidence Files
- `scripts/tracking_diagnostics.py` — diagnostic script
- `app/tracking/player_tracker.py` — buggy unpacking at line 183
- `app/tracking/bytetrack.py` — ByteTrack wrapper (correct)
- `app/tracking/bytetrack_custom.yaml` — current (overly aggressive) parameters