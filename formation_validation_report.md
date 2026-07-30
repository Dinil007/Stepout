# Formation Validation Report

## Executive Summary

**Root Cause:** The formation detection interval (125 frames) exceeds the total number of processed frames (100). No detection frame is ever reached.

**Status:** NOT A BUG - DESIGN LIMITATION

---

## 1. Root Cause Analysis

### Exact Line Responsible

**File:** `app/analytics/automatic_formation_engine.py`, **Line 214**

```python
def process_frame(self, frame_number, players, team_assignments):
    # Check if this is a detection frame
    if frame_number % self.detection_interval_frames != 0:
        return []
```

### Why This Causes Zero Detections

**Configuration:**
- `fps = 25` (from video)
- `detection_interval_seconds = 5.0` (default)
- `detection_interval_frames = int(5.0 * 25) = 125`

**Execution:**
- Pipeline processes frames 1 through 100
- Detection only triggers when `frame_number % 125 == 0`
- **No frame in 1-100 satisfies this condition**
- `process_frame()` returns `[]` for every single frame

**Result:** Zero snapshots, zero detections, zero changes.

---

## 2. Evidence

### Pipeline Log Output

```
[2026-07-27 11:58:05] [INFO] [app.analytics.automatic_formation_engine]: 
Formation analysis saved: 0 detections, 0 changes
```

### Module Timing

```
Formation .......................... 0.01 ms
```

The 0.01 ms timing confirms the function returned immediately without performing any work.

### Frame-by-Frame Analysis

| Frame | Frame % 125 | Detection Triggered | Players | Result |
|-------|-------------|---------------------|---------|--------|
| 20 | 20 | NO | 8 | [] |
| 40 | 40 | NO | 8 | [] |
| 60 | 60 | NO | 8 | [] |
| 80 | 80 | NO | 8 | [] |
| 100 | 100 | NO | 8 | [] |
| 125 | 0 | YES | N/A | Not processed |

---

## 3. Secondary Issues (Would Apply Even with 125+ Frames)

### Issue A: Insufficient Players Per Team

**File:** `app/analytics/automatic_formation_engine.py`, **Line 229**

```python
if len(player_positions) < self.config.minimum_tracked_players:
    continue
```

- `minimum_tracked_players = 6`
- Pipeline detects 8 players total across 3 teams
- Even at 125 frames, per-team counts would be:
  - Team 0: ~3-4 players
  - Team 1: ~3-4 players
  - Unknown: ~1 player
- **None of these meet the minimum of 6**

### Issue B: Three Teams Instead of Two

- Team classification produces 3 teams (Team 0, Team 1, Unknown)
- Formation engine expects exactly 2 teams
- The "Unknown" team dilutes the player pool

### Issue C: Confidence Threshold

**File:** `app/analytics/automatic_formation_engine.py`, **Line 190**

```python
if detection.confidence < self.min_confidence:
    return None
```

- `min_confidence = 0.6`
- Even if detection ran, noisy data would likely produce confidence < 0.6

---

## 4. Data Flow Trace

```
YOLO Detection
  ↓ 8 players detected
ByteTrack
  ↓ 8 players tracked (IDs: 1, 2, 3, 5, 7, 23, 137, 140)
Team Classification
  ↓ 3 teams (0, 1, Unknown)
Player Positions (meters)
  ↓ Coordinates in pitch range
Formation Engine
  ↓ process_frame() called 100 times
  ↓ Line 214: frame_number % 125 != 0 → return []
  ↓ 0 snapshots generated
```

**Information lost at:** Line 214 of `automatic_formation_engine.py`

---

## 5. Conclusion

**STATUS: NOT A BUG - DESIGN LIMITATION**

### Root Cause

The formation detection interval (125 frames = 5 seconds at 25 fps) exceeds the total processed frames (100). The modulo check at line 214 prevents any detection from occurring.

### Data Issue vs Algorithm Issue

| Aspect | Type | Details |
|--------|------|---------|
| Detection interval > total frames | Design limitation | 125 > 100 |
| Insufficient players per team | Data issue | 8 total, need 6 per team |
| Three teams instead of two | Data issue | Team classification noise |
| Confidence threshold | Algorithm issue | 0.6 may be too high for noisy data |

**Primary cause:** Design limitation (detection interval)
**Secondary causes:** Data quality (insufficient players, team noise)

### Recommended Fix

**Option A:** Reduce detection interval for short runs
```python
# In run_match_analysis.py initialization:
self.formation_engine = AutomaticFormationEngine(
    fps=self.fps,
    detection_interval_seconds=1.0,  # Reduced from 5.0
    ...
)
```

**Option B:** Process more frames (500+)
- Run pipeline with `--max-frames 500` to reach frame 125, 250, 375, 500

**Option C:** Make detection interval adaptive
- Use `min(detection_interval_frames, max_frames // 4)` to ensure at least 4 detection opportunities

### Current Status

- Formation engine: WORKING CORRECTLY
- Configuration: NOT MATCHING INPUT DATA (125-frame interval vs 100-frame run)
- Result: CORRECT (0 detections given the constraints)