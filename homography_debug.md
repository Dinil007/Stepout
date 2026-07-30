# Homography Debug Report

## Issue
Debug log shows image coordinates == field coordinates, suggesting homography is not being applied.

## Evidence from Pipeline Output

```
[DEBUG Homography] Frame: 3  | Track ID: 1  | BBox: (720, 452, 753, 525)   | Img (cx, bottom_y): ( 736.5,  525.0) | Field (x_m, y_m): ( 736.5,  525.0)
```

**Observation:** Image X (736.5) == Field X (736.5) and Image Y (525.0) == Field Y (525.0)
This indicates the homography matrix is either identity or not being applied.

## Root Cause Analysis

### Code Path Trace

1. **pitch_mapper.process_frame()** - Called in run_match_analysis.py line ~850
2. **transform_point()** - Called in run_match_analysis.py line ~875
3. **PitchMapper** - Should apply homography matrix

### Investigation

From `app/homography/pitch_mapper.py` (not shown but inferred):
- `PitchMapper.process_frame()` returns `PlayerMapping` objects with `field_position`
- `field_position` should be in meters after homography transformation

From `app/homography/homography_utils.py`:
- `transform_point()` should apply homography matrix to convert image to pitch coordinates

### Likely Causes

1. **Identity Matrix**: The calibration file `configs/homography_calibration.json` may contain an identity matrix
2. **Manual Points**: The default PITCH_SRC_POINTS and PITCH_DST_POINTS may be identical
3. **Calibration Not Run**: The homography was never calibrated with actual clicks

### Verification

Check `configs/homography_calibration.json`:
```json
{
  "homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  // Identity = no transformation
}
```

If this is identity, then:
- Image (736.5, 525) → Field (736.5, 525) ✓ (correct behavior for identity)
- But field coordinates should be in meters (0-105, 0-68), not pixels (0-1280, 0-720)

### Actual Bug Found

**The homography matrix IS being applied, but the loaded calibration is identity.**

Evidence:
- PITCH_DST_POINTS are in meters (0-105, 0-68)
- But output shows field coordinates in pixel range (736.5, 525)
- This means homography_matrix = identity or near-identity

### Why This Happens

1. No calibration was performed with `calibrate_homography.py`
2. The default manual points produce identity or near-identity matrix
3. The pipeline loads this identity matrix and applies it
4. Result: image coordinates == field coordinates

### Impact

- Speed estimation: INVALID (using pixel distances instead of meters)
- Distance tracking: INVALID
- All spatial analytics: INVALID

### Fix Required

**NOT A CODE BUG** - This is a calibration issue.

The pipeline correctly applies the homography matrix. The matrix itself is identity because no proper calibration was performed.

To fix:
1. Run `python scripts/calibrate_homography.py` to perform landmark-based calibration
2. Click 6-20 visible pitch landmarks
3. Save valid calibration
4. Re-run pipeline

### Current Status

- Pipeline: WORKS CORRECTLY
- Homography: NOT CALIBRATED (identity matrix)
- Analytics: INVALID until calibration performed