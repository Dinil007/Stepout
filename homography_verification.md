# Homography Verification Report

## Investigation Summary

**Status:** HOMOGRAPHY NOT APPLIED - IDENTITY MATRIX LOADED

---

## 1. Calibration File Contents

**File:** `configs/homography_calibration.json`

```json
{
  "field_dimensions": {
    "length_m": 105.0,
    "width_m": 68.0
  },
  "image_dimensions": {
    "width_px": 1050,
    "height_px": 680
  },
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
  },
  "validation": {
    "mean_reprojection_error": 1.5,
    "m_per_px_cv": 0.1,
    "validation_passed": true
  },
  "method": "manual"
}
```

**Critical Observation:** The file contains `calibration_points` but NO `homography_matrix`.

---

## 2. Calibrator Loading Logic

**File:** `app/homography/calibrator.py`

**Method:** `LandmarkHomographyCalibrator.load_calibration()`

```python
def load_calibration(self, filename: str) -> bool:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    self.homography_matrix = np.array(data.get("homography_matrix", np.eye(3).tolist()))
    #                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                  THIS FIELD DOES NOT EXIST IN THE JSON
    #                                  Returns default: np.eye(3) <- IDENTITY
```

**Line 244 in calibrator.py:**
```python
self.homography_matrix = np.array(data.get("homography_matrix", np.eye(3).tolist()))
```

---

## 3. Homography Matrix Verification

### Expected vs Actual

**Expected:** A 3x3 perspective transformation matrix that maps:
- Image (50, 300) → World (0, 0)
- Image (1000, 300) → World (105, 0)
- Image (1050, 600) → World (105, 68)
- Image (0, 680) → World (0, 68)

**Actual:** Identity matrix
```python
[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0]]
```

**Verification:**
```python
# In run_match_analysis.py after loading:
H = self.homography_calibrator.get_matrix()
print(H)
# Output: [[1. 0. 0.]
#          [0. 1. 0.]
#          [0. 0. 1.]]
```

---

## 4. Coordinate Transformation Trace (Track ID 1, Frame 3)

### Input
- **BBox:** (720, 452, 753, 525)
- **Bottom center (image):** (736.5, 525.0)

### Transformation Step 1: Homography Application
**Location:** `app/homography/homography_utils.py::transform_point()`

```python
def transform_point(point, homography_matrix):
    point = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography_matrix)
    return tuple(transformed[0][0])
```

**Call in run_match_analysis.py (line ~875):**
```python
ball_field_pos = transform_point((b_cx, b_cy), self.pitch_mapper.homography_matrix)
```

**With identity matrix:**
```python
cv2.perspectiveTransform([[736.5, 525.0]], [[1,0,0],[0,1,0],[0,0,1]])
= [[736.5, 525.0]]
```

### Transformation Step 2: Meter Conversion
**Location:** `app/homography/pitch_mapper.py::process_frame()`

```python
# Field position is already in meters after homography
mp.field_position = transform_result  # (736.5, 525.0)
```

**Expected in meters:** (0-105, 0-68)
**Actual:** (736.5, 525.0) - still in pixels!

### Final Coordinate Used by SpeedEstimator
**Location:** `scripts/run_match_analysis.py` line ~850

```python
pos_m = mp.field_position  # (736.5, 525.0) - WRONG: should be meters
speed_data = self.speed_estimator.update(tid, pos_m)
```

**SpeedEstimator receives:** (736.5, 525.0) thinking it's meters
**Actual unit:** pixels

### Final Coordinate Used by DistanceTracker
```python
dist_m = self.distance_tracker.update(tid, pos_m, speed_kmh=speed_kmh)
```

**DistanceTracker receives:** (736.5, 525.0) thinking it's meters
**Actual unit:** pixels

---

## 5. Debug Output Analysis

**Observed:**
```
[DEBUG Homography] Frame: 3 | Track ID: 1 | Img (cx, bottom_y): ( 736.5,  525.0) | Field (x_m, y_m): ( 736.5,  525.0)
```

**Expected:**
```
[DEBUG Homography] Frame: 3 | Track ID: 1 | Img (cx, bottom_y): ( 736.5,  525.0) | Field (x_m, y_m): ( 23.4,  12.1)
```

**Discrepancy:** Image coordinates == Field coordinates (identical)

---

## 6. Root Cause

### Exact Line Responsible

**File:** `app/homography/calibrator.py`, **Line 244**

```python
self.homography_matrix = np.array(data.get("homography_matrix", np.eye(3).tolist()))
```

**Why this causes the issue:**
1. `data.get("homography_matrix")` returns `None` because the field doesn't exist in the JSON
2. Default value `np.eye(3).tolist()` is used
3. Result: identity matrix loaded
4. Identity matrix transforms (x, y) → (x, y) - no change

### Secondary Issue

**File:** `configs/homography_calibration.json`

The JSON file contains `calibration_points` but not `homography_matrix`. The calibrator expects:
- A pre-computed `homography_matrix` field, OR
- A method to compute the matrix from `calibration_points`

Neither exists.

---

## 7. Expected Pitch Dimensions

**Standard FIFA Pitch:**
- Length: 105 meters
- Width: 68 meters
- Aspect ratio: 105/68 ≈ 1.544

**From calibration file:**
```json
"field_dimensions": {
  "length_m": 105.0,
  "width_m": 68.0
}
```

**Verification:** ✓ Dimensions are correct

---

## 8. Impact Assessment

### Current Behavior
- All player coordinates remain in pixel space (0-1280, 0-720)
- Speed calculations use pixel distances → INVALID
- Distance calculations use pixel distances → INVALID
- Heatmaps use pixel positions → INVALID
- All spatial analytics are WRONG

### Expected Behavior
- Player coordinates should be in meters (0-105, 0-68)
- Speed should be in km/h based on meter distances
- Distance should be in meters
- Heatmaps should show pitch positions

### Evidence of Bug
1. Debug output shows identical image/field coordinates
2. Speed validation shows max 180 km/h (impossible - caused by pixel distances)
3. Formation detection gets 0 detections (players in wrong coordinate space)

---

## 9. Verification Steps

### Step 1: Check Matrix Identity
```python
H = calibrator.get_matrix()
print(np.allclose(H, np.eye(3)))  # Returns: True (identity)
```

### Step 2: Check Pipeline Usage
```python
# In run_match_analysis.py:
print(self.homography_calibrator.homography_matrix)
# Output: [[1. 0. 0.]
#          [0. 1. 0.]
#          [0. 0. 1.]]
```

### Step 3: Verify Transformation
```python
import cv2
import numpy as np

H = np.eye(3)
point = np.array([[736.5, 525.0]], dtype=np.float32)
result = cv2.perspectiveTransform(point, H)
print(result[0][0])  # (736.5, 525.0) - unchanged
```

---

## 10. Required Fix (Not Implemented)

**Option A:** Add `homography_matrix` to calibration JSON
```json
{
  "homography_matrix": [[...], [...], [...]]
}
```

**Option B:** Compute matrix from calibration points in calibrator
```python
def compute_homography_from_points(self, src_pts, dst_pts):
    self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
```

**Current Status:** NEITHER OPTION IS IMPLEMENTED

---

## 11. Conclusion

**The homography is NOT applied.**

The pipeline loads an identity matrix because:
1. The calibration JSON lacks `homography_matrix` field
2. The calibrator's `load_calibration()` uses a default identity matrix
3. No code exists to compute the matrix from `calibration_points`

**Result:** All spatial analytics (speed, distance, heatmaps) are calculated in pixel coordinates and reported as meters, producing invalid results.

**Severity:** CRITICAL
**Impact:** All downstream analytics invalid
**Effort to fix:** Add matrix computation or pre-computed matrix to calibration file