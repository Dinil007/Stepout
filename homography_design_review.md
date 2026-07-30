# Homography Design Review

## Executive Summary

**Status:** DESIGN MISMATCH BETWEEN CALIBRATOR AND CALIBRATION FILE FORMAT

The `LandmarkHomographyCalibrator` expects a pre-computed `homography_matrix` in the JSON file, but the actual calibration file stores `calibration_points` (source/destination pairs) that need to be computed into a matrix.

---

## 1. Why Does homography_calibration.json Store calibration_points?

**Answer:** The JSON format is designed for **dynamic computation** of the homography matrix from source/destination point pairs, rather than storing a pre-computed matrix.

**Evidence:**

```json
{
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
}
```

The file explicitly stores:
- `source`: Image coordinates where pitch corners were clicked
- `destination`: Real-world pitch coordinates (meters) where those corners map to

This is a **human-editable** format that allows manual correction of calibration points without needing to recompute the matrix externally.

---

## 2. Intended Design: A or B?

**Answer: B) Compute the homography matrix dynamically from calibration_points every time the calibration is loaded.**

**Evidence from codebase:**

### File: `app/homography/homography_utils.py` (lines 69-108)

```python
def compute_homography(
    source_points: Union[List[Tuple[float, float]], np.ndarray],
    destination_points: Union[List[Tuple[float, float]], np.ndarray],
    method: int = cv2.RANSAC,
    ransac_reproj_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the 3x3 Homography Matrix mapping source points to target destination points."""
    src_pts, dst_pts = validate_points(source_points, destination_points)
    
    src_pts_reshaped = src_pts.reshape(-1, 1, 2)
    dst_pts_reshaped = dst_pts.reshape(-1, 1, 2)
    
    homography_matrix, mask = cv2.findHomography(
        src_pts_reshaped,
        dst_pts_reshaped,
        method=method,
        ransacReprojThreshold=ransac_reproj_threshold
    )
    
    if homography_matrix is None or homography_matrix.shape != (3, 3):
        logger.error("cv2.findHomography returned None or invalid matrix shape.")
        raise RuntimeError("Failed to compute valid 3x3 Homography Matrix.")
    
    logger.info("Homography matrix computed successfully.")
    return homography_matrix, mask
```

### File: `app/homography/pitch_mapper.py` (lines 64-82)

```python
def load_homography(
    self,
    source_points: Union[List[Tuple[float, float]], np.ndarray],
    destination_points: Union[List[Tuple[float, float]], np.ndarray]
) -> np.ndarray:
    """Computes and loads the 3x3 Homography matrix from source and destination point sets."""
    matrix, _ = compute_homography(source_points, destination_points)
    self.homography_matrix = matrix
    logger.info("PitchMapper loaded new Homography matrix.")
    return self.homography_matrix
```

### File: `scripts/run_pipeline.py` (usage example)

```python
from app.homography.homography_utils import compute_homography, transform_points

# Homography initialization
H_matrix, _ = compute_homography(self.src_homography_pts, self.dst_homography_pts)
pitch_mapper = PitchMapper(homography_matrix=H_matrix)
```

### File: `scripts/validate_soccernet_match.py` (usage example)

```python
from app.homography.homography_utils import compute_homography, transform_point
H, mask = compute_homography(PITCH_SRC_POINTS, PITCH_DST_POINTS)
result["homography_matrix_shape"] = str(H.shape)
```

**Pattern:** Every other script in the repository calls `compute_homography()` with source and destination points to dynamically compute the matrix.

---

## 3. Repository Search Results

**Files containing `cv2.findHomography`:**
1. `app/homography/calibrator.py` - Uses it in `calibrate()` method for landmark-based calibration
2. `app/homography/homography_utils.py` - Wraps it in `compute_homography()` utility

**Files containing `compute_homography` calls:**
1. `app/homography/pitch_mapper.py` - `load_homography()` method calls it
2. `scripts/run_pipeline.py` - Computes H from source/destination points
3. `scripts/full_platform_integration.py` - Computes H from source/destination points
4. `scripts/tracking_diagnostics.py` - Computes H from source/destination points
5. `scripts/validate_soccernet_match.py` - Computes H from source/destination points
6. `scripts/validate_tracking.py` - Computes H from source/destination points
7. `inspect_tracking_frames.py` - Computes H from source/destination points

**Files containing `cv2.getPerspectiveTransform`:** None found in search

**Files containing `calculate_homography`:** None found in search

---

## 4. Existing Code to Compute the Matrix

**Yes, the code exists and is actively used throughout the project.**

### Function: `compute_homography()` in `app/homography/homography_utils.py`

**Location:** Lines 69-108

**Function signature:**
```python
def compute_homography(
    source_points: Union[List[Tuple[float, float]], np.ndarray],
    destination_points: Union[List[Tuple[float, float]], np.ndarray],
    method: int = cv2.RANSAC,
    ransac_reproj_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]
```

**What it does:**
1. Validates source and destination points (minimum 4 pairs)
2. Reshapes for OpenCV format (N x 1 x 2)
3. Calls `cv2.findHomography()` with RANSAC
4. Validates the resulting 3x3 matrix
5. Returns matrix and inlier mask

**Usage pattern across the codebase:**
```python
from app.homography.homography_utils import compute_homography

# Define source and destination points
src_points = np.array([[x1,y1], [x2,y2], ...], dtype=np.float32)
dst_points = np.array([[0,0], [105,0], [105,68], [0,68]], dtype=np.float32)

# Compute homography matrix
H, mask = compute_homography(src_points, dst_points)

# Use the matrix
pitch_mapper = PitchMapper(homography_matrix=H)
```

---

## 5. Why Is This Code NOT Being Called?

**Reason:** `LandmarkHomographyCalibrator.load_calibration()` does not call `compute_homography()`.

**Exact code path:**

### File: `app/homography/calibrator.py` (Line 244)

```python
def load_calibration(self, filename: str) -> bool:
    """Load calibration from JSON file."""
    if not os.path.exists(filename):
        return False
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.homography_matrix = np.array(data.get("homography_matrix", np.eye(3).tolist()))
        #                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                  EXPECTS PRE-COMPUTED MATRIX
        #                                  BUT JSON CONTAINS calibration_points
        
        self.calibration_method = data.get("validation_message", "loaded")
        return True
    except Exception as e:
        print(f"Failed to load calibration: {e}")
        return False
```

**Problem:**
1. The method tries to load `data.get("homography_matrix")` which doesn't exist in the JSON
2. It defaults to `np.eye(3)` (identity matrix)
3. It NEVER calls `compute_homography()` to compute the matrix from `calibration_points`
4. Result: Identity matrix is loaded and used throughout the pipeline

**What SHOULD happen:**

```python
def load_calibration(self, filename: str) -> bool:
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Check if pre-computed matrix exists
    if "homography_matrix" in data:
        self.homography_matrix = np.array(data["homography_matrix"])
        return True
    
    # Otherwise compute from calibration_points
    elif "calibration_points" in data:
        src_pts = np.array(data["calibration_points"]["source"], dtype=np.float32)
        dst_pts = np.array(data["calibration_points"]["destination"], dtype=np.float32)
        self.homography_matrix, _ = compute_homography(src_pts, dst_pts)
        return True
    
    else:
        return False
```

---

## 6. Summary of Intended Architecture

### Design Pattern A (Pre-computed Matrix) - NOT USED
```json
{
  "homography_matrix": [[...], [...], [...]]
}
```
Calibrator loads matrix directly. No computation on load.

### Design Pattern B (Dynamic Computation) - INTENDED
```json
{
  "calibration_points": {
    "source": [[x1,y1], ...],
    "destination": [[x2,y2], ...]
  }
}
```
Calibrator calls `compute_homography(src_pts, dst_pts)` to compute matrix on load.

### Evidence for Pattern B

1. **Calibration file format:** Uses `calibration_points`, not `homography_matrix`
2. **All other scripts:** Use `compute_homography(source, destination)` pattern
3. **Utility function exists:** `compute_homography()` in `homography_utils.py`
4. **PitchMapper has method:** `load_homography(source_points, destination_points)` that computes matrix
5. **Documentation:** File headers describe "computing" homography from point pairs

### Why Pattern A Was Implemented

**Likely cause:** The new `LandmarkHomographyCalibrator` class was written without reference to the existing calibration file format or the established `compute_homography()` utility pattern.

The implementer assumed:
- JSON would contain a pre-computed `homography_matrix`
- No need to call `compute_homography()` during load
- Simple array load would suffice

This broke compatibility with the existing calibration workflow.

---

## 7. Correct Implementation Path

**Option 1: Modify `LandmarkHomographyCalibrator.load_calibration()`**
```python
if "homography_matrix" in data:
    self.homography_matrix = np.array(data["homography_matrix"])
elif "calibration_points" in data:
    src = np.array(data["calibration_points"]["source"])
    dst = np.array(data["calibration_points"]["destination"])
    self.homography_matrix, _ = compute_homography(src, dst)
```

**Option 2: Modify calibration JSON to include both**
```json
{
  "calibration_points": {...},
  "homography_matrix": [[...], [...], [...]]
}
```

**Option 3: Use `PitchMapper.load_homography()`**
```python
from app.homography.pitch_mapper import PitchMapper
pm = PitchMapper()
pm.load_homography(src_pts, dst_pts)
self.homography_matrix = pm.homography_matrix
```

**Current Status:** NONE OF THESE OPTIONS ARE IMPLEMENTED

---

## 8. Conclusion

**Intended Design:** Pattern B - Dynamic computation from `calibration_points`

**Actual Implementation:** Pattern A attempted, but broken because JSON doesn't contain `homography_matrix`

**Root Cause:** `LandmarkHomographyCalibrator.load_calibration()` does not call `compute_homography()` to compute the matrix from the `calibration_points` stored in the JSON file.

**Fix Required:** Update `load_calibration()` to detect `calibration_points` and call `compute_homography()` to generate the matrix on load.

**Severity:** CRITICAL - Prevents all spatial analytics from functioning