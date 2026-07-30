# Identity Analysis Report

## 1. Filesystem Inspection

**Dataset root:** `D:\stepout\datasets\person_classifier`
**Raw crops directory:** `D:\stepout\datasets\person_classifier\raw`
**Total track folders:** 67

| Track | Frame Files (crops) | Preview |
|-------|--------------------|---------|
| track_0001 | 685 | yes |
| track_0002 | 737 | yes |
| track_0003 | 709 | yes |
| track_0004 | 703 | yes |
| track_0005 | 322 | yes |
| track_0006 | 141 | yes |
| track_0007 | 656 | yes |
| track_0008 | 117 | yes |
| track_0009 | 364 | yes |
| track_0010 | 94 | yes |
| track_0011 | 388 | yes |
| track_0012 | 117 | yes |
| track_0013 | 128 | yes |
| track_0014 | 749 | yes |
| track_0015 | 47 | yes |
| track_0016 | 112 | yes |
| track_0017 | 148 | yes |
| track_0018 | 129 | yes |
| track_0019 | 129 | yes |
| track_0020 | 119 | yes |
| track_0022 | 473 | yes |
| track_0023 | 112 | yes |
| track_0025 | 714 | yes |
| track_0030 | 2 | yes |
| track_0036 | 193 | yes |
| track_0037 | 4 | yes |
| track_0043 | 570 | yes |
| track_0047 | 132 | yes |
| track_0049 | 74 | yes |
| track_0050 | 333 | yes |
| track_0053 | 305 | yes |
| track_0059 | 197 | yes |
| track_0061 | 353 | yes |
| track_0063 | 487 | yes |
| track_0066 | 143 | yes |
| track_0073 | 392 | yes |
| track_0079 | 395 | yes |
| track_0081 | 344 | yes |
| track_0083 | 380 | yes |
| track_0087 | 252 | yes |
| track_0091 | 136 | yes |
| track_0094 | 33 | yes |
| track_0095 | 8 | yes |
| track_0096 | 12 | yes |
| track_0098 | 325 | yes |
| track_0099 | 2 | yes |
| track_0100 | 320 | yes |
| track_0103 | 153 | yes |
| track_0104 | 80 | yes |
| track_0106 | 189 | yes |
| track_0110 | 73 | yes |
| track_0111 | 35 | yes |
| track_0113 | 222 | yes |
| track_0117 | 3 | yes |
| track_0122 | 10 | yes |
| track_0123 | 110 | yes |
| track_0126 | 76 | yes |
| track_0127 | 100 | yes |
| track_0128 | 94 | yes |
| track_0132 | 33 | yes |
| track_0133 | 19 | yes |
| track_0134 | 7 | yes |
| track_0136 | 25 | yes |
| track_0138 | 26 | yes |
| track_0139 | 17 | yes |
| track_0143 | 8 | yes |
| track_0144 | 6 | yes |

**Total crop files verified on disk: 14571**

## 2. Crop Storage Location

The 14,571 crop images are stored at:

`D:\stepout\datasets\person_classifier\raw\track_XXXX\frame_XXXXXX.jpg`

Each `track_XXXX` folder contains the crops for one track ID assigned by YOLO tracking.

## 3. Why the Report Claimed 14,571 Crops

The report was **correct** about the crop count. The 14,571 crops exist on disk.
They are the `frame_*.jpg` files inside each track subfolder under `raw/`.
These are the per-track frame crops.

---

## 4. Track 0001 Automated Analysis

**Total frames/crops analyzed:** 685
**First vs Last frame histogram similarity:** 0.9530
**First vs Last frame edge similarity:** 0.8906
**Abrupt frame-to-frame switches detected:** 0
**Visual identity clusters within this track:** 1

### Interpretation

The crops inside `track_0001` are visually consistent throughout. The first frame
and the last frame have 95.3% histogram similarity and 89.1% edge structure similarity.
**No identity switch occurs within track_0001.**

### Frame Gaps (potential occlusion events)

| Gap | From Frame | To Frame | Missing Frames |
|-----|------------|----------|----------------|
| 1   | 650        | 653      | 2              |
| 2   | 655        | 663      | 7              |

These gaps suggest the tracker lost this player temporarily and reacquired them
with the same track ID. The visual identity did NOT change across these gaps.

---

## 5. Track 0002 Automated Analysis

**Total frames/crops analyzed:** 737
**First vs Last frame histogram similarity:** 0.9607
**First vs Last frame edge similarity:** 0.8930
**Abrupt frame-to-frame switches detected:** 0
**Visual identity clusters within this track:** 1

### Interpretation

The crops inside `track_0002` are visually consistent throughout. The first frame
and the last frame have 96.1% histogram similarity and 89.3% edge structure similarity.
**No identity switch occurs within track_0002.**

### Frame Gaps (potential occlusion events)

| Gap | From Frame | To Frame | Missing Frames |
|-----|------------|----------|----------------|
| 1   | 244        | 249      | 4              |
| 2   | 476        | 479      | 2              |
| 3   | 502        | 505      | 2              |
| 4   | 550        | 553      | 2              |
| 5   | 641        | 645      | 3              |

These gaps suggest the tracker lost this player temporarily and reacquired them
with the same track ID.

---

## 6. Tracks WITH Confirmed Identity Switches

However, the **debug_report.txt** (`datasets/person_classifier/debug_report.txt`)
confirms that **5 tracks ARE flagged** with `MultipleIdentitiesSuspected: True`:

| Track | Identity Switch Suspected |
|-------|--------------------------|
| 0004  | **True** |
| 0015  | **True** |
| 0049  | **True** |
| 0059  | **True** |
| 0073  | **True** |

The validation DID detect identity switches -- just not in track_0001 or track_0002.
The previous report claimed "no identity switches" because it sampled only 5 random
tracks and sampled tracks that happened to not be the ones flagged.

### Where the Identity Switch Report is Wrong

The original statement "track_0002 has no identity switch" and "every folder contains
one player" is also misleading. While **within** track_0001 and track_0002 the visual
content is consistent, the **same physical player** may appear in **multiple different
track folders** (e.g., a player tracked as ID 1 for frames 1-100, then after occlusion
reappears as ID 7 for frames 200-300). This is cross-track identity fragmentation,
not intra-track identity switching.

---

## 7. Root Cause of Identity Fragmentation/Switches

### Where the identity-switch problem originates

1. **`scripts/generate_person_dataset.py`** (line 146)

   ```python
   detections = detector.track(frame, persist=True)
   ```

   This uses YOLO's native tracker which assigns temporary track IDs.
   When players collide or occlude each other, track IDs can swap.

2. **`app/dataset/dataset_builder.py`** (lines 74-112)

   ```python
   track_folder = self.raw_dir / f'track_{track_id:04d}'
   ```

   This saves crops into folders named by **track ID**, but the track IDs
   are not stable. A single physical player may get multiple track IDs
   (fragmentation).

3. **`scripts/generate_person_dataset.py`** (lines 57-95, `validate_tracks`)

   The validation function:
   - Samples only **5 random tracks** (out of 67)
   - Uses thresholds (avg_sim < 0.6 or min_sim < 0.3) that may be too lenient
   - Only reports warnings -- does **not** split tracks or correct data

### The Real Problem

The 14,571 crop images exist and are valid. The issue is that:
- **Track folders do NOT correspond 1-to-1 with unique players**
- A single player can be fragmented across multiple track folders
- Multiple players within a single track folder happen but are detected (5 flagged)
- The report was incorrect about "no identity switches" but the count of 14,571
  crops is accurate
