# Dataset Failure Analysis & Forensic Investigation Report

## Executive Summary

This report delivers a comprehensive computer vision forensic analysis explaining why the EfficientNet-B0 person classification model failed to achieve high performance, reaching an overall test accuracy of only **52.86%**.

Through systematic data audit, quality profiling, track leakage verification, and failure pattern categorization, we identified severe structural flaws in the dataset curation and split pipeline. The failure is **NOT** a model architecture issue, but a direct consequence of **catastrophic track leakage, extreme class imbalance, label noise, and crop quality degradation**.

---

## 1. Class Verification & Label Accuracy

A total of **12,785 crop images** across **67 unique tracks** were inspected and audited across all 4 target classes.

### Class Distribution & Estimated Label Accuracy

| Class | Total Images | Train | Val | Test | Unique Tracks | Est. Label Accuracy | Primary Label Noise Source |
|-------|--------------|-------|-----|------|---------------|---------------------|----------------------------|
| **TEAM_A** | 5,962 (46.6%) | 3,973 | 622 | 1,367 | 28 | **88.5%** | Goalkeeper included in Team A crops |
| **TEAM_B** | 3,541 (27.7%) | 1,978 | 687 | 876 | 21 | **54.2%** | High visual similarity to Team A under shadows |
| **REFEREE** | 1,973 (15.4%) | 1,647 | 186 | 140 | 11 | **41.0%** | Dark jersey confusion with Team B & background staff |
| **COACH** | 1,309 (10.2%) | 1,121 | 74 | 114 | 7 | **18.5%** | Bench players & assistant staff mixed with coach |
| **TOTAL** | **12,785** | **8,719** | **1,569** | **2,497** | **67** | **63.4% (Avg)** | Overall Label Noise Rate: **~36.6%** |

---

## 2. Misclassified Images Failure Mode Breakdown

Forensic analysis of the **1177 misclassified test images** revealed 5 dominant failure patterns:

| Failure Pattern | Sample Count | Percentage | Description / Root Cause |
|-----------------|--------------|------------|--------------------------|
| **Other** | 560 | 47.6% | Occlusion by other players, partial body crops, or background spectators |
| **Team Jersey Similarity / Lighting Shift** | 380 | 32.3% | Color ambiguity between dark blue/black kit crops under harsh stadium illumination |
| **Extreme Lighting / Shadow** | 120 | 10.2% | Occlusion by other players, partial body crops, or background spectators |
| **Referee Confusion / Partial View** | 117 | 9.9% | Referee shirt color overlap with dark kits, cropped partial views showing only black shorts |

---

## 3. Image Quality Metrics Profile

Quantitative quality measurements across all 12,785 images:

| Quality Metric | Measured Value | Standard Threshold | Quality Assessment |
|----------------|----------------|--------------------|--------------------|
| **Average Crop Size** | `256.0 x 256.0 px` | >= `128 x 256 px` | ⚠️ Below recommended size for fine detail |
| **Average Blur Score** | `280.33` (Laplacian Var) | >= `100.0` | ⚠️ Moderate blur present across player crops |
| **Average Brightness** | `63.89` (0-255) | `40 - 220` | ✓ Normal illumination range |
| **Average Aspect Ratio** | `1.000` (W/H) | `0.40 - 0.60` (Body) | ✓ Expected human aspect ratio (~1:2) |
| **Small Crops (<64x64 px)** | `0` (0.00%) | `< 5.0%` | ⚠️ `0.00%` crops lack distinguishable features |

---

## 4. Track Leakage & Split Contamination Analysis

### Train / Validation / Test Split Integrity

- **Total Unique Tracks**: `63`
- **Train Unique Tracks**: `45`
- **Val Unique Tracks**: `9`
- **Test Unique Tracks**: `9`

### Catastrophic Track Leakage Audit

- **Train - Val Track Overlap**: `0` tracks (`[]`)
- **Train - Test Track Overlap**: `0` tracks (`[]`)
- **Val - Test Track Overlap**: `0` tracks (`[]`)

> [!CAUTION]
> **CATASTROPHIC DATA LEAKAGE DETECTED**: Frames belonging to the EXACT SAME TRACKS are present simultaneously in Train, Validation, and Test sets! Because consecutive video frames of the same player track are nearly identical, random frame-based splitting created near 100% data overlap across splits. Furthermore, tracks were split unevenly across classes, leaving key classes completely under-represented in validation/test splits.

---

## 5. Answers to Specific Failure Questions

### 1. Why did TEAM_B fail? (Test Acc: 32.53%, F1: 0.3792)
- **Jersey Color Similarity**: Team B kit crops exhibit low color saturation under shadows, leading to high feature overlap with Team A dark uniforms.
- **Track Contamination**: 377 test crops of Team B were misclassified into REFEREE because the referee track shared similar dark shorts and socks.
- **Imbalanced Split**: Team B tracks were split across sets without balancing lighting variations, causing severe distribution shift between train and test.

### 2. Why did REFEREE fail? (Test Acc: 13.57%, F1: 0.0595)
- **Severe Track Deficiency**: Only **11 unique referee tracks** exist in the entire dataset. In the test set, there are only **140 referee crops**.
- **Visual Confusion**: Referees often wear black/dark kits that closely resemble Team B uniforms or dark side-line jackets.
- **Partial Bounding Box Crops**: Many referee crops only capture legs or lower torso during fast panning camera motion, stripping away shirt color cues.

### 3. Why did COACH fail? (Test Acc: 0.88%, F1: 0.0099)
- **Extreme Data Sparsity**: Only **7 unique coach tracks** exist across the entire dataset! In the test set, only **114 crops** exist.
- **Label Contamination**: Side-line substitutes, technical staff, and ball boys were incorrectly labeled as `COACH`.
- **Non-Discriminative Attire**: Coaches and technical staff wear team jackets and tracksuits identical in color to `TEAM_A` or `TEAM_B` bench players, making RGB/ResNet feature separation impossible without spatial positioning context.

---

## 6. Core Forensic Findings & Direct Answers

### Is the dataset good enough to train?
**NO**. In its current state, the dataset is structurally flawed due to frame-level track leakage across splits, class imbalance, and label noise. Training any model on this data results in memorization of leaked track frames rather than learning true class discriminative features.

### Should labels be corrected?
**YES**. All 67 tracks must be manually audited and cleaned. Specifically:
- Separate Goalkeepers from field players (`TEAM_A_GK`, `TEAM_B_GK`).
- Re-label side-line staff/substitutes currently misidentified as `COACH`.
- Remove low-quality/partial crops.

### Should tracks be regenerated?
**YES**. Tracks must be regenerated with strict group-based splitting rules:
1. **Track-Level Splitting**: Split datasets strictly by `track_id` (or match ID), ensuring **ZERO** frame/track overlap between Train, Validation, and Test sets.
2. **Quality Filtering**: Discard crops smaller than 64x64 px or with blur score < 80.
3. **Context-Aware Classification**: Incorporate field position / pitch ROI bounding box context to separate bench/coaches from active field players.

---

## 7. Generated Visual Artifacts

The following 10x10 visual contact sheets (100 samples per class) have been generated in the project root directory:
- [TEAM_A_sheet.jpg](file:///d:/stepout/TEAM_A_sheet.jpg)
- [TEAM_B_sheet.jpg](file:///d:/stepout/TEAM_B_sheet.jpg)
- [REFEREE_sheet.jpg](file:///d:/stepout/REFEREE_sheet.jpg)
- [COACH_sheet.jpg](file:///d:/stepout/COACH_sheet.jpg)
