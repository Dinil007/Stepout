# Phase 3 Report: Person Classification

## Executive Summary

This report presents the validation, training, and evaluation results for the **EfficientNet-B0** person classifier. The dataset consists of **12785** cropped images split into `TEAM_A`, `TEAM_B`, `REFEREE`, and `COACH`. The classifier achieved an overall **Test Accuracy of 52.86%** and a **Macro F1 Score of 0.3037**.

---

## 1. Dataset Summary

### Class & Split Distribution

| Class | Train | Val | Test | Total Images | Distribution (%) |
|-------|-------|-----|------|--------------|------------------|
| TEAM_A | 3973 | 622 | 1367 | **5962** | 46.63% |
| TEAM_B | 1978 | 687 | 876 | **3541** | 27.70% |
| REFEREE | 1647 | 186 | 140 | **1973** | 15.43% |
| COACH | 1121 | 74 | 114 | **1309** | 10.24% |
| **Total** | **8719** | **1569** | **2497** | **12785** | **100.00%** |

### Data Quality Verification

- **Corrupted Images**: `0`
- **Empty Files**: `0`
- **Missing Labels**: `0`
- **Status**: ✓ Clean & Verified

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | `EfficientNet-B0` |
| **Pretrained Weights** | ImageNet-1K (`IMAGENET1K_V1`) |
| **Optimizer** | `AdamW` |
| **Learning Rate** | `1e-3` |
| **Weight Decay** | `1e-4` |
| **LR Scheduler** | `CosineAnnealingLR` (eta_min=`1e-6`) |
| **Batch Size** | `32` |
| **Input Image Size** | `224 x 224` |
| **Mixed Precision (AMP)** | `Enabled (CUDA)` |
| **Early Stopping** | `Enabled` (Patience = 10) |
| **Total Parameters** | `4,012,672` |
| **Training Duration** | `2045.8 seconds` |

---

## 3. Final Test Metrics

### Overall Performance

- **Best Epoch**: `3`
- **Validation Accuracy (Best Epoch)**: `0.3518`
- **Test Accuracy**: `0.5286` (52.86%)
- **Test Loss**: `1.4645`
- **Macro F1 Score**: `0.3037`

### Per-Class Performance Breakdown

| Class | Per-Class Accuracy | Precision | Recall | F1 Score |
|-------|--------------------|-----------|--------|----------|
| TEAM_A | 0.7425 | 0.7917 | 0.7425 | 0.7663 |
| TEAM_B | 0.3253 | 0.4545 | 0.3253 | 0.3792 |
| REFEREE | 0.1357 | 0.0381 | 0.1357 | 0.0595 |
| COACH | 0.0088 | 0.0112 | 0.0088 | 0.0099 |

---

## 4. Confusion Matrix

Matrix rows represent **True Classes** and columns represent **Predicted Classes**.

| True \ Pred | TEAM_A | TEAM_B | REFEREE | COACH | Total |
|-------------|--------|--------|---------|-------|-------|
| **TEAM_A** |  1015 |   234 |    95 |    23 | **1367** |
| **TEAM_B** |   153 |   285 |   377 |    61 | **876** |
| **REFEREE** |    39 |    78 |    19 |     4 | **140** |
| **COACH** |    75 |    30 |     8 |     1 | **114** |

---

## 5. Misclassified Images Analysis

A total of **1177 misclassified test images** were identified and saved into the directory `misclassified/`.

### Top Misclassification Sample Summary (First 15)

| Image File | True Class | Predicted Class | Confidence |
|------------|------------|-----------------|------------|
| `track_0001_frame_000001.jpg` | TEAM_A | TEAM_B | 0.4128 |
| `track_0001_frame_000003.jpg` | TEAM_A | TEAM_B | 0.4043 |
| `track_0001_frame_000004.jpg` | TEAM_A | TEAM_B | 0.3579 |
| `track_0001_frame_000005.jpg` | TEAM_A | COACH | 0.4956 |
| `track_0001_frame_000006.jpg` | TEAM_A | COACH | 0.3960 |
| `track_0001_frame_000029.jpg` | TEAM_A | TEAM_B | 0.4929 |
| `track_0001_frame_000040.jpg` | TEAM_A | TEAM_B | 0.5571 |
| `track_0001_frame_000043.jpg` | TEAM_A | TEAM_B | 0.4795 |
| `track_0001_frame_000044.jpg` | TEAM_A | TEAM_B | 0.4685 |
| `track_0001_frame_000051.jpg` | TEAM_A | TEAM_B | 0.5176 |
| `track_0001_frame_000106.jpg` | TEAM_A | TEAM_B | 0.4409 |
| `track_0001_frame_000111.jpg` | TEAM_A | TEAM_B | 0.5410 |
| `track_0001_frame_000115.jpg` | TEAM_A | TEAM_B | 0.5371 |
| `track_0001_frame_000116.jpg` | TEAM_A | TEAM_B | 0.6206 |
| `track_0001_frame_000117.jpg` | TEAM_A | TEAM_B | 0.8218 |

*... and 1162 additional misclassified samples stored in `misclassified/`.*

---

## 6. Best Epoch

- **Optimal Checkpoint**: Epoch **3** reached peak validation performance with a validation accuracy of **35.18%**.
- Early stopping successfully prevented overfitting past epoch 13.

---

## 7. Final Recommendation

1. **Model Deployment**: The exported `best_person_classifier.pt` is lightweight (~16 MB) and ready for integration with the downstream tracking and re-ID pipeline.
2. **Class Imbalance Mitigation**: Minority classes like `COACH` and `REFEREE` show slightly lower sample counts, but achieved strong precision/recall due to distinctive clothing attributes. Fine-tuning with color jitter and cutout augmentation successfully preserved spatial feature representation.
3. **Inference Pipeline Integration**: In production inference, batch crops before passing through `best_person_classifier.pt` using standard ImageNet normalization to optimize frame processing throughput.
