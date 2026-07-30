# GPU Deployment Guide

**Date:** 2026-07-27
**Purpose:** Deploy the football analytics platform on a CUDA-enabled machine without modifying source code
**Code Changes Required:** None - the project is already GPU-ready

---

## Table of Contents

1. [Deployment Requirements](#1-deployment-requirements)
2. [Requirements File](#2-requirements-file)
3. [Environment Setup](#3-environment-setup)
4. [GPU Verification](#4-gpu-verification)
5. [Benchmark Plan](#5-benchmark-plan)
6. [Output Validation](#6-output-validation)

---

## 1. Deployment Requirements

### 1.1 Minimum GPU Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Model | NVIDIA GTX 1660 (6GB VRAM) | NVIDIA RTX 3060 (12GB VRAM) or better |
| Compute Capability | 6.0+ | 7.0+ |
| CUDA Cores | 1200+ | 3500+ |
| VRAM | 6GB | 12GB+ |
| Power Supply | 450W | 550W+ |
| PCIe Interface | PCIe 3.0 | PCIe 4.0 |

**Compatible GPUs:**
- NVIDIA GTX 1660, 1660 Ti, 1660 Super
- NVIDIA RTX 20-series (2060, 2070, 2080)
- NVIDIA RTX 30-series (3060, 3070, 3080, 3090)
- NVIDIA RTX 40-series (4060, 4070, 4080, 4090)
- NVIDIA Quadro RTX series
- NVIDIA A100 (datacenter)

### 1.2 NVIDIA Driver

- **Minimum Version:** 450.80.02
- **Recommended Version:** Latest stable from NVIDIA
- **Download:** https://www.nvidia.com/Download/index.aspx

**Verify driver installation:**
```bash
nvidia-smi
# Expected output: GPU name, driver version, CUDA version
```

### 1.3 CUDA Toolkit

- **Minimum Version:** CUDA 11.7
- **Recommended Version:** CUDA 12.1
- **Download:** https://developer.nvidia.com/cuda-toolkit

**Verify CUDA installation:**
```bash
nvcc --version
# Expected: Cuda compilation tools, release 11.7 or 12.1
```

### 1.4 cuDNN

- **Minimum Version:** cuDNN 8.5
- **Recommended Version:** cuDNN 8.9
- **Download:** https://developer.nvidia.com/cudnn (requires NVIDIA developer account)

**Installation:**
1. Download cuDNN for your CUDA version
2. Extract to CUDA installation directory
3. Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin` (Windows) or `/usr/local/cuda/bin` (Linux)

### 1.5 Python

- **Minimum Version:** Python 3.8
- **Recommended Version:** Python 3.10 or 3.11
- **Download:** https://www.python.org/downloads/

**Note:** Python 3.12 is not yet fully supported by all dependencies.

### 1.6 PyTorch CUDA Build

- **Minimum Version:** PyTorch 1.13 + CUDA 11.7
- **Recommended Version:** PyTorch 2.1.0 + CUDA 12.1
- **Installation:**
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch CUDA:**
```python
import torch
print(torch.__version__)  # Expected: 2.1.0+cu121 or similar
print(torch.cuda.is_available())  # Expected: True
print(torch.version.cuda)  # Expected: 12.1 or 11.8
```

### 1.7 Required pip Packages

**Core dependencies (with GPU support):**
```
torch>=2.0.0+cu121
torchvision>=0.15.0+cu121
torchaudio>=2.0.0+cu121
ultralytics>=8.0.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=9.5.0
```

**Additional dependencies:**
```
# Tracking
cython>=0.29.0
filterpy>=1.4.5

# Pose estimation
mediapipe>=0.10.0

# Analytics
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0

# Logging
loguru>=0.7.0

# Utilities
tqdm>=4.65.0
```

### 1.8 Ultralytics YOLO

- **Minimum Version:** YOLOv8 8.0.0
- **Recommended Version:** YOLOv8 8.1.0 or later
- **Installation:**
```bash
pip install ultralytics>=8.1.0
```

**Verify YOLO installation:**
```python
from ultralytics import YOLO
print(YOLO.__version__)  # Expected: 8.1.0 or later
```

### 1.9 OpenCV

- **Minimum Version:** OpenCV 4.8.0
- **Recommended Version:** OpenCV 4.8.1
- **Installation:**
```bash
pip install opencv-python>=4.8.0
```

**Note:** OpenCV CUDA modules are optional and not required for this project. The project uses OpenCV for image processing (CPU) only.

---

## 2. Requirements File

### Current requirements.txt

The project already has a `requirements.txt` file. Verify it includes GPU-compatible versions:

```bash
# Verify PyTorch CUDA support
pip show torch | grep "Version"
# Expected: Contains "+cu121" or "+cu118"

# Verify all packages
pip check
# Expected: No conflicts
```

### Generating Updated requirements.txt

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install GPU-enabled PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Generate locked requirements
pip freeze > requirements_gpu.txt
```

**Note:** Do not replace `requirements.txt` with `requirements_gpu.txt`. The base requirements.txt should remain generic.

---

## 3. Environment Setup

### 3.1 Windows Setup

**Step 1: Install NVIDIA Driver**
```powershell
# Download from https://www.nvidia.com/Download/index.aspx
# Run installer
# Reboot if required

# Verify
nvidia-smi
```

**Step 2: Install CUDA Toolkit**
```powershell
# Download CUDA 12.1 from https://developer.nvidia.com/cuda-toolkit
# Run installer
# Select "Custom" installation
# Ensure "CUDA" and "cuDNN" are selected

# Add to PATH (if not automatically added)
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin", "User")

# Verify
nvcc --version
```

**Step 3: Install cuDNN**
```powershell
# Download cuDNN 8.9 from https://developer.nvidia.com/cudnn
# Extract to CUDA directory
# Copy files:
#   cudnn*.dll -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
#   cudnn*.h -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include
#   cudnn*.lib -> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64
```

**Step 4: Install Python**
```powershell
# Download Python 3.10 from https://www.python.org/downloads/
# Run installer
# Check "Add Python to PATH"

# Verify
python --version  # Expected: Python 3.10.x
```

**Step 5: Create Virtual Environment**
```powershell
# Create venv
python -m venv venv

# Activate
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

**Step 6: Install PyTorch with CUDA**
```powershell
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

**Step 7: Install Project Dependencies**
```powershell
# Clone repository
git clone https://github.com/yourusername/football-analytics.git
cd football-analytics

# Install dependencies
pip install -r requirements.txt

# Verify
pip list | findstr torch
# Expected: torch 2.1.0+cu121
```

**Step 8: Configure**
```powershell
# Edit config.yaml
# Change: device: "cpu" -> device: "cuda"
notepad config.yaml
```

**Step 9: Verify GPU Detection**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True, GPU: <your GPU name>
```

**Step 10: Run Smoke Test**
```powershell
# 10-frame test
python scripts/run_match_analysis.py --max-frames 10

# Verify GPU utilization
nvidia-smi
# Expected: Python process using GPU memory
```

---

### 3.2 Ubuntu Setup

**Step 1: Install NVIDIA Driver**
```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install driver
sudo apt install nvidia-driver-535  # or latest version

# Reboot
sudo reboot

# Verify
nvidia-smi
```

**Step 2: Install CUDA Toolkit**
```bash
# Download CUDA 12.1 runfile from https://developer.nvidia.com/cuda-toolkit
# Or use apt:
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
```

**Step 3: Install cuDNN**
```bash
# Download cuDNN from https://developer.nvidia.com/cudnn
# Extract and copy:
sudo cp cudnn*.h /usr/local/cuda/include
sudo cp cudnn*.lib /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Verify
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

**Step 4: Install Python**
```bash
# Install Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Verify
python3 --version  # Expected: Python 3.10.x
```

**Step 5: Create Virtual Environment**
```bash
# Clone repository
git clone https://github.com/yourusername/football-analytics.git
cd football-analytics

# Create venv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Step 6: Install PyTorch with CUDA**
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

**Step 7: Install Project Dependencies**
```bash
pip install -r requirements.txt

# Verify
pip list | grep torch
# Expected: torch 2.1.0+cu121
```

**Step 8: Configure**
```bash
# Edit config.yaml
nano config.yaml
# Change: device: "cpu" -> device: "cuda"
```

**Step 9: Verify GPU Detection**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected: CUDA: True, GPU: <your GPU name>
```

**Step 10: Run Smoke Test**
```bash
# 10-frame test
python scripts/run_match_analysis.py --max-frames 10

# Verify GPU utilization
nvidia-smi
# Expected: Python process using GPU memory
```

---

## 4. GPU Verification

### 4.1 System Verification

**Verify NVIDIA driver:**
```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 3060       Off  | 00000000:01:00.0  On |                  N/A |
| 40%   45C    P8    25W / 170W |   2500MiB / 12288MiB |      5%      Default |
+-------------------------------+----------------------+----------------------+
```

### 4.2 PyTorch CUDA Verification

```python
import torch

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())
# Expected: True

# Check GPU count
print("GPU count:", torch.cuda.device_count())
# Expected: 1 or more

# Check GPU name
print("GPU name:", torch.cuda.get_device_name(0))
# Expected: NVIDIA <GPU Model>

# Check CUDA version
print("CUDA version:", torch.version.cuda)
# Expected: 12.1 or 11.8

# Check PyTorch version
print("PyTorch version:", torch.__version__)
# Expected: 2.1.0+cu121 or similar

# Test GPU tensor
x = torch.rand(5, 3).cuda()
print("Tensor device:", x.device)
# Expected: cuda:0
```

### 4.3 YOLO GPU Verification

```python
from ultralytics import YOLO
import torch

# Load model
model = YOLO('yolov8x.pt')

# Check device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Run inference on dummy image
import cv2
import numpy as np
dummy_img = np.zeros((1280, 1280, 3), dtype=np.uint8)
results = model(dummy_img, device=device)

# Verify GPU usage
print("Model device:", next(model.model.parameters()).device)
# Expected: cuda:0

# Check nvidia-smi during inference
# Expected: GPU memory usage increases
```

### 4.4 MediaPipe Verification

```python
# MediaPipe runs on CPU - verify it doesn't break on GPU system
from app.pose.pose_estimator import PoseEstimator

pose = PoseEstimator(fps=25.0, model_complexity=1)
print("Pose estimator initialized")

# Test with dummy image
import cv2
import numpy as np
dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
result = pose.estimate(dummy_img, track_id=1)
print("Pose result:", result.success)
```

### 4.5 Full Pipeline Verification

```bash
# Run 10-frame smoke test
python scripts/run_match_analysis.py --max-frames 10

# Check logs for:
# - "YOLO Device: cuda:0"
# - "Loaded homography calibration"
# - "Models & Pipelines Initialized"
# - "Frame processing loop completed"

# Verify outputs
ls outputs/
# Expected: detection.mp4, tracking.mp4, speed_debug.csv, etc.
```

### 4.6 Performance Verification

```bash
# Run 500-frame benchmark
python scripts/run_match_analysis.py --max-frames 500

# Monitor GPU utilization in another terminal
watch -n 1 nvidia-smi
# Expected: GPU utilization 50-100% during YOLO inference

# Check performance report
cat outputs/performance_report.json | python -m json.tool
# Expected: Average FPS > 10 (GPU) vs 0.1 (CPU)
```

---

## 5. Benchmark Plan

### 5.1 Benchmark Scenarios

| Scenario | Frames | Purpose |
|----------|--------|---------|
| Smoke test | 10 | Quick validation |
| Short benchmark | 500 | Standard validation |
| Medium benchmark | 1000 | Performance profiling |
| Large benchmark | 10000 | Stress test (requires ~30 min on GPU) |
| Full match | 135000 | Production validation (requires ~1-2 hours on GPU) |

### 5.2 Metrics to Collect

```python
# In scripts/run_match_analysis.py, the pipeline already collects:
# - Module timings (ms/frame)
# - Processed frames
# - FPS

# Additional metrics to collect on GPU:
```

**YOLO Inference Time:**
```python
import time

t_start = time.perf_counter()
results = model(frame, device="cuda:0")
t_end = time.perf_counter()
yolo_time_ms = (t_end - t_start) * 1000
```

**Tracking Time:**
```python
t_start = time.perf_counter()
# ... tracking code ...
t_end = time.perf_counter()
tracking_time_ms = (t_end - t_start) * 1000
```

**Pose Estimation Time:**
```python
t_start = time.perf_counter()
# ... pose code ...
t_end = time.perf_counter()
pose_time_ms = (t_end - t_start) * 1000
```

**Analytics Time:**
```python
t_start = time.perf_counter()
# ... analytics code ...
t_end = time.perf_counter()
analytics_time_ms = (t_end - t_start) * 1000
```

**GPU Utilization:**
```bash
# In separate terminal during benchmark
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
# Log to gpu_utilization.csv
```

**CPU Utilization:**
```bash
# Windows
typeperf "\Processor(_Total)\% Processor Time" -sc 1000 > cpu_utilization.csv

# Linux
mpstat -P ALL 1 > cpu_utilization.csv
```

### 5.3 Benchmark Script

```bash
#!/bin/bash
# benchmark_gpu.sh

echo "Starting GPU benchmark..."

# Clean outputs
rm -rf outputs/*
mkdir -p outputs/benchmark

# Smoke test (10 frames)
echo "Smoke test (10 frames)..."
python scripts/run_match_analysis.py --max-frames 10
cp outputs/performance_report.json outputs/benchmark/perf_10frames.json

# Short benchmark (500 frames)
echo "Short benchmark (500 frames)..."
python scripts/run_match_analysis.py --max-frames 500
cp outputs/performance_report.json outputs/benchmark/perf_500frames.json
cp outputs/speed_debug.csv outputs/benchmark/speed_debug_500frames.csv

# Medium benchmark (1000 frames)
echo "Medium benchmark (1000 frames)..."
python scripts/run_match_analysis.py --max-frames 1000
cp outputs/performance_report.json outputs/benchmark/perf_1000frames.json
cp outputs/speed_debug.csv outputs/benchmark/speed_debug_1000frames.csv

# Large benchmark (10000 frames) - optional
echo "Large benchmark (10000 frames)..."
python scripts/run_match_analysis.py --max-frames 10000
cp outputs/performance_report.json outputs/benchmark/perf_10000frames.json
cp outputs/speed_debug.csv outputs/benchmark/speed_debug_10000frames.csv

echo "Benchmark complete. Results in outputs/benchmark/"
```

### 5.4 Expected Performance Targets

| Metric | Target (GPU) | Target (CPU) | Notes |
|--------|--------------|--------------|-------|
| Average FPS | > 20 FPS | < 1 FPS | Real-time capable |
| YOLO inference | < 50 ms | < 5000 ms | Primary bottleneck |
| Total frame time | < 100 ms | < 10000 ms | End-to-end latency |
| GPU utilization | > 80% | N/A | During YOLO inference |
| VRAM usage | < 8GB | N/A | For RTX 3060 12GB |

---

## 6. Output Validation

### 6.1 Validation Procedure

**Step 1: Run on CPU**
```bash
# Configure for CPU
# Edit config.yaml: device: "cpu"

# Run 500-frame benchmark
python scripts/run_match_analysis.py --max-frames 500

# Save outputs
cp -r outputs outputs_cpu
```

**Step 2: Run on GPU**
```bash
# Configure for GPU
# Edit config.yaml: device: "cuda"

# Run 500-frame benchmark
python scripts/run_match_analysis.py --max-frames 500

# Save outputs
cp -r outputs outputs_gpu
```

**Step 3: Compare Outputs**
```bash
# Compare file lists
ls outputs_cpu/
ls outputs_gpu/
# Expected: Same files

# Compare tracking
diff <(sort outputs_cpu/ball_tracks.json) <(sort outputs_gpu/ball_tracks.json)
# Expected: Minor differences in tracking IDs

# Compare homography
diff <(sort outputs_cpu/homography_validation.json) <(sort outputs_gpu/homography_validation.json)
# Expected: Identical

# Compare player statistics
diff <(sort outputs_cpu/player_statistics.csv) <(sort outputs_gpu/player_statistics.csv)
# Expected: Minor numerical differences (< 1%)

# Compare speed debug
python -c "
import pandas as pd
cpu = pd.read_csv('outputs_cpu/speed_debug.csv')
gpu = pd.read_csv('outputs_gpu/speed_debug.csv')
print('CPU speed stats:')
print(cpu['speed_kmh'].describe())
print('\nGPU speed stats:')
print(gpu['speed_kmh'].describe())
print('\nMax difference:', (cpu['speed_kmh'] - gpu['speed_kmh']).abs().max())
"
# Expected: Max difference < 0.1 km/h
```

### 6.2 Expected Differences

| Output | Expected Difference | Cause |
|--------|---------------------|-------|
| Tracking IDs | Minor | GPU inference order may differ |
| Homography | None | CPU-only module |
| Speed/Distance | < 1% | FP16 precision on GPU |
| Heatmaps | < 1% | FP16 precision on GPU |
| Reports | < 1% | Derived from speed/distance |
| Formation | Same or better | GPU may detect more formations |

### 6.3 Validation Checklist

- [ ] All output files generated on both CPU and GPU
- [ ] Tracking IDs match within 5% tolerance
- [ ] Homography coordinates identical
- [ ] Speed values match within 0.5 km/h
- [ ] Distance values match within 1%
- [ ] Heatmaps visually identical
- [ ] Reports contain same number of entries
- [ ] Performance improved by > 100x

### 6.4 Troubleshooting

**Issue: `torch.cuda.is_available()` returns False**
- Check NVIDIA driver: `nvidia-smi`
- Check PyTorch build: `pip show torch | grep cu`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

**Issue: YOLO still runs on CPU**
- Verify device: `python -c "import torch; print(torch.cuda.is_available())"`
- Check YOLO device: `print(next(model.model.parameters()).device)`
- Ensure model loaded with `.to(device)` after initialization

**Issue: Out of Memory (OOM)**
- Reduce batch size in YOLO inference
- Switch to smaller model (YOLOv8m or YOLOv8n)
- Enable gradient checkpointing
- Reduce image size (1280 -> 640)

**Issue: Different tracking IDs on GPU**
- This is expected and acceptable
- Downstream analytics are unaffected
- If critical, set deterministic mode: `torch.backends.cudnn.deterministic = True`

---

## 7. Production Configuration

### 7.1 config.yaml for GPU

```yaml
# config.yaml
device: "cuda"  # Change from "cpu" to "cuda"

models:
  yolo_model_path: "yolov8x.pt"  # Keep for accuracy
  # yolo_model_path: "yolov8m.pt"  # Use for faster inference
  confidence_threshold: 0.25
  iou_threshold: 0.5
  image_size: 1280  # Reduce to 640 for faster inference

video:
  input_path: "path/to/video.mp4"
  output_dir: "outputs"
  max_frames: 500
  fps: 25.0
  resolution: [1280, 720]

speed_estimation:
  ema_alpha: 0.15
  max_displacement_m: 0.5
  min_movement_m: 0.2

# ... rest of config
```

### 7.2 Docker Configuration (Optional)

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt update && apt install -y python3.10 python3-pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch with CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics
RUN pip install ultralytics>=8.1.0

# Install OpenCV
RUN pip install opencv-python>=4.8.0

# Install MediaPipe
RUN pip install mediapipe>=0.10.0

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Run pipeline
CMD ["python", "scripts/run_match_analysis.py", "--max-frames", "500"]
```

**Run with Docker:**
```bash
docker build -t football-analytics .
docker run --gpus all -v $(pwd)/outputs:/app/outputs football-analytics
```

### 7.3 Environment Variables

```bash
# Optional: Force GPU
export CUDA_VISIBLE_DEVICES=0

# Optional: Disable CUDA (for testing)
export CUDA_VISIBLE_DEVICES=""

# Optional: Set PyTorch deterministic mode (reproducibility)
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

---

## 8. Troubleshooting Guide

### Common Issues

**Issue 1: CUDA out of memory**
```python
# Solution 1: Reduce batch size
results = model(frame, device="cuda:0", batch=1)

# Solution 2: Use smaller model
MODEL_WEIGHTS = "yolov8n.pt"  # Instead of yolov8x.pt

# Solution 3: Enable gradient checkpointing
model.model.gradient_checkpointing = True
```

**Issue 2: Slow GPU performance**
```bash
# Check GPU utilization
nvidia-smi
# If utilization < 50%, bottleneck is elsewhere

# Check PCIe bandwidth
nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv
# Expected: Gen 3 or 4, x16
```

**Issue 3: Different results on GPU vs CPU**
```python
# Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

**Issue 4: MediaPipe CPU bottleneck**
```bash
# MediaPipe runs on CPU by design
# Acceptable: MediaPipe is ~50ms/frame (1.2% of total time on GPU)
# No action required
```

---

## 9. Maintenance

### Updating CUDA

When new CUDA versions are released:
1. Update NVIDIA driver
2. Install new CUDA toolkit
3. Reinstall PyTorch with matching CUDA version
4. Test with smoke test

### Monitoring GPU Health

```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Check GPU errors
nvidia-smi --query-gpu=errors.uncorrected.parity --format=csv

# Check GPU utilization over time
nvidia-smi dmon -s pucvmet
```

### Backup Configuration

```bash
# Export configuration
cp config.yaml config.yaml.backup
cp outputs/performance_report.json outputs/performance_report.json.backup

# Document GPU settings
nvidia-smi --query-gpu=all --format=csv > gpu_info.txt
```

---

## 10. Support

For issues related to:
- **NVIDIA drivers:** https://www.nvidia.com/Download/index.aspx
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit
- **cuDNN:** https://discuss.nvidia.com/cudnn
- **PyTorch:** https://pytorch.org/get-started/locally/
- **Ultralytics YOLO:** https://docs.ultralytics.com/
- **MediaPipe:** https://google.github.io/mediapipe/

---

## Summary

**This guide documents the complete process for deploying the football analytics platform on a CUDA-enabled machine.**

**Key points:**
1. No source code changes required - the project is already GPU-ready
2. Only hardware and software installation is needed
3. Dynamic device selection is already implemented
4. Expected performance improvement: 200-300x faster on GPU

**Quick start:**
1. Install NVIDIA GPU, driver, CUDA, cuDNN
2. Install Python 3.10+
3. Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
4. Install dependencies: `pip install -r requirements.txt`
5. Configure: `config.yaml` -> `device: "cuda"`
6. Verify: `python -c "import torch; print(torch.cuda.is_available())"`
7. Run: `python scripts/run_match_analysis.py --max-frames 500`

**Expected performance:**
- CPU: 0.1 FPS (70 minutes for 500 frames)
- GPU: 20-30 FPS (20-30 seconds for 500 frames)