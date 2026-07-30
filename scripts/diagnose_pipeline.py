"""
Pipeline Diagnostic Script
===========================
Diagnoses the exact cause of 44-byte corrupted output video files.

Steps executed:
  1. VideoWriter open/isOpened() check for every output stream.
  2. Per-frame logging: frame shape, detection count, tracking count.
  3. Wraps every stage in try/except with full traceback.
  4. After loop: frames read, detected, tracked, written.
  5. If detection always zero: tests YOLO loading, classes, confidence, prints prediction.
"""

import sys
import traceback
import time
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch

# ── locate input video ────────────────────────────────────────────────────────
VIDEO_CANDIDATES = [
    "videos/sample_video.mp4",
    "videos/input.mp4",
    "videos/match.mp4",
]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_VIDEO = None
for candidate in VIDEO_CANDIDATES:
    p = Path(candidate)
    if p.exists() and p.stat().st_size > 1000:
        INPUT_VIDEO = str(p)
        break

if INPUT_VIDEO is None:
    # Try finding any .mp4 in videos/
    for p in Path("videos").glob("*.mp4"):
        if p.stat().st_size > 1000:
            INPUT_VIDEO = str(p)
            break
    # Also try preprocessed frames
    if INPUT_VIDEO is None:
        for p in Path("outputs").glob("*.mp4"):
            if "preprocessing" in p.name.lower() and p.stat().st_size > 1000:
                INPUT_VIDEO = str(p)
                break

print("=" * 65)
print("STEPOUT AI PIPELINE DIAGNOSTIC TOOL")
print("=" * 65)
print(f"Python       : {sys.version}")
print(f"OpenCV       : {cv2.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
print(f"Input Video  : {INPUT_VIDEO}")
print("=" * 65)

if INPUT_VIDEO is None:
    print("\n[FATAL] No valid input video found in videos/ or outputs/.")
    print("Available files in videos/:")
    for p in Path("videos").iterdir():
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Open video and verify properties
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Opening input video...")
try:
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[FATAL] cv2.VideoCapture failed to open: {INPUT_VIDEO}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution   : {w}x{h}")
    print(f"  FPS          : {fps:.2f}")
    print(f"  Total frames : {total}")

    if w == 0 or h == 0:
        print("[FATAL] Video reports 0x0 resolution. File is likely corrupted.")
        sys.exit(1)

    # Read a test frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[FATAL] Cannot read first frame from video.")
        sys.exit(1)
    actual_shape = test_frame.shape
    print(f"  First frame shape (actual): {actual_shape}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[OK] Video opened successfully.")
except Exception:
    print("[FATAL] Exception opening video:")
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: VideoWriter isOpened() check
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Testing VideoWriter for every output stream...")

wh = (w, h)
pwh = (1050, 680)
fps_out = fps if fps > 0 else 30.0

CODECS_TO_TRY = [
    ("mp4v", "detection_diag.mp4"),
    ("avc1", "detection_diag_avc1.mp4"),
    ("XVID", "detection_diag_xvid.avi"),
]

working_fourcc = None
working_ext = ".mp4"
for codec_str, test_name in CODECS_TO_TRY:
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    test_path = str(OUTPUT_DIR / test_name)
    tw = cv2.VideoWriter(test_path, fourcc, fps_out, wh)
    opened = tw.isOpened()
    print(f"  Codec '{codec_str}' @ {w}x{h} -> isOpened()={opened}  [{test_path}]")
    if opened:
        # Write one real frame to confirm
        tw.write(test_frame)
        tw.release()
        sz = Path(test_path).stat().st_size
        print(f"    -> File size after writing 1 frame: {sz} bytes")
        if sz > 100 and working_fourcc is None:
            working_fourcc = codec_str
            working_ext = Path(test_name).suffix
            print(f"    -> CODEC '{codec_str}' IS WORKING.")
    else:
        tw.release()

if working_fourcc is None:
    print("\n[FATAL] No codec produced a working VideoWriter on this system.")
    print("  Possible fixes:")
    print("    1. Install ffmpeg: winget install Gyan.FFmpeg")
    print("    2. Install opencv-python full build: pip install opencv-python")
    sys.exit(1)

print(f"\n[OK] Using codec '{working_fourcc}' with extension '{working_ext}'")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Open all output writers with working codec
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Opening all output VideoWriters...")
fourcc = cv2.VideoWriter_fourcc(*working_fourcc)

writers = {
    "detection":         cv2.VideoWriter(str(OUTPUT_DIR / f"detection{working_ext}"),         fourcc, fps_out, wh),
    "tracking":          cv2.VideoWriter(str(OUTPUT_DIR / f"tracking{working_ext}"),          fourcc, fps_out, wh),
    "team_classification": cv2.VideoWriter(str(OUTPUT_DIR / f"team_classification{working_ext}"), fourcc, fps_out, wh),
    "pitch_view":        cv2.VideoWriter(str(OUTPUT_DIR / f"pitch_view{working_ext}"),        fourcc, fps_out, pwh),
}

all_open = True
for name, wtr in writers.items():
    opened = wtr.isOpened()
    print(f"  writer['{name}'].isOpened() = {opened}")
    if not opened:
        print(f"  [FATAL] writer['{name}'] FAILED TO OPEN.")
        all_open = False

if not all_open:
    print("[FATAL] One or more VideoWriters failed. Aborting.")
    sys.exit(1)
print("[OK] All VideoWriters opened successfully.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Load YOLO model
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Loading YOLO model...")
model = None
MODEL_CANDIDATES = ["yolov8x.pt", "yolov8m.pt", "yolov8n.pt"]
model_path_used = None
try:
    from ultralytics import YOLO
    for mp in MODEL_CANDIDATES:
        mpath = Path(mp)
        if mpath.exists():
            model_path_used = str(mpath)
            break
    if model_path_used is None:
        print("[FATAL] No YOLO model found. Checked:", MODEL_CANDIDATES)
        sys.exit(1)
    print(f"  Loading model: {model_path_used}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path_used)
    # Warmup
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, classes=[0, 32], conf=0.25, iou=0.5,
                      imgsz=640, verbose=False, device=device)
    print(f"[OK] YOLO model loaded: {model_path_used} on {device}")
    print(f"  Classes in model: {list(model.names.values())[:10]}...")
    print(f"  Class 0 = '{model.names.get(0, 'N/A')}'   Class 32 = '{model.names.get(32, 'N/A')}'")
except Exception:
    print("[FATAL] Exception loading YOLO:")
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Run per-frame diagnostic loop
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Running per-frame diagnostic loop (up to 30 frames)...")
print("-" * 65)

MAX_DIAG_FRAMES = 30
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frames_read = 0
frames_detected = 0
frames_tracked = 0
det_written = 0
trk_written = 0
team_written = 0
pitch_written = 0

first_pred_printed = False
pitch_canvas_blank = np.zeros((680, 1050, 3), dtype=np.uint8)

with torch.inference_mode():
    while cap.isOpened() and frames_read < MAX_DIAG_FRAMES:
        # ---- Read frame ----
        try:
            ret, frame = cap.read()
        except Exception:
            print(f"[FATAL] Exception on cap.read() at frame {frames_read + 1}:")
            traceback.print_exc()
            break

        if not ret or frame is None:
            print(f"[WARN] cap.read() returned False at frame {frames_read + 1}. End of stream.")
            break

        frames_read += 1
        fshape = frame.shape

        # ---- YOLO Detection ----
        det_count = 0
        trk_count = 0
        annotated_det = frame.copy()
        annotated_track = frame.copy()
        annotated_team = frame.copy()
        pitch_frame = pitch_canvas_blank.copy()

        try:
            results = model.track(
                source=frame, persist=True,
                classes=[0, 32], conf=0.25, iou=0.5,
                imgsz=640, verbose=False, device=device
            )

            if results and results[0].boxes is not None:
                det_count = len(results[0].boxes)
                trk_count = sum(1 for b in results[0].boxes if b.id is not None)

                # Print FIRST prediction details
                if not first_pred_printed:
                    first_pred_printed = True
                    print(f"\n[FIRST PREDICTION @ frame {frames_read}]")
                    print(f"  Raw results[0].boxes.cls   = {results[0].boxes.cls}")
                    print(f"  Raw results[0].boxes.conf  = {results[0].boxes.conf}")
                    print(f"  Raw results[0].boxes.id    = {results[0].boxes.id}")
                    print(f"  Total detections: {det_count}")
                    print(f"  YOLO names map (0, 32): person={model.names.get(0)}, sports ball={model.names.get(32)}")
                    print()

                # Draw bboxes
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    color = (0, 255, 0) if cls_id == 0 else (0, 255, 255)
                    cv2.rectangle(annotated_det, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(annotated_track, (x1, y1), (x2, y2), color, 2)

        except Exception:
            print(f"[FATAL] Exception during YOLO detection at frame {frames_read}:")
            traceback.print_exc()
            break

        if det_count > 0:
            frames_detected += 1
        if trk_count > 0:
            frames_tracked += 1

        # ---- Write frames ----
        write_ok_det = False
        write_ok_trk = False
        write_ok_team = False
        write_ok_pitch = False

        try:
            writers["detection"].write(annotated_det)
            write_ok_det = writers["detection"].isOpened()
            det_written += 1
        except Exception:
            print(f"[FATAL] Exception writing detection frame {frames_read}:")
            traceback.print_exc()
            break

        try:
            writers["tracking"].write(annotated_track)
            write_ok_trk = writers["tracking"].isOpened()
            trk_written += 1
        except Exception:
            print(f"[FATAL] Exception writing tracking frame {frames_read}:")
            traceback.print_exc()
            break

        try:
            writers["team_classification"].write(annotated_team)
            write_ok_team = writers["team_classification"].isOpened()
            team_written += 1
        except Exception:
            print(f"[FATAL] Exception writing team_classification frame {frames_read}:")
            traceback.print_exc()
            break

        try:
            writers["pitch_view"].write(pitch_frame)
            write_ok_pitch = writers["pitch_view"].isOpened()
            pitch_written += 1
        except Exception:
            print(f"[FATAL] Exception writing pitch_view frame {frames_read}:")
            traceback.print_exc()
            break

        print(f"  Frame {frames_read:3d}/{MAX_DIAG_FRAMES} | "
              f"shape={fshape} | det={det_count:2d} trk={trk_count:2d} | "
              f"write_det={write_ok_det} write_trk={write_ok_trk} "
              f"write_team={write_ok_team} write_pitch={write_ok_pitch}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Release & report
# ══════════════════════════════════════════════════════════════════════════════
cap.release()
for wtr in writers.values():
    wtr.release()

print("\n" + "=" * 65)
print("[STEP 6] DIAGNOSTIC SUMMARY")
print("=" * 65)
print(f"  Frames read              : {frames_read}")
print(f"  Frames with detections   : {frames_detected}  ({frames_detected}/{frames_read})")
print(f"  Frames with tracking IDs : {frames_tracked}  ({frames_tracked}/{frames_read})")
print(f"  Frames written (detection) : {det_written}")
print(f"  Frames written (tracking)  : {trk_written}")
print(f"  Frames written (team)      : {team_written}")
print(f"  Frames written (pitch)     : {pitch_written}")
print()
print("  Output file sizes:")
for name in ["detection", "tracking", "team_classification", "pitch_view"]:
    p = OUTPUT_DIR / f"{name}{working_ext}"
    if p.exists():
        print(f"    {name}{working_ext}: {p.stat().st_size:,} bytes")
    else:
        print(f"    {name}{working_ext}: NOT FOUND")

print("\n[DONE] Diagnostic complete.")
if frames_detected == 0:
    print("\n[!] WARNING: Zero detections across all frames.")
    print("    Possible causes:")
    print("    1. Video resolution/classes mismatch (check YOLO class names above).")
    print("    2. Confidence threshold too high.")
    print("    3. Model loaded but wrong class indices.")
