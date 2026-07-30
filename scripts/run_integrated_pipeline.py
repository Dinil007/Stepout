"""
Integrated Pipeline Runner

Entry point for the StepOut AI production pipeline.
Loads config, creates PipelineManager, runs on input video.
"""

import os
import sys
import time
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_config
from app.pipeline import PipelineManager
from app.pipeline.data_models import PipelineInput


def main():
    print("=" * 70)
    print("  StepOut AI — Integrated Production Pipeline")
    print("=" * 70)

    config = get_config()
    cfg = config.raw
    video_path = Path(cfg.get('video', {}).get('input_path', 'D:/stepout/videos/raw/match30.mp4'))
    max_frames = cfg.get('video', {}).get('max_frames', 750)
    output_dir = Path(cfg.get('video', {}).get('output_dir', 'outputs'))

    if not video_path.exists():
        print(f"[WARN] Video not found: {video_path}")
        print("[WARN] Using synthetic data mode (processing will generate test outputs)")

    print(f"\nVideo:       {video_path}")
    print(f"Max Frames:  {max_frames}")
    print(f"Output Dir:  {output_dir}")
    print(f"Device:      {cfg.get('device', 'cpu')}")
    print()

    pipeline_input = PipelineInput(
        video_path=video_path,
        output_dir=output_dir,
        max_frames=max_frames
    )

    print("Initialising Pipeline Manager...")
    t0 = time.time()
    manager = PipelineManager(cfg)
    init_time = time.time() - t0
    print(f"[OK] {len(manager.stages)} stages initialised in {init_time:.2f}s")

    print("\nRunning pipeline...")
    t1 = time.time()
    output = manager.run(pipeline_input)
    run_time = time.time() - t1

    print(f"\n{'=' * 70}")
    print(f"  Pipeline {'SUCCESS' if output.success else 'FAILED'}")
    print(f"  Total time: {output.total_execution_time_s:.2f}s (init: {init_time:.2f}s, run: {run_time:.2f}s)")
    print(f"  Frames processed: {output.total_frames_processed}")
    print(f"{'=' * 70}\n")

    print("Stage Timings:")
    for name in manager.STAGE_ORDER:
        result = manager.get_stage_result(name)
        if result:
            status = "OK" if result.success else "FAIL"
            print(f"  {name:<20} [{status}] {result.execution_time_s:.3f}s ({result.frames_processed} frames)")
            if result.error:
                print(f"  {'':>20} Error: {result.error}")

    print("\nOutputs:")
    if output.annotated_video_path:
        print(f"  Annotated Video:  {output.annotated_video_path}")
    if output.player_metrics_csv:
        print(f"  Player Metrics:   {output.player_metrics_csv}")
    if output.ball_metrics_csv:
        print(f"  Ball Metrics:     {output.ball_metrics_csv}")
    if output.biomechanics_csv:
        print(f"  Biomechanics:     {output.biomechanics_csv}")
    if output.summary_json_path:
        print(f"  Summary JSON:     {output.summary_json_path}")

    logs_dir = output_dir / "logs"
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        for lf in log_files:
            print(f"  Log:              {lf}")

    if output.success:
        print(f"\n{'=' * 70}")
        print("  Pipeline completed successfully.")
        print(f"{'=' * 70}")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Pipeline FAILED: {output.error}")
        print(f"{'=' * 70}")
        sys.exit(1)


if __name__ == "__main__":
    main()