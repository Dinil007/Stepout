"""
Pipeline Logger

Structured logging for the production pipeline with
per-stage timing, frame counts, and error tracking.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class PipelineLogger:
    """
    Centralised logger for the StepOut AI pipeline.
    Writes structured logs to both console and file.
    Tracks per-stage execution times, frame counts, and errors.
    """

    def __init__(self, log_dir: Path, pipeline_name: str = "stepout_pipeline"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.timings: Dict[str, float] = {}
        self.frame_counts: Dict[str, int] = {}
        self.errors: Dict[str, str] = {}

        log_file = self.log_dir / f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logging.getLogger(f"Pipeline_{pipeline_name}")
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "[%(levelname)s] %(message)s"
        ))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.propagate = False

        self.start_time = time.time()

    def stage_start(self, stage_name: str) -> None:
        self.logger.info(">>> [%s] starting", stage_name)
        self.timings[f"{stage_name}_start"] = time.time()

    def stage_end(self, stage_name: str, frames_processed: int = 0) -> float:
        elapsed = time.time() - self.timings.pop(f"{stage_name}_start", time.time())
        self.timings[stage_name] = elapsed
        self.frame_counts[stage_name] = frames_processed
        self.logger.info(
            "<<< [%s] completed in %.3fs (%d frames)",
            stage_name, elapsed, frames_processed
        )
        return elapsed

    def stage_error(self, stage_name: str, error: str) -> None:
        self.errors[stage_name] = error
        self.logger.error("!!! [%s] FAILED: %s", stage_name, error)

    def stage_warning(self, stage_name: str, warning: str) -> None:
        self.logger.warning("[%s] %s", stage_name, warning)

    def stage_info(self, stage_name: str, info: str) -> None:
        self.logger.info("[%s] %s", stage_name, info)

    def log_summary(self, total_frames: int) -> Dict:
        total_time = time.time() - self.start_time
        summary = {
            'total_execution_time_s': round(total_time, 3),
            'total_frames': total_frames,
            'stages': len(self.timings),
            'errors': len(self.errors),
            'stage_timings': {k: round(v, 3) for k, v in self.timings.items() if not k.endswith('_start')},
            'frame_counts': self.frame_counts,
            'error_details': self.errors
        }
        self.logger.info(
            "Pipeline completed in %.3f s | %d frames | %d errors",
            total_time, total_frames, len(self.errors)
        )
        return summary

    def save_summary_json(self, output_path: Path) -> None:
        summary = {
            'pipeline_status': 'FAILED' if self.errors else 'SUCCESS',
            'total_execution_time_s': round(time.time() - self.start_time, 3),
            'stage_timings_s': {k: round(v, 3) for k, v in self.timings.items() if not k.endswith('_start')},
            'frame_counts': self.frame_counts,
            'errors': self.errors
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info("Pipeline summary saved to %s", output_path)