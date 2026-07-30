"""Run quick pipeline check and write verification report to file."""
import sys, json
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_config
from app.pipeline import PipelineManager
from app.pipeline.data_models import PipelineInput

report_path = ROOT_DIR / 'outputs' / 'verify_report.txt'
cfg = get_config().raw
cfg['video']['max_frames'] = 20
pipeline_input = PipelineInput(
    video_path=Path(cfg['video']['input_path']),
    output_dir=Path(cfg['video']['output_dir']),
    max_frames=20
)
manager = PipelineManager(cfg)
output = manager.run(pipeline_input)

lines = []
lines.append("===== INTEGRATION VERIFICATION REPORT =====")
lines.append(f"Success: {output.success}")
if output.annotated_video_path:
    av = output.annotated_video_path
    size = av.stat().st_size if av.exists() else 0
    lines.append(f"Annotated video: {av} ({size//1024}KB)")
if output.player_metrics_csv:
    lines.append(f"Player metrics: {output.player_metrics_csv}")
if output.ball_metrics_csv:
    lines.append(f"Ball metrics: {output.ball_metrics_csv}")
if output.summary_json_path:
    lines.append(f"Summary JSON: {output.summary_json_path}")
for name in manager.STAGE_ORDER:
    r = manager.get_stage_result(name)
    if r:
        status = "OK" if r.success else "FAIL"
        lines.append(f"  {name:<20} [{status}] {r.execution_time_s:.3f}s ({r.frames_processed} frames)")
        if r.error:
            lines.append(f"  {'':>20} Error: {r.error}")

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nReport saved to: {report_path}")