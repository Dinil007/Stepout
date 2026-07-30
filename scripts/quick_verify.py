"""Quick integration verification — small frame count to validate data flow."""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_config
from app.pipeline import PipelineManager
from app.pipeline.data_models import PipelineInput

cfg = get_config().raw
cfg['video']['max_frames'] = 15
pipeline_input = PipelineInput(
    video_path=Path(cfg['video']['input_path']),
    output_dir=Path(cfg['video']['output_dir']),
    max_frames=15
)
manager = PipelineManager(cfg)
output = manager.run(pipeline_input)
print("\n===== INTEGRATION VERIFICATION =====")
print(f"Success: {output.success}")
if output.annotated_video_path:
    print(f"Annotated video: {output.annotated_video_path}")
if output.player_metrics_csv:
    print(f"Player metrics: {output.player_metrics_csv}")
if output.ball_metrics_csv:
    print(f"Ball metrics: {output.ball_metrics_csv}")
if output.summary_json_path:
    print(f"Summary JSON: {output.summary_json_path}")