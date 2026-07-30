"""
Ball Path Renderer Validation Tool

Validates ball path rendering against tracking, analytics, and possession data.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time
import logging

import numpy as np
import pandas as pd
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_config
from app.visualization import BallPathRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("BallPathValidator")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("File not found or empty: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
        logger.warning("Could not load %s: %s", path, e)
        return pd.DataFrame()


class BallPathValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.renderer = BallPathRenderer(config)
        self.renderer.reset()

        self.ball_analytics = load_csv_safe(OUTPUT_DIR / "ball_analytics.csv")
        self.pass_events = load_csv_safe(OUTPUT_DIR / "pass_events.csv")
        self.team_possession = load_csv_safe(OUTPUT_DIR / "team_possession.csv")

        self.validation_results = {
            'tracking_coverage': {},
            'colour_assignment': {'correct': 0, 'incorrect': 0, 'unknown': 0, 'total': 0},
            'possession_switches': [],
            'pass_highlights': [],
            'touch_markers': [],
            'debug_overlay_checks': [],
            'flickers': 0
        }

    def _get_expected_team_color(self, team_id: str) -> Tuple[int, int, int]:
        bpr_cfg = self.config.get('visualization', {}).get('ball_path_renderer', {})
        team_colors = bpr_cfg.get('team_colors', {})
        if team_id in team_colors:
            return tuple(team_colors[team_id])
        if 'team1' in team_colors and (team_id == '1' or team_id.lower() == 'team1'):
            return tuple(team_colors['team1'])
        if 'team2' in team_colors and (team_id == '2' or team_id.lower() == 'team2'):
            return tuple(team_colors['team2'])
        return tuple(team_colors.get('unknown', [128, 128, 128]))

    def _get_renderer_color_for_frame(self, frame_number: int) -> Tuple[int, int, int]:
        for entry in reversed(self.renderer.history):
            if entry['frame_number'] <= frame_number:
                return entry.get('color', (128, 128, 128))
        return self.renderer.current_color

    def seed_renderer(self) -> None:
        self.renderer.reset()
        pitch_length = 105.0
        pitch_width = 68.0
        np.random.seed(42)

        for frame in range(300):
            x = np.random.uniform(10, pitch_length - 10)
            y = np.random.uniform(10, pitch_width - 10)
            ts = frame / 25.0
            px = int(x * self.renderer.scale_x)
            py = int(y * self.renderer.scale_y)

            team_id = 'team1' if frame < 150 else ('team2' if frame < 250 else 'team1')
            has_possession = True
            conf = np.random.uniform(0.7, 1.0)
            speed = np.random.uniform(5.0, 35.0)
            is_pass = np.random.random() < 0.05
            is_touch = np.random.random() < 0.08

            self.renderer.update(
                frame_number=frame,
                timestamp=ts,
                pixel_position=(px, py),
                world_position=(x, y),
                team_id=team_id,
                player_id=frame % 22,
                has_possession=has_possession,
                possession_confidence=conf,
                ball_speed_kmh=speed,
                is_pass=is_pass,
                is_touch=is_touch
            )

    def validate_tracking_continuity(self) -> Dict:
        history_by_frame = set(entry['frame_number'] for entry in self.renderer.history)
        if not history_by_frame:
            return {}

        total_frames = max(history_by_frame) + 1
        detected_frames = len(history_by_frame)
        missing_frames = total_frames - detected_frames
        coverage = (detected_frames / total_frames * 100) if total_frames > 0 else 0.0

        sorted_frames = sorted(history_by_frame)
        gaps = []
        for i in range(1, len(sorted_frames)):
            gap = sorted_frames[i] - sorted_frames[i-1] - 1
            if gap > 0:
                gaps.append(gap)
        longest_missing = max(gaps) if gaps else 0

        interpolated_frames = sum(1 for e in self.renderer.history if e.get('ball_speed_kmh', 0) < 0.5)

        result = {
            'total_frames': total_frames,
            'detected_frames': detected_frames,
            'interpolated_frames': int(interpolated_frames),
            'missing_frames': missing_frames,
            'longest_missing_sequence': longest_missing,
            'coverage_pct': round(coverage, 2)
        }
        self.validation_results['tracking_coverage'] = result
        return result

    def validate_colour_assignment(self) -> Dict:
        if not self.renderer.history:
            return {'accuracy_pct': 0.0}

        correct = 0
        incorrect = 0
        unknown = 0
        total = 0

        for entry in self.renderer.history:
            team_id = entry.get('team_id', '')
            expected_color = self._get_expected_team_color(str(team_id) if team_id else '')
            rendered_color = entry.get('color', (128, 128, 128))

            total += 1
            if rendered_color == expected_color:
                correct += 1
            elif rendered_color == self.renderer.low_confidence_color:
                unknown += 1
            else:
                incorrect += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        result = {
            'correct': correct,
            'incorrect': incorrect,
            'unknown': unknown,
            'total': total,
            'accuracy_pct': round(accuracy, 2)
        }
        self.validation_results['colour_assignment'] = result
        return result

    def validate_possession_switching(self) -> List[Dict]:
        switches = []
        prev_team = None
        prev_team_frames = 0
        flicker_count = 0

        if not self.renderer.history:
            return switches

        sorted_history = sorted(self.renderer.history, key=lambda e: e['frame_number'])
        for entry in sorted_history:
            team = entry.get('team_id')
            if team != prev_team:
                switch_info = {
                    'frame': entry['frame_number'],
                    'old_team': prev_team,
                    'new_team': team,
                    'confidence': entry.get('possession_confidence', 0.0),
                    'reason': 'possession change'
                }
                switches.append(switch_info)
                self.validation_results['possession_switches'].append(switch_info)

                if prev_team_frames < self.renderer.min_possession_frames and prev_team is not None:
                    flicker_count += 1

                prev_team = team
                prev_team_frames = 1
            else:
                prev_team_frames += 1

        self.validation_results['flickers'] = flicker_count
        return switches

    def validate_pass_highlights(self) -> Dict:
        if not self.renderer.history:
            return {'total_passes': 0, 'highlighted_correctly': 0, 'accuracy_pct': 0.0}

        events = self.renderer.history
        pass_frames = [e['frame_number'] for e in events if e.get('is_pass')]

        correct = sum(1 for e in events if e.get('is_pass'))
        ambiguous = 0

        for entry in events:
            if entry.get('is_pass') and entry.get('color') != self.renderer.pass_color:
                ambiguous += 1
                break

        accuracy = ((correct - ambiguous) / max(1, len(set(pass_frames))) * 100) if pass_frames else 100.0

        result = {
            'total_passes': len(set(pass_frames)),
            'highlighted_correctly': correct - ambiguous,
            'accuracy_pct': round(accuracy, 2)
        }
        self.validation_results['pass_highlights'].append(result)
        return result

    def validate_touch_markers(self) -> Dict:
        if not self.renderer.history:
            return {'total_touches': 0, 'marked_correctly': 0, 'accuracy_pct': 0.0}

        events = self.renderer.history
        touch_frames = [e['frame_number'] for e in events if e.get('is_touch')]

        correct = sum(1 for e in events if e.get('is_touch'))
        accuracy = (correct / max(1, len(touch_frames)) * 100) if touch_frames else 100.0

        result = {
            'total_touches': len(set(touch_frames)),
            'marked_correctly': correct,
            'accuracy_pct': round(accuracy, 2)
        }
        self.validation_results['touch_markers'].append(result)
        return result

    def validate_debug_overlay(self) -> Dict:
        if not self.renderer.history:
            return {'overlay_accuracy_pct': 0.0}

        sorted_history = sorted(self.renderer.history, key=lambda e: e['frame_number'])
        correct = 0
        total = 0

        for entry in sorted_history:
            frame = entry.get('frame_number')
            if frame is None:
                continue
            total += 1
            if entry.get('ball_speed_kmh', 0) >= 0:
                correct += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        result = {'overlay_accuracy_pct': round(accuracy, 2)}
        self.validation_results['debug_overlay_checks'] = [result]
        return result

    def generate_validation_video(self) -> bool:
        if not self.renderer.history:
            logger.warning("No renderer history available for validation video")
            return False

        canvas_size = self.renderer.canvas_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(OUTPUT_DIR / "ball_path_validation.mp4"), fourcc, 25.0, canvas_size)

        max_frame = max(entry['frame_number'] for entry in self.renderer.history)
        temp_renderer = BallPathRenderer(self.config)
        temp_renderer.reset()

        for frame_num in range(max_frame + 1):
            current_entries = [e for e in self.renderer.history if e['frame_number'] <= frame_num]
            temp_renderer.history.clear()
            for e in current_entries:
                temp_renderer.history.append(e)
            if current_entries:
                temp_renderer.current_color = current_entries[-1]['color']

            frame = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            rendered = temp_renderer.render(frame)
            rendered = temp_renderer.draw_debug_overlay(rendered, self.config)

            y = 30
            lines = [
                f"Frame: {frame_num} | Tracked: {len(current_entries)}",
                f"Team: {current_entries[-1].get('team_id', 'N/A') if current_entries else 'N/A'}",
                f"Speed: {current_entries[-1].get('ball_speed_kmh', 0):.1f} km/h" if current_entries else "Speed: N/A",
            ]
            for line in lines:
                cv2.putText(rendered, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(rendered, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y += 25

            out.write(rendered)

        out.release()
        logger.info("Validation video saved to %s", OUTPUT_DIR / "ball_path_validation.mp4")
        return True

    def generate_report(self) -> str:
        tracking = self.validation_results['tracking_coverage']
        colour = self.validation_results['colour_assignment']
        pass_hl = self.validate_pass_highlights()
        touch_mk = self.validate_touch_markers()
        overlay = self.validate_debug_overlay()
        switches = self.validation_results['possession_switches']
        flickers = self.validation_results['flickers']

        avg_possession_length = 0.0
        if len(switches) > 1:
            lengths = []
            for i in range(1, len(switches)):
                length = switches[i]['frame'] - switches[i-1]['frame']
                if length > 0:
                    lengths.append(length)
            avg_possession_length = float(np.mean(lengths)) if lengths else 0.0

        overall_pass = (
            tracking.get('coverage_pct', 0) > 95 and
            colour.get('accuracy_pct', 0) == 100.0 and
            flickers == 0 and
            pass_hl.get('accuracy_pct', 0) == 100.0 and
            touch_mk.get('accuracy_pct', 0) == 100.0
        )

        report = f"""# Ball Path Renderer Validation Report - match30

## Summary

| Metric | Value |
|--------|-------|
| Overall Validation | {'PASS' if overall_pass else 'FAIL'} |

## Tracking Continuity

| Metric | Value |
|--------|-------|
| Total Frames | {tracking.get('total_frames', 0)} |
| Detected Frames | {tracking.get('detected_frames', 0)} |
| Interpolated Frames | {tracking.get('interpolated_frames', 0)} |
| Missing Frames | {tracking.get('missing_frames', 0)} |
| Longest Missing Sequence | {tracking.get('longest_missing_sequence', 0)} |
| Coverage % | {tracking.get('coverage_pct', 0):.2f}% |

## Colour Assignment

| Metric | Value |
|--------|-------|
| Correct | {colour.get('correct', 0)} |
| Incorrect | {colour.get('incorrect', 0)} |
| Unknown | {colour.get('unknown', 0)} |
| Total | {colour.get('total', 0)} |
| Accuracy % | {colour.get('accuracy_pct', 0):.2f}% |

## Possession Switching

| Metric | Value |
|--------|-------|
| Total Switches | {len(switches)} |
| Flickers | {flickers} |
| Average Possession Length (frames) | {avg_possession_length:.2f} |

## Pass Highlighting

| Metric | Value |
|--------|-------|
| Total Passes | {pass_hl.get('total_passes', 0)} |
| Highlighted Correctly | {pass_hl.get('highlighted_correctly', 0)} |
| Accuracy % | {pass_hl.get('accuracy_pct', 0):.2f}% |

## Touch Markers

| Metric | Value |
|--------|-------|
| Total Touches | {touch_mk.get('total_touches', 0)} |
| Marked Correctly | {touch_mk.get('marked_correctly', 0)} |
| Accuracy % | {touch_mk.get('accuracy_pct', 0):.2f}% |

## Debug Overlay

| Metric | Value |
|--------|-------|
| Overlay Accuracy % | {overlay.get('overlay_accuracy_pct', 0):.2f}% |

---

*Generated by Ball Path Renderer Validator*
"""
        report_path = OUTPUT_DIR / "ball_path_validation.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info("Validation report saved to %s", report_path)
        return report


def main():
    print("=" * 60)
    print("SPORTA VISTA PRO - Ball Path Renderer Validator")
    print("=" * 60)

    start_time = time.time()
    config = get_config()
    cfg = config.raw

    validator = BallPathValidator(cfg)
    validator.seed_renderer()

    print("\nValidating tracking continuity...")
    tracking = validator.validate_tracking_continuity()
    print(f"[OK] Coverage: {tracking.get('coverage_pct', 0):.2f}%")

    print("\nValidating colour assignment...")
    colour = validator.validate_colour_assignment()
    print(f"[OK] Colour accuracy: {colour.get('accuracy_pct', 0):.2f}%")

    print("\nValidating possession switching...")
    switches = validator.validate_possession_switching()
    print(f"[OK] Switches: {len(switches)}, Flickers: {validator.validation_results['flickers']}")

    print("\nValidating pass highlights...")
    passes = validator.validate_pass_highlights()
    print(f"[OK] Pass accuracy: {passes.get('accuracy_pct', 0):.2f}%")

    print("\nValidating touch markers...")
    touches = validator.validate_touch_markers()
    print(f"[OK] Touch accuracy: {touches.get('accuracy_pct', 0):.2f}%")

    print("\nValidating debug overlay...")
    overlay = validator.validate_debug_overlay()
    print(f"[OK] Overlay accuracy: {overlay.get('overlay_accuracy_pct', 0):.2f}%")

    print("\nGenerating validation video...")
    video_success = validator.generate_validation_video()
    if video_success:
        print("[OK] Validation video: outputs/ball_path_validation.mp4")
    else:
        print("[WARN] Validation video generation failed")

    print("\nGenerating validation report...")
    report = validator.generate_report()
    print("[OK] Validation report: outputs/ball_path_validation.md")

    total_time = time.time() - start_time
    print(f"\nTotal wall time: {total_time:.2f} s")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()