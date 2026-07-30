"""
Team Tactical Analytics Engine

Computes team-level tactical metrics including:
- Heatmaps (player, team, ball)
- Pass networks
- Team shape
- Possession stats
- Territory control
- Pressing metrics (PPDA)

All coordinates in meters. All percentages 0-100.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.homography.field_config import (
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS,
    PITCH_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
)

logger = logging.getLogger(__name__)


class TacticalAnalyzer:
    """Compute team tactical metrics from tracking data."""

    def __init__(
        self,
        fps: float = 30.0,
        bin_size: float = 1.0,
        output_dir: Optional[str] = None,
    ):
        self.fps = fps
        self.bin_size = bin_size
        self.output_dir = output_dir

        # Data store: list of per-frame snapshots
        self.frames: List[Dict[str, Any]] = []

        # Pass events
        self.pass_events: List[Dict[str, Any]] = []

        # Defensive actions for PPDA
        self.defensive_actions: List[Dict[str, Any]] = []

    def add_frame(
        self,
        frame_number: int,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
        team_assignments: Dict[int, Any],
        possessor_id: Optional[int] = None,
    ) -> None:
        """Record one frame state."""
        self.frames.append({
            "frame": frame_number,
            "players": players,
            "ball": ball,
            "team_assignments": team_assignments,
            "possessor_id": possessor_id,
        })

    def add_pass_event(self, event: Dict[str, Any]) -> None:
        self.pass_events.append(event)

    def add_defensive_action(self, action: Dict[str, Any]) -> None:
        self.defensive_actions.append(action)

    # ------------------------------------------------------------------
    # 1. Heatmaps
    # ------------------------------------------------------------------
    def compute_heatmaps(self) -> Dict[str, Any]:
        """Compute player, team, and ball heatmaps."""
        player_maps: Dict[int, np.ndarray] = {}
        team_maps: Dict[Any, np.ndarray] = {}
        ball_map = np.zeros(
            (int(FIELD_WIDTH_METERS / self.bin_size) + 1,
             int(FIELD_LENGTH_METERS / self.bin_size) + 1),
            dtype=np.float32,
        )

        for fr in self.frames:
            for p in fr.get("players", []):
                tid = p.get("track_id")
                team = fr.get("team_assignments", {}).get(tid)
                pos = p.get("field_position")
                if not pos:
                    continue
                x, y = pos
                ix = int(x / self.bin_size)
                iy = int(y / self.bin_size)
                player_maps.setdefault(tid, np.zeros_like(ball_map)).flat[
                    np.ravel_multi_index(
                        (min(iy, ball_map.shape[0] - 1),
                         min(ix, ball_map.shape[1] - 1)),
                        ball_map.shape
                    )
                ] += 1
                if team is not None:
                    team_maps.setdefault(team, np.zeros_like(ball_map)).flat[
                        np.ravel_multi_index(
                            (min(iy, ball_map.shape[0] - 1),
                             min(ix, ball_map.shape[1] - 1)),
                            ball_map.shape
                        )
                    ] += 1
            ball = fr.get("ball")
            if ball and ball.get("field_position"):
                bx, by = ball["field_position"]
                bix = int(bx / self.bin_size)
                biy = int(by / self.bin_size)
                ball_map[
                    min(biy, ball_map.shape[0] - 1),
                    min(bix, ball_map.shape[1] - 1),
                ] += 1

        # Attacking/defensive third occupancy (assume left→right attacking direction)
        third = FIELD_LENGTH_METERS / 3
        attacking_third_occupancy = 0
        defensive_third_occupancy = 0
        total_positions = 0
        for team_id, tmap in team_maps.items():
            total = float(tmap.sum())
            if total <= 0:
                continue
            # attacking third: x in [0, third]
            att = float(tmap[:, : int(third / self.bin_size)].sum())
            # defensive third: x in [2*third, FIELD_LENGTH_METERS]
            ddef = float(tmap[:, int(2 * third / self.bin_size):].sum())
            attacking_third_occupancy += att
            defensive_third_occupancy += ddef
            total_positions += total

        if total_positions > 0:
            attacking_third_occupancy = attacking_third_occupancy / total_positions * 100
            defensive_third_occupancy = defensive_third_occupancy / total_positions * 100

        return {
            "player_heatmaps": {str(k): v.tolist() for k, v in player_maps.items()},
            "team_heatmaps": {str(k): v.tolist() for k, v in team_maps.items()},
            "ball_heatmap": ball_map.tolist(),
            "attacking_third_occupancy_pct": round(float(attacking_third_occupancy), 2),
            "defensive_third_occupancy_pct": round(float(defensive_third_occupancy), 2),
        }

    # ------------------------------------------------------------------
    # 2. Pass Network
    # ------------------------------------------------------------------
    def compute_pass_network(self) -> Dict[str, Any]:
        """Build player-to-player passing graph."""
        pass_counts: Dict[Tuple[int, int], int] = {}
        avg_positions: Dict[int, List[Tuple[float, float]]] = {}

        for fr in self.frames:
            for p in fr.get("players", []):
                tid = p.get("track_id")
                pos = p.get("field_position")
                if tid is not None and pos:
                    avg_positions.setdefault(tid, []).append(pos)

        for ev in self.pass_events:
            from_id = ev.get("from_player")
            to_id = ev.get("to_player")
            if from_id is None or to_id is None:
                continue
            key = (int(from_id), int(to_id))
            pass_counts[key] = pass_counts.get(key, 0) + 1

        # Most connected players
        connection_count: Dict[int, int] = {}
        for (a, b), cnt in pass_counts.items():
            connection_count[a] = connection_count.get(a, 0) + cnt
            connection_count[b] = connection_count.get(b, 0) + cnt

        # Filter out None values before sorting
        valid_connections = {k: v for k, v in connection_count.items() if v is not None}
        most_connected = sorted(valid_connections.items(), key=lambda x: x[1], reverse=True)[:5]

        # Build nodes and edges
        nodes = []
        for tid, positions in avg_positions.items():
            if not positions:
                continue
            ax = sum(p[0] for p in positions) / len(positions)
            ay = sum(p[1] for p in positions) / len(positions)
            nodes.append({
                "track_id": tid,
                "avg_x": round(ax, 2),
                "avg_y": round(ay, 2),
                "passes_received": sum(v for (a, b), v in pass_counts.items() if b == tid),
                "passes_made": sum(v for (a, b), v in pass_counts.items() if a == tid),
            })

        edges = [
            {"from": a, "to": b, "count": cnt}
            for (a, b), cnt in pass_counts.items()
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "total_passes": len(self.pass_events),
            "most_connected_players": [{"track_id": t, "connections": c} for t, c in most_connected],
        }

    # ------------------------------------------------------------------
    # 3. Team Shape
    # ------------------------------------------------------------------
    def compute_team_shape(self) -> Dict[str, Any]:
        """Compute team shape metrics per frame and average."""
        team_frames: Dict[Any, List[Dict[str, float]]] = {}

        for fr in self.frames:
            players = fr.get("players", [])
            teams: Dict[Any, List[Tuple[float, float]]] = {}
            for p in players:
                tid = p.get("track_id")
                team = fr.get("team_assignments", {}).get(tid)
                pos = p.get("field_position")
                if team is not None and pos:
                    teams.setdefault(team, []).append(pos)

            for team_id, positions in teams.items():
                if len(positions) < 2:
                    continue
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                width = max(xs) - min(xs)
                length = max(ys) - min(ys)
                # compactness: average distance to centroid
                compactness = sum(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in positions) / len(positions)
                team_frames.setdefault(team_id, []).append({
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "width": width,
                    "length": length,
                    "compactness": compactness,
                })

        avg_shape: Dict[str, Any] = {}
        for team_id, frames in team_frames.items():
            avg_shape[str(team_id)] = {
                "avg_centroid_x": round(sum(f["centroid_x"] for f in frames) / len(frames), 2),
                "avg_centroid_y": round(sum(f["centroid_y"] for f in frames) / len(frames), 2),
                "avg_width_m": round(sum(f["width"] for f in frames) / len(frames), 2),
                "avg_length_m": round(sum(f["length"] for f in frames) / len(frames), 2),
                "avg_compactness_m": round(sum(f["compactness"] for f in frames) / len(frames), 2),
            }
        return avg_shape

    # ------------------------------------------------------------------
    # 4. Possession
    # ------------------------------------------------------------------
    def compute_possession(self) -> Dict[str, Any]:
        """Compute possession statistics."""
        team_possession_frames: Dict[Any, int] = {}
        total_possessor_frames = 0
        current_possessor = None
        current_chain_length = 0
        max_chain = 0
        chains: List[int] = []

        for fr in self.frames:
            poss = fr.get("possessor_id")
            team = fr.get("team_assignments", {}).get(poss) if poss is not None else None
            if team is not None:
                team_possession_frames[team] = team_possession_frames.get(team, 0) + 1
                total_possessor_frames += 1
                if team == current_possessor:
                    current_chain_length += 1
                else:
                    if current_possessor is not None:
                        chains.append(current_chain_length)
                        max_chain = max(max_chain, current_chain_length)
                    current_possessor = team
                    current_chain_length = 1

        if current_possessor is not None and current_chain_length > 0:
            chains.append(current_chain_length)
            max_chain = max(max_chain, current_chain_length)

        total_frames = len(self.frames) if self.frames else 1
        pct = {
            str(team): round((frames / total_possessor_frames) * 100, 2) if total_possessor_frames > 0 else 0.0
            for team, frames in team_possession_frames.items()
        }
        avg_chain = round(sum(chains) / len(chains), 2) if chains else 0.0

        return {
            "possession_pct": pct,
            "total_possession_frames": total_possessor_frames,
            "avg_possession_duration_frames": avg_chain,
            "longest_possession_chain_frames": max_chain,
            "num_possession_chains": len(chains),
        }

    # ------------------------------------------------------------------
    # 5. Territory Control
    # ------------------------------------------------------------------
    def compute_territory(self) -> Dict[str, Any]:
        """Compute territory control metrics."""
        third = FIELD_LENGTH_METERS / 3
        team_touches: Dict[Any, Dict[str, int]] = {}
        total_touches = 0

        for fr in self.frames:
            for p in fr.get("players", []):
                team = fr.get("team_assignments", {}).get(p.get("track_id"))
                pos = p.get("field_position")
                if team is None or not pos:
                    continue
                x, y = pos
                zone = "defensive_third" if x > 2 * third else ("attacking_third" if x < third else "middle_third")
                in_penalty = 0 <= y <= FIELD_WIDTH_METERS and x <= 18.0  # approximate penalty box
                d = team_touches.setdefault(team, {
                    "attacking_third": 0,
                    "middle_third": 0,
                    "defensive_third": 0,
                    "penalty_area": 0,
                    "final_third_entries": 0,
                    "progressive_carries": 0,
                })
                d[zone] += 1
                if in_penalty:
                    d["penalty_area"] += 1
                total_touches += 1

        # Compute final-third entries: a player enters attacking third from middle/defensive
        prev_zone: Dict[int, str] = {}
        for fr in self.frames:
            for p in fr.get("players", []):
                tid = p.get("track_id")
                team = fr.get("team_assignments", {}).get(tid)
                pos = p.get("field_position")
                if team is None or not pos:
                    continue
                x = pos[0]
                zone = "attacking_third" if x < third else ("defensive_third" if x > 2 * third else "middle_third")
                prev = prev_zone.get(tid)
                if prev != "attacking_third" and zone == "attacking_third":
                    team_touches[team]["final_third_entries"] += 1
                prev_zone[tid] = zone

        if total_touches > 0:
            for team_id, d in team_touches.items():
                for k in ["attacking_third", "middle_third", "defensive_third", "penalty_area"]:
                    d[f"{k}_pct"] = round(d[k] / total_touches * 100, 2)

        return dict(team_touches)

    # ------------------------------------------------------------------
    # 6. Pressing Metrics (PPDA)
    # ------------------------------------------------------------------
    def compute_pressing(self) -> Dict[str, Any]:
        """Compute PPDA and pressing events."""
        ppda: Dict[Any, Dict[str, int]] = {}
        for act in self.defensive_actions:
            team = act.get("team")
            if team is None:
                continue
            ppda.setdefault(team, {"defensive_actions": 0, "opponent_passes": 0})

        # Count opponent passes in final third when defending team has players
        for ev in self.pass_events:
            frame = ev.get("frame")
            from_team = ev.get("from_team")
            to_team = ev.get("to_team")
            if from_team is None or to_team is None or from_team == to_team:
                continue
            # Find frame context
            fr = next((f for f in self.frames if f["frame"] == frame), None)
            if not fr:
                continue
            # Check if pass occurred in final third
            ball = fr.get("ball")
            if not ball or not ball.get("field_position"):
                continue
            bx, _ = ball["field_position"]
            if bx < FIELD_LENGTH_METERS / 3:  # final third
                ppda.setdefault(from_team, {"defensive_actions": 0, "opponent_passes": 0})
                ppda[from_team]["opponent_passes"] += 1

        for act in self.defensive_actions:
            team = act.get("team")
            if team is not None and team in ppda:
                ppda[team]["defensive_actions"] += 1

        ppda_scores = {}
        for team, d in ppda.items():
            opp = d["opponent_passes"]
            ppda_scores[str(team)] = round(d["defensive_actions"] / opp, 2) if opp > 0 else 0.0

        return {
            "ppda": ppda_scores,
            "details": {str(k): v for k, v in ppda.items()},
        }

    # ------------------------------------------------------------------
    # Master export
    # ------------------------------------------------------------------
    def compute_all(self) -> Dict[str, Any]:
        logger.info("Computing tactical analytics...")
        heatmaps = self.compute_heatmaps()
        pass_network = self.compute_pass_network()
        team_shape = self.compute_team_shape()
        possession = self.compute_possession()
        territory = self.compute_territory()
        pressing = self.compute_pressing()

        result = {
            "team_heatmap": heatmaps.get("team_heatmaps", {}),
            "player_heatmaps": heatmaps.get("player_heatmaps", {}),
            "pass_network": pass_network,
            "team_shape": team_shape,
            "possession_summary": possession,
            "territory_control": territory,
            "pressing_metrics": pressing,
        }
        logger.info("Tactical analytics computation complete.")
        return result