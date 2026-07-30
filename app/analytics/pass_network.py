"""
Pass Network & Tactical Intelligence Module

Constructs directed passing graphs, calculates average player spatial positions,
evaluates team tactical shape metrics (width, depth, compactness, defensive line height, center of mass),
classifies progressive passes and switches of play, and renders publication-quality 2D pitch pass network graphics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.homography.field_config import (
    PITCH_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    SCALE_X,
    SCALE_Y,
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS
)
from app.homography.visualize_pitch import PitchVisualizer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PassNetworkAnalyzer:
    """
    Analyzes pass network topology, spatial player average positions,
    team tactical shape, and progressive passing telemetry.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def compute_average_positions(
        self,
        player_histories: Dict[int, List[Tuple[float, float]]],
        team_assignments: Optional[Dict[int, Any]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculates average position, median position, movement standard deviation,
        and movement radius for every tracked player.
        """
        avg_positions = {}
        for pid, history in player_histories.items():
            if not history:
                continue

            arr = np.array(history)
            avg_x = float(np.mean(arr[:, 0]))
            avg_y = float(np.mean(arr[:, 1]))
            med_x = float(np.median(arr[:, 0]))
            med_y = float(np.median(arr[:, 1]))
            std_x = float(np.std(arr[:, 0]))
            std_y = float(np.std(arr[:, 1]))

            # Movement radius: average Euclidean distance from player's mean position
            dists = np.hypot(arr[:, 0] - avg_x, arr[:, 1] - avg_y)
            movement_radius = float(np.mean(dists))

            # Team name mapping
            team_id = team_assignments.get(pid) if team_assignments else None
            team_name = "Red" if str(team_id) == "0" or team_id == 0 else ("Blue" if str(team_id) == "1" or team_id == 1 else "Unknown")

            avg_positions[pid] = {
                "player_id": pid,
                "team": team_name,
                "average_position": [round(avg_x, 2), round(avg_y, 2)],
                "median_position": [round(med_x, 2), round(med_y, 2)],
                "std_dev": [round(std_x, 2), round(std_y, 2)],
                "movement_radius": round(movement_radius, 2),
                "total_samples": len(history)
            }

        return avg_positions

    def compute_team_shape(
        self,
        player_histories: Dict[int, List[Tuple[float, float]]],
        team_assignments: Optional[Dict[int, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes team tactical width, depth, compactness ratio,
        defensive line height, midfield line height, and center of mass.
        """
        avg_pos_dict = self.compute_average_positions(player_histories, team_assignments)

        team_players: Dict[str, List[Tuple[float, float]]] = {}
        for pid, data in avg_pos_dict.items():
            t = data["team"]
            if t not in team_players:
                team_players[t] = []
            team_players[t].append(tuple(data["average_position"]))

        team_shapes = {}
        for team_name, positions in team_players.items():
            if not positions:
                continue

            pos_arr = np.array(positions)
            xs = pos_arr[:, 0]
            ys = pos_arr[:, 1]

            width_m = float(np.max(ys) - np.min(ys))
            depth_m = float(np.max(xs) - np.min(xs))
            com_x = float(np.mean(xs))
            com_y = float(np.mean(ys))

            # Defensive line height: 20th percentile of X positions
            defensive_line = float(np.percentile(xs, 20))
            # Midfield line height: 50th percentile of X positions
            midfield_line = float(np.percentile(xs, 50))

            # Compactness: ratio of convex hull area or bounding box area relative to field
            bbox_area = width_m * depth_m
            pitch_area = FIELD_LENGTH_METERS * FIELD_WIDTH_METERS
            compactness = round(min(1.0, bbox_area / max(pitch_area, 1.0)), 2)

            team_shapes[team_name] = {
                "team": team_name,
                "player_count": len(positions),
                "width_m": round(width_m, 2),
                "depth_m": round(depth_m, 2),
                "compactness": compactness,
                "center_of_mass": [round(com_x, 2), round(com_y, 2)],
                "defensive_line_height_m": round(defensive_line, 2),
                "midfield_line_height_m": round(midfield_line, 2),
                "average_line_m": round(com_x, 2)
            }

        return team_shapes

    def analyze_pass_network(
        self,
        pass_events: List[Dict[str, Any]],
        player_histories: Dict[int, List[Tuple[float, float]]],
        team_assignments: Optional[Dict[int, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds directed pass graph and classifies progressive passes.
        """
        avg_positions = self.compute_average_positions(player_histories, team_assignments)
        team_shapes = self.compute_team_shape(player_histories, team_assignments)

        # Directed pass edges & touch counts
        pass_counts: Dict[Tuple[int, int], int] = {}
        successful_counts: Dict[Tuple[int, int], int] = {}
        player_touches: Dict[int, int] = {pid: 0 for pid in avg_positions.keys()}
        player_passes_attempted: Dict[int, int] = {pid: 0 for pid in avg_positions.keys()}
        player_passes_completed: Dict[int, int] = {pid: 0 for pid in avg_positions.keys()}
        player_progressive_passes: Dict[int, int] = {pid: 0 for pid in avg_positions.keys()}

        progressive_pass_events = []

        for p in pass_events:
            passer = p.get("passer")
            receiver = p.get("receiver")
            successful = p.get("successful", False)
            start_pos = p.get("start_position")
            end_pos = p.get("end_position")

            if passer is not None:
                player_passes_attempted[passer] = player_passes_attempted.get(passer, 0) + 1
                player_touches[passer] = player_touches.get(passer, 0) + 1

            if successful and passer is not None and receiver is not None:
                pair = (passer, receiver)
                pass_counts[pair] = pass_counts.get(pair, 0) + 1
                successful_counts[pair] = successful_counts.get(pair, 0) + 1
                player_passes_completed[passer] = player_passes_completed.get(passer, 0) + 1
                player_touches[receiver] = player_touches.get(receiver, 0) + 1

                # Progressive pass classification: dx >= 10m forward toward opponent end
                if start_pos and end_pos:
                    dx = end_pos[0] - start_pos[0]
                    dy = abs(end_pos[1] - start_pos[1])
                    is_progressive = (dx >= 10.0) or (start_pos[0] < 52.5 and dx >= 15.0)
                    is_switch = (dy >= 30.0)

                    if is_progressive or is_switch:
                        player_progressive_passes[passer] = player_progressive_passes.get(passer, 0) + 1
                        prog_dict = {**p, "is_progressive": is_progressive, "is_switch": is_switch}
                        progressive_pass_events.append(prog_dict)

        # Build graph JSON structure
        nodes = []
        for pid, pos_data in avg_positions.items():
            nodes.append({
                "id": pid,
                "label": f"#{pid}",
                "team": pos_data["team"],
                "average_position": pos_data["average_position"],
                "total_touches": player_touches.get(pid, 0),
                "passes_attempted": player_passes_attempted.get(pid, 0),
                "passes_completed": player_passes_completed.get(pid, 0),
                "progressive_passes": player_progressive_passes.get(pid, 0),
                "accuracy_pct": round((player_passes_completed.get(pid, 0) / max(player_passes_attempted.get(pid, 0), 1)) * 100.0, 1)
            })

        edges = []
        for (passer, receiver), count in pass_counts.items():
            edges.append({
                "passer": passer,
                "receiver": receiver,
                "pass_count": count,
                "weight": count
            })

        # Sort edges by weight descending
        edges.sort(key=lambda e: e["pass_count"], reverse=True)

        # Team Passing Summary
        team_summary = {}
        for team_name in ["Red", "Blue"]:
            t_nodes = [n for n in nodes if n["team"] == team_name]
            tot_att = sum(n["passes_attempted"] for n in t_nodes)
            tot_cmp = sum(n["passes_completed"] for n in t_nodes)
            tot_prog = sum(n["progressive_passes"] for n in t_nodes)

            team_summary[team_name] = {
                "team": team_name,
                "total_passes_attempted": tot_att,
                "completed_passes": tot_cmp,
                "completion_pct": round((tot_cmp / max(tot_att, 1)) * 100.0, 1) if tot_att > 0 else 0.0,
                "progressive_passes": tot_prog,
                "tactical_shape": team_shapes.get(team_name, {})
            }

        return {
            "nodes": nodes,
            "edges": edges,
            "average_positions": avg_positions,
            "team_shapes": team_shapes,
            "team_passing_summary": team_summary,
            "progressive_passes": progressive_pass_events
        }


class PassNetworkVisualizer:
    """Renders 2D tactical pass network diagrams overlaid on FIFA pitch canvas."""

    def __init__(self, pitch_visualizer: Optional[PitchVisualizer] = None):
        self.pitch_visualizer = pitch_visualizer or PitchVisualizer(width=PITCH_IMAGE_WIDTH, height=PITCH_IMAGE_HEIGHT)

    def render_pass_network(
        self,
        network_data: Dict[str, Any],
        team_filter: Optional[str] = None
    ) -> np.ndarray:
        """
        Renders directed pass graph on top-down FIFA pitch canvas.

        Args:
            network_data: Dict returned by PassNetworkAnalyzer.analyze_pass_network().
            team_filter: Optional "Red" or "Blue" to filter graph by team.

        Returns:
            Rendered BGR image canvas.
        """
        canvas = self.pitch_visualizer.base_pitch_image.copy()

        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])

        if team_filter:
            nodes = [n for n in nodes if n["team"] == team_filter]
            valid_node_ids = {n["id"] for n in nodes}
            edges = [e for e in edges if e["passer"] in valid_node_ids and e["receiver"] in valid_node_ids]

        node_dict = {n["id"]: n for n in nodes}

        # 1. Draw Edges (Arrows)
        max_edge_weight = max((e["pass_count"] for e in edges), default=1)

        for edge in edges:
            passer_id = edge["passer"]
            receiver_id = edge["receiver"]
            count = edge["pass_count"]

            p_node = node_dict.get(passer_id)
            r_node = node_dict.get(receiver_id)

            if p_node and r_node:
                px_m, py_m = p_node["average_position"]
                rx_m, ry_m = r_node["average_position"]

                px, py = int(round(px_m * SCALE_X)), int(round(py_m * SCALE_Y))
                rx, ry = int(round(rx_m * SCALE_X)), int(round(ry_m * SCALE_Y))

                # Edge thickness & color based on pass frequency
                thickness = max(1, int(round((count / max_edge_weight) * 5)))
                color = (0, 255, 0) if p_node["team"] == "Red" else (255, 180, 0)

                cv2.arrowedLine(canvas, (px, py), (rx, ry), color, thickness, cv2.LINE_AA, tipLength=0.15)

        # 2. Draw Nodes (Player Average Positions)
        max_touches = max((n["total_touches"] for n in nodes), default=1)
        max_touches = max(max_touches, 1)  # Guard against zero division

        for node in nodes:
            px_m, py_m = node["average_position"]
            px, py = int(round(px_m * SCALE_X)), int(round(py_m * SCALE_Y))
            touches = node["total_touches"]

            # Radius proportional to touch volume
            radius = max(10, int(round(12 + (touches / max_touches) * 16)))
            color = (0, 0, 255) if node["team"] == "Red" else (255, 0, 0)

            # Node circle & border
            cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), radius + 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Label player ID inside node
            lbl = f"#{node['id']}"
            cv2.putText(canvas, lbl, (px - 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Header Stats Banner
        title_text = f"TACTICAL PASS NETWORK - {team_filter.upper() if team_filter else 'ALL TEAMS'}"
        cv2.rectangle(canvas, (10, 10), (450, 45), (20, 20, 20), -1)
        cv2.rectangle(canvas, (10, 10), (450, 45), (0, 255, 255), 1)
        cv2.putText(canvas, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

        return canvas
