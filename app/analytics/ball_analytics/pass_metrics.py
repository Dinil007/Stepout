"""
Pass Metrics Module

Computes pass network metrics and passing statistics.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PassMetricsCalculator:
    """
    Computes pass network and passing performance metrics.
    """

    def compute_network_metrics(self, passes: List[Dict]) -> Dict:
        """
        Computes pass network metrics.

        Args:
            passes: List of pass event dicts.

        Returns:
            Network metrics dict.
        """
        if not passes:
            return {'total_passes': 0, 'unique_passers': 0, 'unique_receivers': 0}

        passers = [p['passer_id'] for p in passes]
        receivers = [p['receiver_id'] for p in passes if p.get('receiver_id') is not None]

        # Build adjacency
        pass_matrix = {}
        for p in passes:
            key = (p['passer_id'], p['receiver_id'])
            pass_matrix[key] = pass_matrix.get(key, 0) + 1

        return {
            'total_passes': len(passes),
            'unique_passers': len(set(passers)),
            'unique_receivers': len(set(receivers)),
            'pass_matrix': pass_matrix
        }

    def compute_team_metrics(self, passes: List[Dict], team_id: str) -> Dict:
        """
        Computes passing metrics for a specific team.

        Args:
            passes: List of pass event dicts.
            team_id: Team identifier.

        Returns:
            Team passing metrics dict.
        """
        team_passes = [p for p in passes if p.get('passer_team_id') == team_id or p.get('receiver_team_id') == team_id]
        successful = [p for p in team_passes if p.get('successful', False)]
        total_distance = sum(p['distance_m'] for p in team_passes)

        return {
            'team_id': team_id,
            'total_passes': len(team_passes),
            'successful_passes': len(successful),
            'accuracy_pct': round(len(successful) / len(team_passes) * 100, 2) if team_passes else 0.0,
            'total_distance_m': round(total_distance, 2),
            'average_distance_m': round(total_distance / len(team_passes), 2) if team_passes else 0.0
        }

    def get_summary(self, passes: List[Dict]) -> Dict:
        """Returns aggregate passing summary."""
        if not passes:
            return {'total_passes': 0, 'successful_passes': 0, 'accuracy_pct': 0.0}

        successful = [p for p in passes if p.get('successful', False)]
        total_distance = sum(p['distance_m'] for p in passes)

        return {
            'total_passes': len(passes),
            'successful_passes': len(successful),
            'accuracy_pct': round(len(successful) / len(passes) * 100, 2) if passes else 0.0,
            'total_distance_m': round(total_distance, 2),
            'average_distance_m': round(total_distance / len(passes), 2) if passes else 0.0,
            'max_speed_kmh': round(max(p['max_speed_kmh'] for p in passes), 2) if passes else 0.0
        }