"""
Centralized Pitch ROI Loader

Single source of truth for loading pitch ROI coordinates.
Prioritizes configs/pitch_roi.json over config.yaml.
"""

import json
import yaml
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def load_pitch_roi(
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Tuple[List[List[int]], str]:
    """
    Load pitch ROI polygon from configs/pitch_roi.json or config.yaml.
    
    Args:
        project_root: Project root directory. If None, uses current working directory.
        verbose: Print ROI source and coordinates.
    
    Returns:
        Tuple of (roi_polygon, source) where:
            - roi_polygon: List of [x, y] coordinates
            - source: String indicating the source ("configs/pitch_roi.json" or "config.yaml")
    
    Raises:
        FileNotFoundError: If neither ROI source exists.
        ValueError: If ROI data is invalid.
    """
    if project_root is None:
        project_root = Path.cwd()
    
    # Try configs/pitch_roi.json first
    roi_json_path = project_root / "configs" / "pitch_roi.json"
    
    if roi_json_path.exists():
        try:
            with open(roi_json_path, 'r') as f:
                roi_data = json.load(f)
            
            points = roi_data.get("points")
            if not points or len(points) < 4:
                raise ValueError(f"Invalid ROI in {roi_json_path}: need at least 4 points")
            
            roi_polygon = [[int(x), int(y)] for x, y in points]
            
            if verbose:
                print(f"ROI Source: configs/pitch_roi.json")
                print(f"ROI Polygon: {roi_polygon}")
            
            return roi_polygon, "configs/pitch_roi.json"
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {roi_json_path}: {e}")
        except Exception as e:
            print(f"Warning: Failed to load {roi_json_path}: {e}")
    
    # Fall back to config.yaml
    config_path = project_root / "config.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            roi_polygon = config.get("pitch", {}).get("roi_polygon")
            
            if not roi_polygon or len(roi_polygon) < 4:
                raise ValueError(f"Invalid ROI in {config_path}: need at least 4 points")
            
            roi_polygon = [[int(x), int(y)] for x, y in roi_polygon]
            
            if verbose:
                print(f"ROI Source: config.yaml")
                print(f"ROI Polygon: {roi_polygon}")
            
            return roi_polygon, "config.yaml"
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load ROI from {config_path}: {e}")
    
    # Neither source exists
    raise FileNotFoundError(
        "No ROI configuration found. Please either:\n"
        f"  1. Create configs/pitch_roi.json using scripts/create_pitch_roi.py\n"
        f"  2. Add roi_polygon to pitch section in config.yaml\n"
        f"Searched:\n"
        f"  - {roi_json_path}\n"
        f"  - {config_path}"
    )


def load_pitch_roi_as_numpy(
    project_root: Optional[Path] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Load pitch ROI as numpy array for OpenCV operations.
    
    Args:
        project_root: Project root directory. If None, uses current working directory.
        verbose: Print ROI source and coordinates.
    
    Returns:
        Tuple of (roi_polygon_np, source) where:
            - roi_polygon_np: NumPy array of shape (N, 2) with dtype int32
            - source: String indicating the source
    """
    roi_polygon, source = load_pitch_roi(project_root, verbose)
    return np.array(roi_polygon, dtype=np.int32), source
