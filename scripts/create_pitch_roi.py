"""
Interactive Pitch ROI Calibration Tool

Creates a pitch Region of Interest (ROI) polygon for filtering detections.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import yaml


class PitchROICalibrator:
    """Interactive ROI calibration tool for pitch area selection."""
    
    def __init__(self, video_path, project_root):
        self.video_path = Path(video_path)
        self.project_root = Path(project_root)
        self.points = []
        self.frame = None
        self.display_frame = None
        self.window_name = "Pitch ROI Calibration"
        
    def load_first_frame(self):
        """Load the frame at 14 seconds from the video."""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        # Get video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # Default fallback
        
        # Compute target frame (14 seconds)
        target_timestamp = 14.0
        target_frame = int(target_timestamp * fps)
        
        # Seek to target frame
        seek_success = cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError("Failed to read frame at 14 seconds")
        
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        
        # Print video information
        print(f"Video FPS: {fps:.2f}")
        print(f"Target timestamp: {target_timestamp} seconds")
        print(f"Target frame number: {target_frame}")
        print(f"Seek successful: {seek_success}")
        print(f"Loaded frame: {frame.shape}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI point manipulation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append([x, y])
            self.update_display()
            print(f"Added point {len(self.points)}: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove nearest point
            if self.points:
                distances = [np.hypot(x - px, y - py) for px, py in self.points]
                min_idx = np.argmin(distances)
                removed = self.points.pop(min_idx)
                self.update_display()
                print(f"Removed point {min_idx + 1}: {removed}")
    
    def update_display(self):
        """Update the display frame with current ROI visualization."""
        self.display_frame = self.frame.copy()
        
        if len(self.points) > 0:
            # Draw points
            for i, (px, py) in enumerate(self.points):
                cv2.circle(self.display_frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(
                    self.display_frame,
                    str(i + 1),
                    (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            # Draw lines between consecutive points
            if len(self.points) > 1:
                pts = np.array(self.points, np.int32)
                cv2.polylines(
                    self.display_frame,
                    [pts],
                    False,
                    (0, 255, 0),
                    2
                )
            
            # Close polygon if at least 4 points
            if len(self.points) >= 4:
                cv2.polylines(
                    self.display_frame,
                    [pts],
                    True,
                    (0, 255, 0),
                    2
                )
        
        # Draw instructions
        self.draw_instructions()
        
        cv2.imshow(self.window_name, self.display_frame)
    
    def draw_instructions(self):
        """Draw on-screen instructions."""
        instructions = [
            "Left Click: Add point",
            "Right Click: Remove nearest point",
            "ENTER: Save ROI",
            "BACKSPACE: Remove last point",
            "R: Reset all points",
            "ESC: Exit without saving"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                self.display_frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                self.display_frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
            y_offset += 25
        
        # Show point count
        point_text = f"Points: {len(self.points)}/4+"
        cv2.putText(
            self.display_frame,
            point_text,
            (10, self.display_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if len(self.points) >= 4 else (0, 0, 255),
            2
        )
    
    def save_roi(self):
        """Save the ROI polygon to JSON file."""
        if len(self.points) < 4:
            print("ERROR: Need at least 4 points to save ROI")
            return False
        
        # Ensure configs directory exists
        configs_dir = self.project_root / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        roi_data = {
            "video": self.video_path.name,
            "points": self.points
        }
        
        output_path = configs_dir / "pitch_roi.json"
        with open(output_path, 'w') as f:
            json.dump(roi_data, f, indent=2)
        
        print(f"\nROI saved to: {output_path}")
        print(f"Points: {self.points}")
        
        # Show success message
        success_frame = self.display_frame.copy()
        cv2.putText(
            success_frame,
            "ROI SAVED SUCCESSFULLY",
            (success_frame.shape[1] // 2 - 200, success_frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )
        cv2.imshow(self.window_name, success_frame)
        cv2.waitKey(2000)
        
        return True
    
    def reset_points(self):
        """Reset all ROI points."""
        self.points = []
        self.update_display()
        print("Reset all points")
    
    def remove_last_point(self):
        """Remove the last added point."""
        if self.points:
            removed = self.points.pop()
            self.update_display()
            print(f"Removed last point: {removed}")
    
    def run(self):
        """Run the interactive calibration tool."""
        self.load_first_frame()
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        
        print("\n" + "=" * 60)
        print("PITCH ROI CALIBRATION")
        print("=" * 60)
        print("Controls:")
        print("  Left Click    - Add ROI point")
        print("  Right Click   - Remove nearest point")
        print("  ENTER         - Save ROI")
        print("  BACKSPACE     - Remove last point")
        print("  R             - Reset all points")
        print("  ESC           - Exit without saving")
        print("=" * 60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("\nExiting without saving")
                break
                
            elif key == 13:  # ENTER
                if self.save_roi():
                    break
                    
            elif key == 8:  # BACKSPACE
                self.remove_last_point()
                
            elif key == ord('r') or key == ord('R'):
                self.reset_points()
        
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pitch ROI Calibration Tool")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (overrides config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Determine project root from script location
    project_root = Path(__file__).resolve().parent.parent
    
    # Get video path
    if args.video:
        video_path = Path(args.video)
        print(f"Video path (from --video): {video_path}")
    else:
        # Read from config.yaml
        config_path = project_root / "config.yaml"
        print(f"Loading config: {config_path}")
        
        if not config_path.exists():
            print(f"ERROR: Config file not found:\n{config_path}")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        video_input = config.get("video", {})
        if "input_path" not in video_input:
            print("ERROR: video.input_path not found in config.yaml")
            return
        
        video_path = Path(video_input["input_path"])
        print(f"Video path (from config): {video_path}")
    
    if not video_path.exists():
        print(f"ERROR: Video file not found:\n{video_path}")
        return
    
    # Run calibrator
    calibrator = PitchROICalibrator(video_path, project_root)
    calibrator.run()


if __name__ == "__main__":
    main()
