import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add root directory to system path for flexible execution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app.team_classification.color_extractor import ColorExtractor
from app.team_classification.team_classifier import TeamClassifier
from app.team_classification.visualize_teams import TeamVisualizer


class FootballTeamPipeline:
    """
    Production-grade pipeline for football video analysis:
    Integrates YOLOv8 object detection, ByteTrack tracking, Pitch ROI filtering,
    Jersey Color Extraction, KMeans Team Classification, and Team Visualizations.
    """

    def __init__(
        self,
        input_video_path: str = "outputs/preprocessed/preprocessed_video.mp4",
        output_video_path: str = "outputs/team_classification/team_classification.mp4",
        model_weights: str = "yolov8x.pt",
        tracker_config: str = "app/tracking/bytetrack_custom.yaml",
        warmup_frames: int = 1,
        max_frames: int = 1000,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.5,
        imgsz: int = 1280,
        pitch_polygon: np.ndarray = None
    ):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model_weights = model_weights
        self.tracker_config = tracker_config
        self.warmup_frames = warmup_frames
        self.max_frames = max_frames
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        # Pitch Region of Interest (ROI) polygon to exclude crowd/spectators
        if pitch_polygon is None:
            self.pitch_polygon = np.array([
                [8, 347],
                [1218, 328],
                [1250, 529],
                [54, 610]
            ], dtype=np.int32)
        else:
            self.pitch_polygon = pitch_polygon

        # Component Initializations
        self.model = None
        self.color_extractor = ColorExtractor(jersey_ratio=0.5)
        self.team_classifier = TeamClassifier()
        self.team_visualizer = TeamVisualizer()

        # State tracking
        self.collected_colors = []
        self.track_color_samples = {}  # track_id -> list of HSV colors
        self.is_classifier_trained = False

    def initialize_model(self):
        """Load YOLO model weights safely."""
        print(f"[INFO] Loading YOLO model: {self.model_weights}...")
        try:
            self.model = YOLO(self.model_weights)
            print("[INFO] YOLO model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_weights}': {e}")

    def is_inside_pitch(self, bbox: tuple) -> bool:
        """
        Check whether the bounding box center is within the defined pitch ROI.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        inside = cv2.pointPolygonTest(self.pitch_polygon, (center_x, center_y), False)
        return inside >= 0

    def collect_color_sample(self, frame: np.ndarray, bbox: tuple, track_id: int):
        """
        Extract jersey color for a player detection and buffer for initial cluster training.
        """
        color = self.color_extractor.get_player_color(frame, bbox)
        if color is not None:
            self.collected_colors.append(color)
            if track_id not in self.track_color_samples:
                self.track_color_samples[track_id] = []
            self.track_color_samples[track_id].append(color)

    def train_team_classifier(self):
        """
        Train KMeans TeamClassifier using accumulated jersey color samples across warm-up frames.
        """
        if len(self.collected_colors) < 2:
            print("[WARNING] Insufficient color samples collected to train TeamClassifier.")
            return

        print(f"\n[INFO] Training TeamClassifier on {len(self.collected_colors)} color samples...")
        self.team_classifier.fit(self.collected_colors)
        self.is_classifier_trained = True

        # Pre-assign team labels for track IDs seen during warm-up using averaged/median colors
        for track_id, colors in self.track_color_samples.items():
            if colors:
                avg_color = np.mean(colors, axis=0)
                self.team_classifier.assign_player(track_id, avg_color)

        print("[INFO] TeamClassifier trained successfully. Player team mappings initialized.")

    def process_video(self):
        """
        Main execution loop for video processing, tracking, team classification, and rendering.
        """
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Input video file not found at: {self.input_video_path}")

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video source: {self.input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        self.initialize_model()

        frame_count = 0
        print(f"[INFO] Starting Team Classification Pipeline (Max Frames: {self.max_frames})...\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= self.max_frames:
                    break

                frame_count += 1

                # 1. Run YOLO + ByteTrack
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    classes=[0],  # Class 0: Person / Player
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.imgsz,
                    verbose=False
                )

                annotated_frame = frame.copy()

                player_detections = []

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2, y2)

                        # Filter out non-pitch detections (spectators, technical area)
                        if not self.is_inside_pitch(bbox):
                            continue

                        track_id = int(box.id[0]) if box.id is not None else -1
                        player_detections.append((bbox, track_id))

                # 2. Phase 1: Warm-up Color Sample Collection
                if frame_count <= self.warmup_frames:
                    for bbox, track_id in player_detections:
                        if track_id != -1:
                            self.collect_color_sample(frame, bbox, track_id)

                    # Trigger training at the end of warm-up phase
                    if frame_count == self.warmup_frames:
                        self.train_team_classifier()

                # 3. Phase 2: Online Prediction & Visualization
                for bbox, track_id in player_detections:
                    team_name = "Unknown"

                    if self.is_classifier_trained and track_id != -1:
                        # Check if player already has an assigned team
                        if track_id in self.team_classifier.player_teams:
                            team_label = self.team_classifier.player_teams[track_id]
                        else:
                            # Extract color on the fly for new track ID and assign team
                            color = self.color_extractor.get_player_color(frame, bbox)
                            team_label = self.team_classifier.assign_player(track_id, color)

                        team_name = self.team_classifier.get_team_name(team_label)

                    # Render bounding box & team label
                    annotated_frame = self.team_visualizer.draw_player(
                        annotated_frame,
                        bbox,
                        track_id,
                        team_name
                    )

                writer.write(annotated_frame)
                print(f"Processed Frame : {frame_count}/{self.max_frames}", end="\r")

        except Exception as e:
            print(f"\n[ERROR] Pipeline interrupted by exception: {e}")
            raise e
        finally:
            cap.release()
            writer.release()

        print("\n" + "=" * 50)
        print("Team Classification Pipeline Completed Successfully")
        print(f"Total Frames Processed : {frame_count}")
        print(f"Annotated Output Video : {self.output_video_path}")
        print("=" * 50)


if __name__ == "__main__":
    pipeline = FootballTeamPipeline(
        input_video_path="outputs/preprocessed/preprocessed_video.mp4",
        output_video_path="outputs/team_classification/team_classification.mp4",
        warmup_frames=1,
        max_frames=1000
    )
    pipeline.process_video()
