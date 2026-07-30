# visualize_teams.py
import cv2


class TeamVisualizer:

    def __init__(self):

        self.team_colors = {
            "Red": (0, 0, 255),         # Red in BGR
            "Blue": (255, 0, 0),        # Blue in BGR
            "Unknown": (0, 255, 255)    # Yellow
        }

    def draw_player(
        self,
        frame,
        bbox,
        track_id,
        team_name
    ):

        x1, y1, x2, y2 = bbox

        color = self.team_colors.get(
            team_name,
            (255,255,255)
        )

        # Reduce bounding box size by 10% on each side
        box_width = x2 - x1
        box_height = y2 - y1
        x1_new = x1 + int(box_width * 0.1)
        y1_new = y1 + int(box_height * 0.1)
        x2_new = x2 - int(box_width * 0.1)
        y2_new = y2 - int(box_height * 0.1)

        # Bounding Box (reduced size)
        cv2.rectangle(
            frame,
            (x1_new, y1_new),
            (x2_new, y2_new),
            color,
            2
        )

        # Background for text (use original y1 for text positioning)
        cv2.rectangle(
            frame,
            (x1, y1_new - 30),
            (x2, y1_new),
            color,
            -1
        )

        label = f"{team_name} | ID {track_id}"

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1_new - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        return frame