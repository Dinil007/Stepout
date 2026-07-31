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

        # Bounding Box
        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        # Background for text
        cv2.rectangle(
            frame,
            (x1,y1-30),
            (x2,y1),
            color,
            -1
        )

        label = f"{team_name} | ID {track_id}"

        cv2.putText(
            frame,
            label,
            (x1+5,y1-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            2
        )

        return frame