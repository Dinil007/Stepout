# color_extractor.py
import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColorExtractor:
    """
    Extract the dominant jersey color from a player's bounding box.
    """

    def __init__(self, jersey_ratio=0.5):
        # Use the upper 50% of the player crop (jersey area)
        self.jersey_ratio = jersey_ratio

    def extract_player_crop(self, frame, bbox):
        """
        Crop the player from the frame.

        Args:
            frame: OpenCV image
            bbox: (x1, y1, x2, y2)

        Returns:
            Cropped player image
        """

        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return frame[y1:y2, x1:x2]

    def extract_jersey(self, player_crop):
        """
        Extract only the upper body (jersey).

        Returns:
            Jersey crop
        """

        if player_crop.size == 0:
            return None

        height = player_crop.shape[0]

        jersey_height = int(height * self.jersey_ratio)

        jersey = player_crop[:jersey_height, :]

        return jersey

    def dominant_color(self, jersey_crop, clusters=2):
        """
        Find the dominant HSV color of the jersey.

        Returns:
            HSV dominant color
        """

        if jersey_crop is None or jersey_crop.size == 0:
            return None

        hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape((-1, 3))

        kmeans = KMeans(
            n_clusters=clusters,
            random_state=42,
            n_init=10
        )

        kmeans.fit(pixels)

        labels = kmeans.labels_

        counts = np.bincount(labels)

        dominant = kmeans.cluster_centers_[np.argmax(counts)]

        return dominant.astype(int)

    def get_player_color(self, frame, bbox):
        """
        Complete pipeline.

        Input:
            Frame
            Bounding Box

        Output:
            Dominant HSV jersey color
        """

        player = self.extract_player_crop(frame, bbox)

        jersey = self.extract_jersey(player)

        color = self.dominant_color(jersey)

        return color