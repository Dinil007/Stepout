from sklearn.cluster import KMeans
import numpy as np
from collections import deque


class TeamClassifier:
    """
    Classify players into two teams based on jersey colors.
    """

    def __init__(self, history_len=30):
        self.model = None
        # IMPORTANT: Match these colors to your actual video jerseys
        # label 0 = Red team, label 1 = Blue team
        self.team_names = {
            0: "Red",
            1: "Blue"
        }
        self.player_teams = {}  # track_id -> team label
        self.player_color_history = {}  # track_id -> deque of recent colors
        self.history_len = history_len
        # Manual overrides for problematic tracks (track_id -> team_label)
        self.manual_overrides = {}

    def _update_color_history(self, track_id, color):
        """Track recent colors for majority-vote fallback."""
        if track_id not in self.player_color_history:
            self.player_color_history[track_id] = deque(maxlen=self.history_len)
        if color is not None:
            self.player_color_history[track_id].append(color)

    def _majority_vote_team(self, track_id):
        """Determine team by majority vote over recent valid colors."""
        history = self.player_color_history.get(track_id, deque())
        if not history:
            return None
        votes = []
        for c in history:
            pred = self.predict(c)
            if pred is not None:
                votes.append(pred)
        if not votes:
            return None
        return int(np.bincount(votes).argmax())

    def fit(self, colors):
        """
        Learn the two dominant team colors.

        Args:
            colors: List of HSV colors
        """

        if len(colors) < 2:
            return

        self.model = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=10
        )

        self.model.fit(colors)

    def predict(self, color):
        """
        Predict the team of a jersey color.

        Args:
            color: HSV color

        Returns:
            0 or 1, or None if unavailable
        """

        if self.model is None or color is None:
            return None

        color = np.array(color).reshape(1, -1)

        return int(self.model.predict(color)[0])

    def set_manual_override(self, track_id, team_label):
        """
        Set a manual team override for a specific track.
        
        Args:
            track_id: The tracking ID
            team_label: 0 or 1 (Team A or Team B)
        """
        self.manual_overrides[track_id] = team_label
        if team_label is not None:
            self.player_teams[track_id] = team_label

    def assign_player(self, track_id, color):
        """
        Assign a team to a player with robust fallback logic.

        Priority:
        1. Manual override for this track_id
        2. Existing assignment for this track_id
        3. Current color cluster prediction
        4. Majority vote over recent frames
        5. None (caller should handle Unknown)
        """

        # 1. Check for manual override
        if track_id in self.manual_overrides:
            return self.manual_overrides[track_id]

        # 2. Preserve existing assignment
        if track_id in self.player_teams and self.player_teams[track_id] is not None:
            return self.player_teams[track_id]

        # Update history
        self._update_color_history(track_id, color)

        # 3. Try current color prediction
        team = self.predict(color)
        if team is not None:
            self.player_teams[track_id] = team
            return team

        # 4. Fallback to majority vote over recent valid colors
        team = self._majority_vote_team(track_id)
        if team is not None:
            self.player_teams[track_id] = team
            return team

        # 5. No valid classification possible
        self.player_teams[track_id] = None
        return None

    def get_team_name(self, team):
        """
        Convert label to readable team name.
        """

        if team is None:
            return "Unknown"

        return self.team_names.get(team, "Unknown")