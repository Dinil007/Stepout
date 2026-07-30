from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class PlayerPosition:
    """Represents one tracked player at a specific moment.

    Attributes:
        player_id: Unique identifier for the player.
        team_id: Unique identifier for the team.
        team_name: Name of the team.
        jersey_number: Jersey number of the player.
        x: X coordinate on the pitch (normalized 0-1).
        y: Y coordinate on the pitch (normalized 0-1).
        frame_number: Frame number in the video sequence.
        timestamp: Timestamp of the detection.
        confidence: Detection confidence score (0.0-1.0).
        is_goalkeeper: Whether the player is the goalkeeper.
        is_visible: Whether the player is currently visible to the detector.
    """

    player_id: int
    team_id: int
    team_name: str
    jersey_number: int
    x: float
    y: float
    frame_number: int
    timestamp: datetime
    confidence: float = 1.0
    is_goalkeeper: bool = False
    is_visible: bool = True

    def is_valid(self) -> bool:
        """Check if the player position data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if self.frame_number < 0 or self.player_id < 0 or self.team_id < 0:
            return False
        if self.jersey_number < 1:
            return False
        return True

    def within_pitch_bounds(self) -> bool:
        """Check if the position lies within standard pitch boundaries.

        Returns:
            True if x and y are within [0.0, 1.0], False otherwise.
        """
        return 0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0

    def distance_to(self, other: PlayerPosition) -> float:
        """Calculate Euclidean distance to another player position.

        Args:
            other: Another PlayerPosition instance.

        Returns:
            Euclidean distance between the two positions.
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class FormationTemplate:
    """Represents a predefined tactical formation template.

    Attributes:
        formation_name: Name of the formation (e.g., "4-3-3", "4-4-2").
        defenders: Number of defenders.
        midfielders: Number of midfielders.
        forwards: Number of forwards.
        normalized_positions: Normalized (x, y) positions for each role.
        description: Human-readable description of the formation.
    """

    formation_name: str
    defenders: int
    midfielders: int
    forwards: int
    normalized_positions: list[tuple[float, float]] = field(default_factory=list)
    description: str = ""

    def player_count(self) -> int:
        """Calculate the total number of outfield players represented.

        Returns:
            Sum of defenders, midfielders, and forwards.
        """
        return self.defenders + self.midfielders + self.forwards

    def is_valid(self) -> bool:
        """Check if the formation template data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if self.defenders < 0 or self.midfielders < 0 or self.forwards < 0:
            return False
        if self.player_count() != len(self.normalized_positions):
            return False
        for x, y in self.normalized_positions:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return False
        return bool(self.formation_name)


@dataclass
class FormationDetection:
    """Represents the output of a formation detection step.

    Attributes:
        detected_formation: Name of the detected formation.
        confidence: Confidence in the detection (0.0-1.0).
        frame_number: Frame number where the detection occurred.
        timestamp: Timestamp of the detection.
        matched_template: Name of the best matching template.
        score: Similarity score to the matched template.
    """

    detected_formation: str
    confidence: float
    frame_number: int
    timestamp: datetime
    matched_template: str
    score: float

    def is_valid(self) -> bool:
        """Check if the formation detection data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if not (0.0 <= self.score <= 1.0):
            return False
        if self.frame_number < 0:
            return False
        return bool(self.detected_formation and self.matched_template)


@dataclass
class FormationMetrics:
    """Stores computed tactical shape metrics for a team.

    Attributes:
        team_width: Width of the team shape (normalized 0-1).
        team_length: Length of the team shape (normalized 0-1).
        compactness: Compactness score of the team shape (0.0-1.0).
        centroid_x: X coordinate of the team centroid (normalized 0-1).
        centroid_y: Y coordinate of the team centroid (normalized 0-1).
        convex_hull_area: Area of the convex hull covering player positions.
        defensive_line: Y coordinate of the defensive line (normalized 0-1).
        midfield_line: Y coordinate of the midfield line (normalized 0-1).
        forward_line: Y coordinate of the forward line (normalized 0-1).
        vertical_stretch: Vertical stretch of the team shape (normalized 0-1).
        horizontal_stretch: Horizontal stretch of the team shape (normalized 0-1).
    """

    team_width: float
    team_length: float
    compactness: float
    centroid_x: float
    centroid_y: float
    convex_hull_area: float
    defensive_line: float
    midfield_line: float
    forward_line: float
    vertical_stretch: float
    horizontal_stretch: float

    def is_valid(self) -> bool:
        """Check if the formation metrics data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if not (0.0 <= self.centroid_x <= 1.0 and 0.0 <= self.centroid_y <= 1.0):
            return False
        if not (0.0 <= self.compactness <= 1.0):
            return False
        if self.team_width < 0.0 or self.team_length < 0.0:
            return False
        if self.vertical_stretch < 0.0 or self.horizontal_stretch < 0.0:
            return False
        if self.convex_hull_area < 0.0:
            return False
        if not (0.0 <= self.defensive_line <= 1.0):
            return False
        if not (0.0 <= self.midfield_line <= 1.0):
            return False
        if not (0.0 <= self.forward_line <= 1.0):
            return False
        return True

    def within_pitch_bounds(self) -> bool:
        """Check if all position-based metrics are within pitch bounds.

        Returns:
            True if centroid and line coordinates are within [0.0, 1.0], False otherwise.
        """
        return (
            0.0 <= self.centroid_x <= 1.0
            and 0.0 <= self.centroid_y <= 1.0
            and 0.0 <= self.defensive_line <= 1.0
            and 0.0 <= self.midfield_line <= 1.0
            and 0.0 <= self.forward_line <= 1.0
        )


@dataclass
class FormationWindow:
    """Represents one analysed time window of tactical shape.

    Attributes:
        start_frame: Starting frame number of the window.
        end_frame: Ending frame number of the window.
        duration_seconds: Duration of the window in seconds.
        formation: Detected formation during this window.
        confidence: Confidence in the formation detection (0.0-1.0).
        metrics: FormationMetrics computed for this window.
    """

    start_frame: int
    end_frame: int
    duration_seconds: float
    formation: str
    confidence: float
    metrics: FormationMetrics

    def is_valid(self) -> bool:
        """Check if the formation window data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if self.start_frame < 0 or self.end_frame < 0:
            return False
        if self.start_frame > self.end_frame:
            return False
        if self.duration_seconds < 0:
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if not self.formation:
            return False
        if not self.metrics.is_valid():
            return False
        return True


@dataclass
class FormationTransition:
    """Represents one tactical formation change event.

    Attributes:
        previous_formation: Name of the previous formation.
        new_formation: Name of the new formation.
        timestamp: Timestamp when the transition occurred.
        frame_number: Frame number where the transition occurred.
        confidence: Confidence in the transition detection (0.0-1.0).
    """

    previous_formation: str
    new_formation: str
    timestamp: datetime
    frame_number: int
    confidence: float

    def is_valid(self) -> bool:
        """Check if the formation transition data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if self.frame_number < 0:
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if not (self.previous_formation and self.new_formation):
            return False
        return True


@dataclass
class TeamShape:
    """Represents average tactical positioning for a team over a period.

    Attributes:
        team_name: Name of the team.
        average_positions: Average (x, y) positions for each player.
        width: Average team width (normalized 0-1).
        length: Average team length (normalized 0-1).
        compactness: Average compactness score (0.0-1.0).
        centroid: Team centroid as (x, y) tuple (normalized 0-1).
        formation: Most common formation during the period.
    """

    team_name: str
    average_positions: list[tuple[float, float]] = field(default_factory=list)
    width: float = 0.0
    length: float = 0.0
    compactness: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    formation: str = ""

    def is_valid(self) -> bool:
        """Check if the team shape data is valid.

        Returns:
            True if all fields contain valid values, False otherwise.
        """
        if not self.team_name:
            return False
        if self.width < 0.0 or self.length < 0.0:
            return False
        if not (0.0 <= self.compactness <= 1.0):
            return False
        cx, cy = self.centroid
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            return False
        for x, y in self.average_positions:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return False
        return True

    def player_count(self) -> int:
        """Return the number of players tracked in average positions.

        Returns:
            Number of (x, y) tuples in average_positions.
        """
        return len(self.average_positions)