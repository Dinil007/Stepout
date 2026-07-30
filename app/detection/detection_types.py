"""Shared detection data types."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    cls_id: int
    conf: float
    bbox: Tuple[int, int, int, int]
    track_id: int = -1
    reject_reason: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def foot(self) -> Tuple[int, int]:
        x1, _, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int(y2))

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
