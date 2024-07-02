import math
from abc import ABC
from typing import Any
import Colors


class Coordinate2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def set_x(self, x: float):
        self.x = x

    def get_x(self) -> float:
        return self.x

    def set_y(self, y: float):
        self.y = y

    def get_y(self) -> float:
        return self.y

    def get_coordinates(self) -> tuple[float, float]:
        return (self.x, self.y)

    def calculate_distance(self, b) -> float:
        return math.sqrt(
            (self.get_x() - b.get_x()) ** 2 + (self.get_y() - b.get_y()) ** 2
        )


class Line(ABC):
    def __init__(self, a: Coordinate2D, b: Coordinate2D, color: Colors = Colors.WHITE):
        self.a = a
        self.b = b
        self.color = color

    def set_a(self, a: Coordinate2D) -> None:
        self.a = a

    def get_a(self) -> Coordinate2D:
        return self.a

    def set_b(self, b: Coordinate2D) -> None:
        self.b = b

    def get_b(self) -> Coordinate2D:
        return self.b

    def get_midpoint(self) -> Coordinate2D:
        return Coordinate2D(
            (self.a[0].x + self.b[0].x) / 2, (self.a[0].y + self.b[0].y) / 2
        )

    def get_color(self) -> Colors:
        return self.color

    def set_color(self, color: Colors):
        self.color = color


class Reward_Line(Line):
    def __init__(self, a: Coordinate2D, b: Coordinate2D, reward_sequence: int):
        super().__init__(a, b, Colors.CYAN)
        self.reward_sequence = reward_sequence

    def get_reward_sequence(self) -> float:
        return self.reward_sequence

    def set_reward_sequence(self, reward_sequence: float):
        self.reward_sequence = reward_sequence


class Wall(Line):
    def __init__(self, a: Coordinate2D, b: Coordinate2D, id: int):
        super().__init__(a, b, Colors.CYAN)
        self.id = id

    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int):
        self.id = id
