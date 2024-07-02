import math
from abc import ABC
from Colors import Colors

from sympy import Point, Line, Segment


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
    def __init__(self, a: Coordinate2D, b: Coordinate2D, color=Colors.WHITE.value):
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
            (self.get_a().get_x() + self.get_b().get_x()) / 2,
            (self.get_a().get_y() + self.get_b().get_y()) / 2,
        )

    def intersect(self, other: "Line") -> Coordinate2D:
        p1, p2, p3, p4 = (
            Point(self.get_a().get_x(), self.get_a().get_y()),
            Point(self.get_b().get_x(), self.get_b().get_y()),
            Point(other.get_a().get_x(), other.get_a().get_y()),
            Point(other.get_b().get_x(), other.get_b().get_y()),
        )
        l1 = Segment(p1, p2)
        s1 = Segment(p3, p4)
        showIntersection = l1.intersection(s1)

        if showIntersection:
            x, y = showIntersection[0].coordinates
            return Coordinate2D(x, y)
        return None

    def get_color(self) -> tuple[float, float, float]:
        return self.color.value


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
        super().__init__(a, b, Colors.GREEN)
        self.id = id

    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int):
        self.id = id
