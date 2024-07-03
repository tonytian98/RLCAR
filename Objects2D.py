import math
import numpy as np
from Colors import Colors

from sympy import Point, Line, Segment, Ray, FiniteSet, EmptySet


class Coordinate2D:
    def __init__(self, x: float, y: float):
        """
        Initialize a Coordinate2D object with given x and y coordinates.

        Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.

        Returns:
        None
        """
        self.x = x
        self.y = y

    def set_x(self, x: float):
        """
        Set the x-coordinate of the point.

        Parameters:
        x (float): The new x-coordinate value.

        Returns:
        None
        """
        self.x = x

    def get_x(self) -> float:
        return self.x

    def set_y(self, y: float):
        self.y = y

    def get_y(self) -> float:
        return self.y

    def get_coordinates(self) -> tuple[float, float]:
        """
        Returns the coordinates of the 2D point.

        Parameters:
        self (Coordinate2D): The instance of the Coordinate2D class.

        Returns:
        tuple[float, float]: A tuple containing the x and y coordinates of the point.
        """
        return (self.x, self.y)

    def calculate_distance(self, b) -> float:
        return math.sqrt(
            (self.get_x() - b.get_x()) ** 2 + (self.get_y() - b.get_y()) ** 2
        )

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


class Line:
    def __init__(self, a: Coordinate2D, b: Coordinate2D, color=Colors.WHITE):
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

    def calculate_angle(self) -> float:
        """
        Calculate the angle between the line segment formed by points a and b.

        Parameters:
        self (Line): The instance of the Line class.

        Returns:
        float: The angle in degrees [0, 360] between the line segment and the positive x-axis.

        Note:
        The angle is calculated using the arctangent of the slope of the line segment.
        The slope is calculated as the difference in y-coordinates divided by the difference in x-coordinates.
        The result is then converted to degrees using the np.degrees function.
        """
        return np.degrees(
            np.math.atan2(
                (self.get_a().get_y() - self.get_b().get_y()),
                (self.get_a().get_x() - self.get_b().get_x()),
            )
        )

    def calculate_perpendicular_angle(self) -> float:
        return 90 + self.calculate_angle()

    def get_sympy_segment(self) -> Segment:
        return Segment(
            Point(self.get_a().get_x(), self.get_a().get_y()),
            Point(self.get_b().get_x(), self.get_b().get_y()),
        )

    def get_color(self) -> tuple[float, float, float]:
        return self.color.value

    def __str__(self) -> str:
        return f"({self.get_a().get_x()}, {self.get_a().get_y()}) - ({self.get_b().get_x()}, {self.get_b().get_y()})"


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


class RayObj:
    def __init__(self, a: Coordinate2D, angle: float):
        self.a = a
        self.angle = angle

    def get_a(self) -> Coordinate2D:
        return self.a

    def set_a(self, a: Coordinate2D) -> None:
        self.a = a

    def get_angle(self) -> float:
        return self.angle

    def set_angle(self, angle: float) -> None:
        self.angle = angle

    def get_sympy_ray(self) -> Ray:
        return Ray(
            Point(self.a.get_x(), self.a.get_y()), angle=math.radians(self.angle)
        )

    def calculate_intersection(self, line: Line) -> Coordinate2D:
        ray = self.get_sympy_ray()
        intersection = ray.intersect(line.get_sympy_segment())

        if isinstance(intersection, Segment):
            return (
                line.get_a()
                if self.a.calculate_distance(line.get_a())
                <= self.a.calculate_distance(line.get_b())
                else line.get_b()
            )

        if isinstance(intersection, type(EmptySet)):
            return None

        if isinstance(intersection, FiniteSet):
            x, y = intersection.args[0].coordinates
            return Coordinate2D(float(x.evalf().evalf()), float(y.evalf().evalf()))


def calculate_intersection(
    ray_start_x,
    ray_start_y,
    ray_angle,
    line_start_x,
    line_start_y,
    line_end_x,
    line_end_y,
):
    # Convert the ray angle to radians
    angle_rad = math.radians(ray_angle)

    # Calculate the slope of the line
    line_slope = (line_end_y - line_start_y) / (line_end_x - line_start_x)

    # Calculate the intersection point using the formula
    x_intersect = (
        line_slope * line_start_x
        - line_start_y
        - ray_start_y
        + ray_start_x * math.tan(angle_rad)
    ) / (line_slope - math.tan(angle_rad))
    y_intersect = line_slope * (x_intersect - line_start_x) + line_start_y

    # Calculate the length of line PT
    length_pt = math.sqrt(
        (x_intersect - ray_start_x) ** 2 + (y_intersect - ray_start_y) ** 2
    )

    return (x_intersect, y_intersect, length_pt)
