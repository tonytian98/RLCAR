import numpy as np


class Car:
    def __init__(
        self,
        start_x: float,
        start_y: float,
        start_speed: float,
        start_angle: float,
        max_forward_speed: float = 4,
        max_backward_speed: float = 0,
        acceleration: float = 0.1,
        turn_angle: float = 5,
    ):
        self.car_x = start_x
        self.car_y = start_y
        self.car_speed = start_speed
        self.car_angle = start_angle
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.acceleration = acceleration
        self.turn_angle = turn_angle

    def get_car_x(self) -> float:
        return self.car_x

    def set_car_x(self, car_x: float) -> None:
        self.car_x = car_x

    def get_car_y(self) -> float:
        return self.car_y

    def set_car_y(self, car_y: float) -> None:
        self.car_y = car_y

    def get_car_speed(self) -> float:
        return self.car_speed

    def set_car_speed(self, car_speed: float) -> None:
        self.car_speed = car_speed

    def get_car_angle(self) -> float:
        return self.car_angle

    def set_car_angle(self, car_angle: float) -> None:
        self.car_angle = car_angle

    def get_max_forward_speed(self) -> float:
        return self.max_forward_speed

    def set_max_forward_speed(self, max_forward_speed: float) -> None:
        self.max_forward_speed = max_forward_speed

    def get_max_backward_speed(self) -> float:
        return self.max_backward_speed

    def set_max_backward_speed(self, max_backward_speed: float) -> None:
        self.max_backward_speed = max_backward_speed

    def get_acceleration(self) -> float:
        return self.acceleration

    def set_acceleration(self, acceleration: float) -> None:
        self.acceleration = acceleration

    def get_turn_angle(self) -> float:
        return self.turn_angle

    def set_turn_angle(self, turn_angle: float) -> None:
        self.turn_angle = turn_angle

    def turn_right(self) -> None:
        self.car_angle -= self.turn_angle

    def turn_left(self) -> None:
        self.car_angle += self.turn_angle

    def accelerate(self) -> None:
        if self.car_speed < self.max_forward_speed:
            self.car_speed += self.acceleration

    def decelerate(self) -> None:
        if self.car_speed > -self.max_backward_speed:
            self.car_speed = max(
                -self.max_backward_speed, self.car_speed - self.acceleration
            )

    def update_car_position(self) -> None:
        self.car_x += self.car_speed * np.cos(np.radians(self.car_angle))
        self.car_y -= self.car_speed * np.sin(np.radians(self.car_angle))
        return self.car_x, self.car_y
