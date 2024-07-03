import pygame
from Objects2D import Coordinate2D, Line, Reward_Line, Wall, RayObj
from Colors import Colors
from Car import Car


class GameEnv:
    def __init__(
        self,
        game_width: float = 800,
        game_height: float = 600,
        background_color: tuple[float, float, float] = Colors.BLACK.value,
        car_image_path: str = "car1.png",
        start_speed=0,
        max_forward_speed: float = 3,
        max_backward_speed: float = 0,
        turn_angle: float = 5,
        acceleration: float = 0.1,
    ) -> None:
        self.walls: list[list[Wall, Wall]] = [[], []]
        self.reward_lines: list[Reward_Line] = []
        self.game_width: float = game_width
        self.game_height: float = game_height
        self.background_color: tuple[float, float, float] = background_color
        self.car_image_path: str = car_image_path
        self.screen: pygame.Surface = None
        self.static_background: pygame.Surface = None
        self.show_speed: bool = False
        self.show_rays: bool = False
        self.rays: list[RayObj] = []
        self.start_speed = start_speed
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.turn_angle = turn_angle
        self.acceleration = acceleration

    def _draw_line(
        self,
        line: Line,
        width: int = 2,
        update_display: bool = False,
    ):
        pygame.draw.line(
            self.screen,
            line.get_color(),
            line.get_a().get_coordinates(),
            line.get_b().get_coordinates(),
            width,
        )
        if update_display:
            pygame.display.update()

    def _draw_text(
        self,
        text: str,
        coord: Coordinate2D,
        color: tuple[float, float, float] = Colors.GOLD.value,
    ):
        text = self.fonts[36].render(f"{text}", True, color)
        text_rect = text.get_rect()
        text_rect.center = coord.get_coordinates()

        self.screen.blit(text, text_rect)

    def set_walls(self, list_of_walls: list[list[list[float, float]]]) -> None:
        idx = 0
        for i, walls in enumerate(list_of_walls):
            for wall in walls:
                self.get_one_side_walls(i).append(
                    Wall(
                        Coordinate2D(wall[0], wall[1]),
                        Coordinate2D(wall[2], wall[3]),
                        idx,
                    )
                )
                idx += 1

    def get_walls(self, flat: bool = False):
        if flat:
            return [wall for wall in self.walls[0] + self.walls[1]]
        return self.walls

    def get_one_side_walls(self, start_idx: int):
        return (
            self.walls[start_idx]
            if start_idx <= len(self.walls) and start_idx >= 0
            else None
        )

    def set_reward_lines(self, reward_lines: list[Reward_Line]) -> None:
        self.reward_lines = reward_lines

    def get_reward_lines(self) -> list[Reward_Line]:
        return self.reward_lines

    def set_reward_lines_from_walls(
        self,
    ) -> None:
        wall1s = self.get_one_side_walls(0)
        wall2s = self.get_one_side_walls(1)
        assigned_wall2s = []
        for wall1 in wall1s:
            mid_point = wall1.get_midpoint()
            min_distance = 99999999.9
            closest_wall = None

            for wall2 in wall2s:
                distance = wall2.get_midpoint().calculate_distance(mid_point)
                if distance < min_distance and wall2 not in assigned_wall2s:
                    min_distance = distance
                    closest_wall = wall2
            reward_Line = Reward_Line(
                mid_point, closest_wall.get_midpoint(), len(self.reward_lines)
            )
            intersect = False
            for wall in self.get_walls(flat=True):
                if (
                    wall is not wall1
                    and wall is not closest_wall
                    and wall.intersect(reward_Line)
                ):
                    intersect = True
                    break
            if not intersect:
                self.reward_lines.append(reward_Line)
                assigned_wall2s.append(closest_wall)

    def generate_ray(self):
        self.rays = []
        self.rays.append(
            RayObj(self.car.get_Coordinate2D(), 360 - self.car.get_car_angle())
        )

    def _draw_rays(self):
        for ray in self.rays:
            min_distance = 99999999.9
            first_intersection = None
            for wall in self.get_walls(flat=True):
                intersection = ray.calculate_intersection(wall)
                if intersection:
                    distance = intersection.calculate_distance(ray.get_a())
                    if distance < min_distance:
                        min_distance = distance
                        first_intersection = intersection

            if first_intersection:
                line = Line(ray.get_a(), first_intersection, Colors.WHITE)
                self._draw_line(line, width=1, update_display=True)

    def _draw_walls(self):
        for wall in self.get_walls(flat=True):
            self._draw_line(wall, width=5)

    def _draw_reward_lines(self, show_reward_sequence=True):
        for line in self.get_reward_lines():
            self._draw_line(line, width=2)
            if show_reward_sequence:
                self._draw_text(line.get_reward_sequence(), line.get_midpoint())

    def _draw_speed(self):
        speed_text = pygame.font.Font(None, 30).render(
            f"Speed: {round(self.car.get_car_speed(), 2)}", True, Colors.WHITE.value
        )
        self.screen.blit(speed_text, (10, 10))

    def initialize_game_data(
        self, walls_data: list[list[list[float, float]]], show_game=True
    ):
        self.set_walls(walls_data)
        self.set_reward_lines_from_walls()
        starting_index = 0
        starting_line = self.get_reward_lines()[starting_index]
        next_line = self.get_reward_lines()[
            (starting_index + 1) % len(self.get_reward_lines())
        ]

        direction_line = Line(starting_line.get_midpoint(), next_line.get_midpoint())
        self.car = GameCar(
            starting_line.get_midpoint().get_x(),
            starting_line.get_midpoint().get_y(),
            start_speed=self.start_speed,
            car_angle=180 - direction_line.calculate_angle(),
            max_forward_speed=self.max_forward_speed,
            max_backward_speed=self.max_backward_speed,
            turn_angle=self.turn_angle,
            acceleration=self.acceleration,
        )
        self.generate_ray()

    def draw(
        self,
        show_walls=True,
        show_reward_lines=True,
        show_reward_sequence=True,
        show_speed=True,
        show_rays=True,
    ):
        pygame.init()
        self.screen = pygame.display.set_mode((self.game_width, self.game_height))
        self.screen.fill(self.background_color)
        self.fonts = {36: pygame.font.Font(None, 36)}

        if show_walls:
            print("drawing walls")
            self._draw_walls()

        if show_reward_lines:
            print("drawing reward lines")
            self._draw_reward_lines(show_reward_sequence=show_reward_sequence)

        pygame.display.update()
        pygame.image.save(self.screen, "static_background.png")
        self.static_background = pygame.image.load("static_background.png")
        self.car.initial_draw_car(self.screen)
        if show_speed:
            self.show_speed = show_speed
            self._draw_speed()
        if show_rays:
            self.show_rays = show_rays
            self._draw_rays()

    def start_game(self):
        running = True
        while running:
            for event in pygame.event.get():
                if (
                    event.type == pygame.QUIT
                    # or pygame.key.get_pressed()[pygame.K_ESCAPE]
                ):
                    running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                self.car.turn_left()
            if keys[pygame.K_RIGHT]:
                self.car.turn_right()

            if keys[pygame.K_UP]:
                self.car.accelerate()

            if keys[pygame.K_DOWN]:
                self.car.decelerate()

            car_x, car_y = self.car.update_car_position()
            self.generate_ray()

            # car_rect = self.car_image.get_rect(center=(car_x, car_y))
            background_color = self.static_background.get_at((int(car_x), int(car_y)))
            if background_color == self.get_walls(flat=True)[0].get_color():
                running = False

            self.screen.blit(self.static_background, (0, 0))
            self.car.draw_car()
            if self.show_speed:
                self._draw_speed()
            if self.show_rays:
                self._draw_rays()
            pygame.display.flip()
            pygame.time.Clock().tick(60)


class GameCar(Car):
    def __init__(
        self,
        car_x: float = 200,
        car_y: float = 200,
        start_speed=0,
        car_angle: float = 0,
        max_forward_speed: float = 3,
        max_backward_speed: float = 0,
        turn_angle: float = 5,
        acceleration: float = 0.1,
    ):
        super().__init__(
            start_x=car_x,
            start_y=car_y,
            start_speed=start_speed,
            start_angle=car_angle,
            max_forward_speed=max_forward_speed,
            max_backward_speed=max_backward_speed,
            turn_angle=turn_angle,
            acceleration=acceleration,
        )
        self.screen = None

    def _rotate_car(self, angle: float, negative_speed=False):
        if negative_speed:
            (pygame.transform.flip(self.car_image, False, True),)
        rect = self.car_image.get_rect()
        rotated_image = pygame.transform.rotate(self.car_image, angle)
        rotated_rect = rotated_image.get_rect(center=rect.center)
        return rotated_image, rotated_rect

    def initial_draw_car(
        self,
        screen,
        path_to_image: str = "car.png",
        resize_width: float = 15,
        resize_height: float = 10,
    ):
        car_image = pygame.image.load(path_to_image)
        self.car_image = pygame.transform.scale(
            car_image, (resize_width, resize_height)
        )
        self.screen = screen
        if self.get_car_speed() >= 0:
            rotated_car, car_rect = self._rotate_car(self.get_car_angle())
        else:  # car_speed < 0
            rotated_car, car_rect = self._rotate_center(self.get_car_angle(), True)
        self.screen.blit(
            rotated_car,
            (
                self.get_car_x() - car_rect.width // 2,
                self.get_car_y() - car_rect.height // 2,
            ),
        )
        pygame.display.flip()

    def draw_car(self):
        if self.get_car_speed() >= 0:
            rotated_car, car_rect = self._rotate_car(self.get_car_angle())
        else:  # car_speed < 0
            rotated_car, car_rect = self._rotate_center(self.get_car_angle(), True)
        self.screen.blit(
            rotated_car,
            (
                self.get_car_x() - car_rect.width // 2,
                self.get_car_y() - car_rect.height // 2,
            ),
        )
