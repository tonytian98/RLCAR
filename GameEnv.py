import pygame
from Objects2D import Coordinate2D, Line, Reward_Line, Wall
from Colors import Colors


class GameEnv:
    def __init__(
        self,
        game_width: int = 800,
        game_height: int = 600,
        background_color=Colors.BLACK.value,
    ) -> None:
        self.walls: list[list[Wall, Wall]] = [[], []]
        self.reward_lines: list[Reward_Line] = []
        pygame.init()
        self.screen = pygame.display.set_mode((game_width, game_height))
        self.screen.fill(background_color)
        self.fonts = {36: pygame.font.Font(None, 36)}

    def _draw_line(self, line: Line, width=2):
        pygame.draw.line(
            self.screen,
            line.get_color(),
            line.get_a().get_coordinates(),
            line.get_b().get_coordinates(),
            width,
        )

    def _draw_text(self, text: str, coord: Coordinate2D, color=Colors.GOLD.value):
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

    def get_walls(self, flat=False):
        if flat:
            return [wall for wall in self.walls[0] + self.walls[1]]
        return self.walls

    def get_one_side_walls(self, start_idx):
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

    def _draw_walls(self):
        for wall in self.get_one_side_walls(0):
            self._draw_line(wall)
        for wall in self.get_one_side_walls(1):
            self._draw_line(wall)

    def _draw_reward_lines(self, show_reward_sequence=True):
        for line in self.get_reward_lines():
            self._draw_line(line)
            if show_reward_sequence:
                self._draw_text(line.get_reward_sequence(), line.get_midpoint())

    def draw(self):
        print("drawing walls")

        self._draw_walls()
        print("drawing reward lines")
        self._draw_reward_lines()
        pygame.display.update()
