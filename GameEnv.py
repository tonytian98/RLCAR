import pygame
from Objects2D import Coordinate2D, Line, Reward_Line, Wall
import Color


class GameEnv:
    def __init__(self, game_width: int = 800, game_height: int = 600):
        self.walls: list[list[Wall, Wall]] = [[], []]
        self.reward_lines: list[Reward_Line] = []
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.screen.fill(Color.BLACK)
        # font = pygame.font.Font(None, 36)

    def _draw_line(self, line: Line, width=2):
        pygame.draw.line(
            self.screen, line.get_color(), line.get_a(), line.get_b(), width
        )

    def set_walls(self, walls: list[list[list[float, float]]]) -> None:
        idx = 0
        for walls in walls:
            for wall in walls:
                self.walls.append(
                    Wall(
                        Coordinate2D(wall[0], wall[1]),
                        Coordinate2D(wall[2], wall[3]),
                        idx,
                    )
                )
                idx += 1

    def get_walls(
        self,
    ):
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

    def set_reward_lines_walls(
        self,
    ) -> None:
        wall1s = self.get_one_side_walls(0)
        wall2s = self.get_one_side_walls(1)
        assigned_wall2s = []
        for wall in wall1s:
            mid_point = wall.get_midpoint()
            min_distance = 99999999.9
            closest_wall = None

            for wall2 in wall2s:
                distance = wall2.get_midpoint.calculate_distance(mid_point)
                if distance < min_distance and wall2 not in assigned_wall2s:
                    min_distance = distance
                    closest_wall = wall2
            self.reward_lines.append(
                Reward_Line(mid_point, wall2.get_midpoint(), len(self.reward_lines))
            )
            assigned_wall2s.append(closest_wall)

    def _draw_walls(self):
        for wall in self.get_one_side_walls(0):
            self._draw_line(wall)
        for wall in self.get_one_side_walls(1):
            self._draw_line(wall)

    def _draw_reward_lines(self):
        for line in self.get_reward_lines():
            self._draw_line(line)

    def draw(self):
        self._draw_walls()
        self._draw_reward_lines()
        pygame.display.update()
