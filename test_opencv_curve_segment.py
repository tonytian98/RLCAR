import cv2
import pygame
from time import sleep


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


class Line:
    def __init__(self, a: Coordinate2D, b: Coordinate2D):
        self.a = a
        self.b = b

    def set_a(self, a: Coordinate2D) -> None:
        self.a = a

    def get_a(self) -> Coordinate2D:
        return self.a

    def set_b(self, b: Coordinate2D) -> None:
        self.b = b

    def get_b(self) -> Coordinate2D:
        return self.b


class Reward_Line(Line):
    def __init__(self, a: Coordinate2D, b: Coordinate2D, reward_sequence: float):
        super().__init__(a, b)
        self.reward_sequence = reward_sequence

    def get_reward_sequence(self) -> float:
        return self.reward_sequence

    def set_reward_sequence(self, reward_sequence: float):
        self.reward_sequence = reward_sequence


class Wall:
    def __init__(
        self,
        line: Line,
        id: int = -1,
    ):
        self.line = line
        self.id = id

    def set_a(self, a: Coordinate2D):
        self.line.set_a(a)

    def get_a(self) -> Coordinate2D:
        return self.line.get_a()

    def set_b(self, b: Coordinate2D):
        self.line.set_b(b)

    def get_b(self) -> Coordinate2D:
        return self.line.get_b()

    def get_id(self) -> int:
        return self.id

    def set_id(self, id: int):
        self.id = id

    def get_midpoint(self) -> Coordinate2D:
        return Coordinate2D(
            (self.a[0].x + self.b[0].x) / 2, (self.a[0].y + self.b[0].y) / 2
        )


class Env:
    def __init__(self, map_path: str):
        self.map = cv2.imread(map_path)
        self.walls: list[list[Wall, Wall]] = [[], []]

    def set_walls(self, walls: list[list[float, float]], start_idx=0) -> None:
        for i, wall in enumerate(walls, start=start_idx):
            self.walls.append(
                Wall(
                    Line(
                        Coordinate2D(wall[0], wall[1]), Coordinate2D(wall[2], wall[3])
                    ),
                    i,
                )
            )


def find_longest_list(lists: list[list[float]]) -> tuple[list[float], int]:
    longest_list = max(lists, key=len, default=None)
    if longest_list:
        length_difference = len(longest_list) - len(min(lists, key=len))
        return longest_list, length_difference
    else:
        return None, 0


# Initialize Pygame
pygame.init()

# Load the image
image = cv2.imread("map2.png")
image = cv2.resize(image, (800, 600))
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary track
_, binary_track = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary track
contours, _ = cv2.findContours(binary_track, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the two largest contours
largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

segments = [[], []]
# Loop through the largest contours to extract the line segments
for j, contour in enumerate(largest_contours):
    # Approximate the contour to a polygon with a maximum of 4 edges
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.004 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Loop through the edges of the polygon to extract the line segments
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        segments[j].append([p1[0], p1[1], p2[0], p2[1]])


longest_segment, length_difference = find_longest_list(segments)


if longest_segment:
    longest_segment[-1 - length_difference][-2:] = longest_segment[-1][-2:]
    del longest_segment[-length_difference:]

# Create a Pygame window
screen = pygame.display.set_mode((800, 600))

font = pygame.font.Font(None, 36)


# Draw the line segments on the Pygame window
for j, line_segments in enumerate(segments):
    for i, line_segment in enumerate(line_segments):
        pygame.draw.line(
            screen,
            (0, 255 if i % 2 == 0 else 100, 0),
            (line_segment[0], line_segment[1]),
            (line_segment[2], line_segment[3]),
            2,
        )
        text = font.render(f"{i}", True, (255, 0, 0))

        # Get the text's rectangular dimensions
        text_rect = text.get_rect()

        # Center the text on the red dot's coordinates
        text_rect.center = (line_segment[0], line_segment[1])

        screen.blit(text, text_rect)

# Update the Pygame display
pygame.display.update()

sleep(100)
