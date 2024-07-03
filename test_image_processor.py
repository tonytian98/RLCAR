from ImageProcessor import ImageProcessor
from GameEnv import GameEnv
from time import sleep
from shapely import Point


class wall:
    def __init__(self, x1, y1, x2, y2):
        self.start = Point(x1, y1)
        self.end = Point(x2, y2)

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def __str__(self):
        return f"({self.start.x}, {self.start.y}) - ({self.end.x}, {self.end.y})"


if __name__ == "__main__":
    width = 1200
    height = 900
    img_processor = ImageProcessor("map1.png", resize=[width, height])

    segments = img_processor.find_contour_segments()
    print(segments)
    print(
        wall(segments[0][0][0], segments[0][0][1], segments[0][1][0], segments[0][1][1])
    )
