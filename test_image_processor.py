from ImageProcessor import ImageProcessor
from GameEnv import GameEnv
from time import sleep
from shapely import Point, LineString, LinearRing

import geopandas as gpd
import matplotlib.pyplot as plt


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


class track:
    def __init__(self, side1, side2):
        self.wall1 = side1
        self.wall2 = side2


def plot_polygon(shapes: list):
    geos = gpd.GeoSeries(shapes)
    df = gpd.GeoDataFrame({"geometry": geos})
    df.plot()
    plt.show()


if __name__ == "__main__":
    width = 800
    height = 600
    img_processor = ImageProcessor("map1.png", resize=[width, height])

    segments = img_processor.find_contour_segments()
    points1 = img_processor.find_segment_points(segments[0])
    print(len(points1))
    points2 = img_processor.find_segment_points(segments[1])
    print((points2))
    pairing = img_processor.assign_closest_points(points1, points2.copy())
    print(len(pairing))

    outer_bound = LinearRing(points1)
    inner_bound = LinearRing(points2)
    lines = [outer_bound, inner_bound]
    print(inner_bound)
    plot_polygon(lines)
    for p in pairing:
        line = LineString([(p[0][0], p[0][1]), (p[1][0], p[1][1])])
        lines.append(line)
