from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
import matplotlib.pyplot as plt
from ImageProcessor import ImageProcessor
from Car import Car
import math


class ShapeEnv:
    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.background = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        self.walls = [[], []]
        self.reward_lines = []
        self.track: Polygon = None
        self.inverse_track: Polygon = None
        self.segmented_track_in_order = []
        self.rays = []

    def set_track_from_segment_points(self, outer, inner):
        outer_polygon = Polygon(outer)
        inner_poly = Polygon(inner)
        self.track = outer_polygon.difference(inner_poly)

    def set_walls(self, list_of_walls: list[list[list[float, float]]]):
        for i, walls in enumerate(list_of_walls):
            for wall in walls:
                self.walls[i].append(
                    LineString([(wall[0], wall[1]), (wall[2], wall[3])])
                )

    def get_walls(self, flat=False):
        if flat:
            return [wall for wall in self.walls[0] + self.walls[1]]
        return self.walls

    def set_reward_lines_from_walls(
        self,
    ) -> None:
        wall1s = self.walls[0]
        wall2s = self.walls[1]
        assigned_wall2s = []
        for wall1 in wall1s:
            mid_point: Point = wall1.centroid
            min_distance = 99999999.9
            closest_wall = None

            for wall2 in wall2s:
                distance = wall2.centroid.distance(mid_point)
                if distance < min_distance and wall2 not in assigned_wall2s:
                    min_distance = distance
                    closest_wall = wall2
            reward_Line = LineString([mid_point, closest_wall.centroid])
            intersect = False
            for wall in self.get_walls(flat=True):
                if (
                    wall is not wall1
                    and wall is not closest_wall
                    and wall.intersects(reward_Line)
                ):
                    intersect = True
                    break
            if not intersect:
                self.reward_lines.append(reward_Line)
                assigned_wall2s.append(closest_wall)

    def set_inverse_track(self):
        self.inverse_track = self.background.difference(self.track)

    def segment_track(self):
        buffer_size = 1e-6
        reward_lines_size = len(self.reward_lines)
        step = 2
        for i in range(0, reward_lines_size, step):
            segmented_track = sorted(
                self.track.difference(self.reward_lines[i].buffer(buffer_size))
                .difference(
                    self.reward_lines[
                        i + step if i + step < reward_lines_size else 0
                    ].buffer(buffer_size)
                )
                .geoms,
                key=lambda x: x.area,
            )[0]

            self.segmented_track_in_order.append(segmented_track)

    def get_unstopping_ray_endpoints_by_quadrant(
        self, car_x, car_y, up: bool, right: bool
    ):
        if up and right:
            corner = [self.width, self.height]
        elif up and not right:
            corner = [0, self.height]
        elif not up and right:
            corner = [self.width, 0]
        else:
            corner = [0, 0]
        boundary_y = int(up) * self.height
        boundary_x = int(right) * self.width
        return [
            corner,
            [car_x, boundary_y],
            [car_y, boundary_x],
            [boundary_x, (car_y + corner[1]) / 2],
            [(car_x + corner[0]) / 2, boundary_y],
        ]

    # only 9 nine rays no matter the situation
    def update_rays(self, car: Car):
        angle = car.get_car_angle()
        car_point = car.get_shapely_point()
        self.rays = []
        unstopping_ray_endpoints = set()
        if angle % 360 <= 90 / 4 * 1 or angle % 360 >= 90 / 4 * 3:
            unstopping_ray_endpoints.union(
                set(
                    self.get_unstopping_ray_endpoints_by_quadrant(
                        car_point.x, car_point.y, True, True
                    )
                )
            )
            quadrant_ii = self.get_unstopping_ray_endpoints_by_quadrant(
                car_point.x, car_point.y, True, False
            )
            left_boundary, left_mid = quadrant_ii[0], quadrant_ii[-1]
            unstopping_ray_endpoints.union()

            unstopping_ray = LineString([(car_point.x, car_point.y), (x, side_y)])
            intersections = self.track.intersection(unstopping_ray).geoms
            for intersection in intersections:
                if intersection.intersects(car_point):
                    self.rays.append(LineString([car_point, intersection.coords[1]]))

    def plot_polygon(self, shapes: list):
        geos = gpd.GeoSeries(shapes)
        df = gpd.GeoDataFrame({"geometry": geos})
        df.plot()
        plt.show()


if __name__ == "__main__":
    width = 800
    height = 600

    # image processing
    img_processor = ImageProcessor("map1.png", resize=[width, height])

    segments = img_processor.find_contour_segments()
    outer = img_processor.find_segment_points(segments[0])
    inner = img_processor.find_segment_points(segments[1])

    # ####
    # game env
    ## track processing
    game_env = ShapeEnv()
    game_env.set_track_from_segment_points(outer, inner)
    game_env.set_walls(segments)
    game_env.set_reward_lines_from_walls()
    game_env.set_inverse_track()
    game_env.segment_track()

    # ####

    # car
    car = Car(150, 200, 0, 0)
    game_env.update_rays(car)
    game_env.plot_polygon(
        [game_env.inverse_track] + [car.get_shapely_point()] + game_env.rays
    )
