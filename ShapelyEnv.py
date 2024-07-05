from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
import matplotlib.pyplot as plt
from ImageProcessor import ImageProcessor
from Car import Car
from Rule import RuleKeyboard, Rule
import time
from pynput import keyboard


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
        self.car: Car = None
        self.rays = []
        self.rule: Rule = None
        self.image_path = ""
        self.fig, self.ax = plt.subplots()
        self.background_axis = self.fig.add_axes(self.ax.get_position(), frameon=False)
        # self.fig.canvas.draw()
        # plt.show(block=False)

    def set_track_environment(self, img_processor: ImageProcessor):
        """
        This method sets up the track environment based on the given image processor.

        Parameters:
        img_processor (ImageProcessor): An instance of the ImageProcessor class, which contains the image data.

        Returns:
        None. The method sets up the track environment by calling other methods within the ShapeEnv class.
        """
        self.image_path = img_processor.get_image_path()
        segments = img_processor.find_contour_segments()
        outer = img_processor.find_segment_points(segments[0])
        inner = img_processor.find_segment_points(segments[1])
        game_env.set_track_from_segment_points(outer, inner)
        game_env.set_walls(segments)
        game_env.set_reward_lines_from_walls()
        game_env.set_inverse_track()
        game_env.segment_track()

    def set_car(self, car: Car):
        self.car = car

    def set_rule(self, rule: Rule):
        self.rule = rule

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
    ) -> list[tuple[float, float]]:
        """
        This method calculates the endpoints of unstopping rays for a specific quadrant.

        Parameters:
        car_x (float): The x-coordinate of the car's position.
        car_y (float): The y-coordinate of the car's position.
        up (bool): A flag indicating whether the quadrant is in the upper half of the track.
        right (bool): A flag indicating whether the quadrant is in the right half of the track.

        Returns:
        list[tuple[float, float]]: A list of tuples representing the endpoints of the unstopping rays.
                                Each tuple contains the x and y coordinates of an endpoint.
        """
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
            tuple(corner),
            tuple([car_x, boundary_y]),
            tuple([boundary_x, car_y]),
            tuple([boundary_x, (car_y + corner[1]) / 2]),
            tuple([(car_x + corner[0]) / 2, boundary_y]),
        ]

    def get_stopped_ray(
        self, unstopping_ray: LineString, car_point: Point
    ) -> LineString:
        """
        This method calculates the stopped ray from the unstopping ray.

        Parameters:
        unstopping_ray (LineString): The unstopping ray from the car to a boundary point.
        car_point (Point): The current position of the car.

        Returns:
        LineString: The stopped ray, which is a segment that is the part of the unstopping ray
                    that originates from the car and stopped at the first contact with the track's border.
        """

        intersections = self.track.intersection(unstopping_ray)

        if isinstance(intersections, LineString):
            return intersections

        # if there are multiple intersections, return the one that originates from the car
        for intersection in intersections.geoms:
            if intersection.intersects(car_point):
                return intersection

    def get_unstopping_ray_endpoints_by_quadrants(
        self, car_point: Point, quadrant_indices: list[int]
    ) -> list[list[tuple[float, float]]]:
        """
        This method calculates the unstopping ray endpoints for specific quadrants, and returns them following the order of i to iv.

        Parameters:
        car_point (Point): The current position of the car.
        quadrant_indices (list[int]): A list of integers representing the quadrant indices.
                                    Each index corresponds to a specific quadrant.
                                    For example, [0, 1, 2, 3] represents all quadrants.

        Returns:
        list[list[tuple[float, float]]]: A list of lists of tuples.
                                        Each inner list represents the unstopping ray endpoints for a specific quadrant.
                                        Each tuple contains the x and y coordinates of an endpoint.
        """

        result = []
        if 0 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, True, True
                )
            )

        if 1 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, True, False
                )
            )

        if 2 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, False, False
                )
            )

        if 3 in quadrant_indices:
            result.append(
                self.get_unstopping_ray_endpoints_by_quadrant(
                    car_point.x, car_point.y, False, True
                )
            )
        return result

    # only 9 nine rays no matter the situation
    def update_rays(self):
        angle = self.car.get_car_angle()
        car_point = self.car.get_shapely_point()
        self.rays = []
        unstopping_ray_endpoints = set()
        # [22.5, 67.5)
        if angle % 360 >= 90 / 4 * 1 and angle % 360 < 90 / 4 * 3:
            """
            quadrant_i = self.get_unstopping_ray_endpoints_by_quadrant(
                car_point.x, car_point.y, True, True
            )
            quadrant_ii = self.get_unstopping_ray_endpoints_by_quadrant(
                car_point.x, car_point.y, True, False
            )
            quadrant_iv = self.get_unstopping_ray_endpoints_by_quadrant(
                car_point.x, car_point.y, False, True
            )
            """
            quadrant_i, quadrant_ii, quadrant_iv = (
                self.get_unstopping_ray_endpoints_by_quadrants(car_point, [0, 1, 3])
            )
            left_boundary, left_mid = quadrant_ii[0], quadrant_ii[-1]
            right_boundary, right_mid = quadrant_iv[0], quadrant_iv[3]
            unstopping_ray_endpoints.update(
                set(quadrant_i + [left_boundary, left_mid, right_mid, right_boundary])
            )
        # [67.5, 112.5)
        elif angle % 360 >= 90 / 4 * 3 and angle % 360 < 90 / 4 * 5:
            quadrant_i, quadrant_ii = self.get_unstopping_ray_endpoints_by_quadrants(
                car_point, [0, 1]
            )
            unstopping_ray_endpoints.update(set(quadrant_i + quadrant_ii))
        # [112.5, 157.5)
        elif angle % 360 >= 90 / 4 * 5 and angle % 360 < 90 / 4 * 7:
            quadrant_i, quadrant_ii, quadrant_iii = (
                self.get_unstopping_ray_endpoints_by_quadrants(car_point, [0, 1, 2])
            )
            left_boundary, left_mid = quadrant_iii[0], quadrant_iii[3]
            right_boundary, right_mid = quadrant_i[0], quadrant_i[-1]
            unstopping_ray_endpoints.update(
                set(quadrant_ii + [left_boundary, left_mid, right_mid, right_boundary])
            )
        # [157.5, 202.5)
        elif angle % 360 >= 90 / 4 * 7 and angle % 360 < 90 / 4 * 9:
            quadrant_ii, quadrant_iii = self.get_unstopping_ray_endpoints_by_quadrants(
                car_point, [1, 2]
            )
            unstopping_ray_endpoints.update(set(quadrant_ii + quadrant_iii))
        # [202.5, 247.5)
        elif angle % 360 >= 90 / 4 * 9 and angle % 360 < 90 / 4 * 11:
            quadrant_ii, quadrant_iii, quadrant_iv = (
                self.get_unstopping_ray_endpoints_by_quadrants(car_point, [1, 2, 3])
            )
            left_boundary, left_mid = quadrant_ii[0], quadrant_ii[3]
            right_boundary, right_mid = quadrant_iv[0], quadrant_iv[-1]
            unstopping_ray_endpoints.update(
                quadrant_iii + [left_boundary, left_mid, right_mid, right_boundary]
            )

        # [247.5, 292.5)
        elif angle % 360 >= 90 / 4 * 11 and angle % 360 < 90 / 4 * 13:
            quadrant_iii, quadrant_iv = self.get_unstopping_ray_endpoints_by_quadrants(
                car_point, [2, 3]
            )
            unstopping_ray_endpoints.update(set(quadrant_iii + quadrant_iv))
        # [292.5, 337.5)
        elif angle % 360 >= 90 / 4 * 13 and angle % 360 < 90 / 4 * 15:
            quadrant_i, quadrant_iii, quadrant_iv = (
                self.get_unstopping_ray_endpoints_by_quadrants(car_point, [0, 2, 3])
            )
            left_boundary, left_mid = quadrant_iii[0], quadrant_iii[-1]
            right_boundary, right_mid = quadrant_i[0], quadrant_i[3]
            unstopping_ray_endpoints.update(
                quadrant_iv + [left_boundary, left_mid, right_mid, right_boundary]
            )
        # [337.5, 360) & [0, 22.5)
        elif angle % 360 >= 90 / 4 * 15 or angle % 360 < 90 / 4 * 1:
            quadrant_i, quadrant_iv = self.get_unstopping_ray_endpoints_by_quadrants(
                car_point, [0, 3]
            )
            unstopping_ray_endpoints.update(set(quadrant_i + quadrant_iv))

        for x, y in unstopping_ray_endpoints:
            unstopping_ray = LineString([(car_point.x, car_point.y), (x, y)])
            self.rays.append(self.get_stopped_ray(unstopping_ray, car_point))

    def plot_shapely_objs(self, shapely_objs: list, show=True, save=False):
        geos = gpd.GeoSeries(shapely_objs)
        df = gpd.GeoDataFrame({"geometry": geos})
        df.plot(ax=self.ax, color="red")
        plt.pause(0.001)

        if save:
            current_time = time.time()
            plt.save(f"figure_{current_time}.png")

    def update_game_frame(self, shapely_objs: list, save=False):
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.plot_shapely_objs(shapely_objs, save=save)

    def plot_polygon2(self, polygons):
        for polygon in polygons.geoms:
            x, y = polygon.exterior.xy
            self.ax.plot(x, y)
            plt.pause(0.001)

    def draw_car(self):
        self.ax.scatter([self.car.get_car_x()], [self.car.get_car_y()], color="red")
        plt.pause(0.0001)

    def game_end(self) -> bool:
        return self.inverse_track.contains(self.car.get_shapely_point())

    def draw_background(self, save=False):
        self.background_axis.set_xlim(0, self.width)
        self.background_axis.set_ylim(0, self.height)
        geos = gpd.GeoSeries([game_env.inverse_track])
        df = gpd.GeoDataFrame({"geometry": geos})
        df.plot(ax=self.background_axis, color="black")
        if save:
            plt.savefig(f"processed_{self.image_path}.png")

    def start_game_keyboard(self, show=True):
        running = True
        listener = keyboard.Listener(
            on_press=self.on_press,
        )
        listener.start()
        if show:
            self.draw_background()
        while running:  # print
            self.update_rays()
            self.car.update_car_position()
            self.update_game_frame([car.get_shapely_point()] + self.rays)
            running = not self.game_end()

    def on_press(self, key):
        if key == keyboard.Key.up:
            self.car.accelerate()
            print("up")
        if key == keyboard.Key.down:
            self.car.decelerate()
            print("down")
        if key == keyboard.Key.left:
            self.car.turn_left()
            print("left")
        if key == keyboard.Key.right:
            self.car.turn_right()
            print("right")


if __name__ == "__main__":
    width = 800
    height = 600
    plt.ion()

    # object creation
    img_processor = ImageProcessor("map1.png", resize=[width, height])
    game_env = ShapeEnv(width, height)
    car = Car(650, 100, 0, 90)

    # environment setup
    game_env.set_track_environment(img_processor)
    game_env.set_car(car)
    game_env.set_rule(RuleKeyboard(game_env.car))

    game_env.start_game_keyboard(show=True)
