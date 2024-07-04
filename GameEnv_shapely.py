from shapely.geometry import Point, Polygon, linestring


class block:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, x3:float , y3: float, x4:float, y4 :float, x5:float, y5):
        
        self.wall1_a = Point(x1, y1)
        self.wall1_b =  Point(x2, y1)
        self.wall2_a
        self.wall2_b
        
        
        
        self.start_line = Point(x1, y1)
        self.end = Point(x2, y2)

class GameEnv_shapely: