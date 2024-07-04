import shapely.geometry as sg
from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    LinearRing,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from Car import Car

# List of coordinates for the polygon
coordinates = [(0, 0), (0, 600), (800, 600), (800, 0)]
track_block1 = [(50, 50), (100, 50), (100, 100), (50, 100)]
track_block2 = [(101, 100), (101, 50), (200, 150), (200, 200)]
car_start = (75, 75)
# Create a Shapely Polygon object
polygon = sg.Polygon(coordinates, holes=[track_block1, track_block2])
geos = gpd.GeoSeries([polygon])
lr = LinearRing(coordinates)
df = gpd.GeoDataFrame({"geometry": geos})
df.plot()
plt.show()
print(polygon.is_valid)
car_start_point = Point(car_start)
print("boundary", polygon.boundary)
print("ext", polygon.exterior)
x = polygon.interiors
for a in x:
    print("inte", a)

print(lr.contains(car_start_point))

car = Car(car_start[0], car_start[1], 0, 0)
running = False
while running:
    car.accelerate()
    point = Point(car.update_car_position())
    print(point)
    if polygon.contains(point):
        running = False

# Print the polygon
