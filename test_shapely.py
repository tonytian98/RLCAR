from shapely.geometry import LineString, Point
import math

# Define the line segment
# segment = LineString([(0, 0), (2, 2)])

# Define the point
# point = Point(8, 0)

# Calculate the distance from the point to the segment
# distance = segment.distance(point)


import time
import numpy as np


def timer(func, *args, **kwargs):
    start_time = time.time() * 1000
    result = func(*args, **kwargs)
    end_time = time.time() * 1000
    execution_time = end_time - start_time
    return result, execution_time


# Example usage
def my_function(dy, dx):
    # Your function code here
    return math.atan2(dy, dx)


if __name__ == "__main__":
    result, execution_time = timer(my_function, 2, -2)
    print(f"Execution time: {execution_time} seconds")
    print(result)
