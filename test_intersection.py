import math


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def find_intersection(x1, y1, q, x2, y2, x3, y3):
    # Convert angle q to radians
    q = math.radians(q)

    # Compute direction of the ray
    dx_ray = math.cos(q)
    dy_ray = math.sin(q)

    # Compute direction of the line segment
    dx_line = x3 - x2
    dy_line = y3 - y2

    # Calculate the determinants
    det = dx_ray * dy_line - dy_ray * dx_line
    if det == 0:
        return None  # Parallel lines, no intersection

    t = ((x2 - x1) * dy_line - (y2 - y1) * dx_line) / det
    s = ((x2 - x1) * dy_ray - (y2 - y1) * dx_ray) / det

    # Check if the intersection is within the bounds
    if t >= 0 and 0 <= s <= 1:
        # Compute the intersection point
        intersection_x = x1 + t * dx_ray
        intersection_y = y1 + t * dy_ray
        return (
            intersection_x,
            intersection_y,
            calculate_distance(x1, y1, intersection_x, intersection_y),
        )
    else:
        return None  # No valid intersection


# Example usage
x1, y1 = 10, 0
q = 90  # in degrees
x2, y2 = 0, 0
x3, y3 = 100, 100

intersection = find_intersection(x1, y1, q, x2, y2, x3, y3)
print(f"Intersection: {intersection}")
