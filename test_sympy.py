from sympy import Ray, Point, Segment, Segment2D, pi, FiniteSet, EmptySet

p1, p2, p3 = Point(0, 0), Point(2, 5), Point(5, 6)
r = Ray(p1, angle=pi / 2)

print(isinstance(r.intersect(Segment(p3, p2)), type(EmptySet)))
