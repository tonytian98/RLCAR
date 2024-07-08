import math

p2 = (2, 0)
p1 = (0, 2)
print(
    math.degrees(
        math.atan2(
            (p1[1] - p2[1]),
            (p1[0] - p2[0]),
        )
    )
    % 360
)

l = [f + "ddd" for f in []]
print(l)
