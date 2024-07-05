import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

polygon = Polygon([(0, 0), (0, 5), (5, 0), (5, 5)])
x, y = polygon.exterior.xy
plt.plot(x, y)
plt.show()
"""
plt.ion()
x = np.linspace(0, 1, 200)
plt.plot(x, F(x))


for i in range(100):
    if "ax" in globals():
        ax.remove()
    newx = np.random.choice(x, size=10)
    ax = plt.scatter(newx, F(newx))
    plt.pause(0.05)

plt.ioff()
# plt.show()
"""
