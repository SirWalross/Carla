import untangle
import matplotlib.pyplot as plt
import numpy as np

xodr = untangle.parse("map07.xodr")


for road in xodr.OpenDRIVE.road:
    plan_view = road.planView

    for geometry in plan_view.geometry:
        s = float(geometry["s"])
        x = float(geometry["x"])
        y = float(geometry["y"])
        hdg = float(geometry["hdg"])
        length = float(geometry["length"])
        try:
            geometry.line
            x = [x, x + np.cos(hdg) * length]
            y = [y, y + np.sin(hdg) * length]
        except AttributeError:
            curvature = float(geometry.arc["curvature"])
            phi = np.linspace(hdg + np.pi/2, hdg + np.pi/2 + 2 * np.pi * length / (2 * np.pi / curvature), 5)
            x = x + np.sin(phi) * 1 / curvature - np.sin(hdg) * 1 / curvature
            y = y + np.cos(phi) * 1 / curvature - np.cos(hdg) * 1 / curvature
        plt.plot(x, y)
plt.show()
