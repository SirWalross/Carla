from typing import Dict, List, Optional, Tuple
from numpy.lib.index_tricks import s_
import untangle
import matplotlib.pyplot as plt
import numpy as np


class OpenDriveMap:
    def __init__(self, xodr_file: str):

        self.xodr = untangle.parse(xodr_file)

        self.roads: List[Road] = []
        for road_node in self.xodr.OpenDRIVE.road:
            road = Road(float(road_node["length"]), int(road_node["id"]), int(road_node["junction"]), road_node["name"])

            # parse road links
            for link in ["predecessor", "successor"]:
                try:
                    road_link_node = getattr(road_node.link, link)
                    road_link = RoadLink(
                        road_link_node["elementId"], road_link_node["elementType"], road_link_node["contactPoint"]
                    )
                    setattr(road, link, road_link)
                except AttributeError:
                    pass

            # parse road geometries
            for geometry_hdr_node in road_node.planView.geometry:
                s = float(geometry_hdr_node["s"])
                x = float(geometry_hdr_node["x"])
                y = float(geometry_hdr_node["y"])
                hdg = float(geometry_hdr_node["hdg"])
                length = float(geometry_hdr_node["length"])

                try:
                    geometry_node = geometry_hdr_node.line
                    road_geometry = Line(s, x, y, hdg, length)
                except AttributeError:
                    geometry_node = geometry_hdr_node.arc
                    curvature = float(geometry_node["curvature"])
                    road_geometry = Arc(s, x, y, hdg, length, curvature)

                road.ref_line.geometries[s] = road_geometry

            # parse lane offsets
            for lane_offset_node in road_node.lanes.laneOffset:
                s = float(lane_offset_node["s"])
                a = float(lane_offset_node["a"])
                b = float(lane_offset_node["b"])
                c = float(lane_offset_node["c"])
                d = float(lane_offset_node["d"])
                road.offset[s] = Poly3(s, a, b, c, d)

            # parse road line sections and lanes
            for lane_section_node in road_node.lanes.laneSection:
                s = float(lane_section_node["s"])
                lane_section = LaneSection(s)
                lane_section.road = road
                road.lane_sections[s] = lane_section

                lane_nodes = []
                try:
                    lane_nodes.extend(lane_section_node.left.lane)
                except AttributeError:
                    pass
                try:
                    lane_nodes.extend(lane_section_node.center.lane)
                except AttributeError:
                    pass
                try:
                    lane_nodes.extend(lane_section_node.right.lane)
                except AttributeError:
                    pass

                for lane_node in lane_nodes:
                    lane_id = int(lane_node["id"])
                    lane_type = lane_node["type"]
                    level = bool(lane_node["level"])

                    if lane_type != "driving":
                        continue
                    lane = Lane(lane_id, level, lane_type)
                    lane.road = road
                    lane_section.lanes[lane_id] = lane

                    try:
                        for lane_width_node in lane_node.width:
                            s_offset = float(lane_width_node["sOffset"])
                            a = float(lane_width_node["a"])
                            b = float(lane_width_node["b"])
                            c = float(lane_width_node["c"])
                            d = float(lane_width_node["d"])
                            lane.widths[s + s_offset] = Poly3(s + s_offset, a, b, c, d)
                    except AttributeError:
                        pass

                    try:
                        lane.predecessor = lane_node.link.predecessor["id"]
                    except AttributeError:
                        pass
                    try:
                        lane.successor = lane_node.link.successor["id"]
                    except AttributeError:
                        pass
            if all([not bool(lane_section.lanes) for _, lane_section in road.lane_sections.items()]):
                continue
            self.roads.append(road)

    def render(self, eps: float = 0.1):
        for road in self.roads:
            ref_line_xy = ([], [])
            for _, geometry in road.ref_line.geometries.items():
                s = np.arange(geometry.s, geometry.s + geometry.length, eps)
                vec = np.vectorize(geometry.get_xy)
                xy = vec(s)
                ref_line_xy[0].extend(xy[0])
                ref_line_xy[1].extend(xy[1])
            plt.plot(ref_line_xy[0], ref_line_xy[1], "b")
            for s, lane_section in road.lane_sections.items():
                for id, lane in lane_section.lanes.items():
                    pass
        plt.show()


class Road:
    def __init__(self, length: float, id: int, junction: int, name: str) -> None:
        self.length = length
        self.id = id
        self.junction = junction
        self.name = name
        self.predecessor: Optional[RoadLink] = None
        self.successor: Optional[RoadLink] = None
        self.ref_line = RefLine(self.length)
        self.lane_sections: Dict[float, LaneSection] = {}  # from s0 to lane section
        self.offset: Dict[float, Poly3] = {}  # from s to poly3


class RefLine:
    def __init__(self, length: float) -> None:
        self.length = length
        self.geometries: Dict[float, RoadGeometry] = {}  # from s0 to road geometry


class RoadLink:
    def __init__(self, elementId: str, elementType: str, contactPoint: str) -> None:
        self.id = elementId
        self.side = elementType
        self.direction = contactPoint


class RoadGeometry:
    def __init__(self, s: float, x: float, y: float, hdg: float, length: float) -> None:
        self.s = s
        self.x = x
        self.y = y
        self.hdg = hdg
        self.length = length

    def get_xy(self, s: float) -> Tuple[float, float]:
        raise NotImplementedError()


class Arc(RoadGeometry):
    def __init__(self, s: float, x: float, y: float, hdg: float, length: float, curvature: float) -> None:
        super().__init__(s, x, y, hdg, length)
        self.curvature = curvature

    def get_xy(self, s: float) -> Tuple[float, float]:
        angle_at_s = (s - self.s) * self.curvature - np.pi / 2
        r = 1 / self.curvature
        xs = r * (np.cos(self.hdg + angle_at_s) - np.sin(self.hdg)) + self.x
        ys = r * (np.sin(self.hdg + angle_at_s) + np.cos(self.hdg)) + self.y
        return (xs, ys)


class Line(RoadGeometry):
    def __init__(self, s: float, x: float, y: float, hdg: float, length: float) -> None:
        super().__init__(s, x, y, hdg, length)

    def get_xy(self, s: float) -> Tuple[float, float]:
        return (np.cos(self.hdg) * (s - self.s) + self.x, np.sin(self.hdg) * (s - self.s) + self.y)


class LaneSection:
    def __init__(self, s: float) -> None:
        self.s = s
        self.road: Road = None
        self.lanes: Dict[int, Lane] = {}  # from id to lane


class Lane:
    def __init__(self, id: int, level: bool, type: str) -> None:
        self.id = id
        self.level = level
        self.type = type
        self.road: Road = None
        self.widths: Dict[float, Poly3] = {}  # from s to poly3
        self.predecessor: int = -1
        self.successor: int = -1


class Poly3:
    def __init__(self, s: float, a: float, b: float, c: float, d: float) -> None:
        self.s = s
        self.a = a - b * s + c * s * s - d * s * s * s
        self.b = b - 2 * c * s + 3 * d * s * s
        self.c = c - 3 * d * s
        self.d = d

    def get(self, s: float):
        return self.a + self.b * s + self.c * s * s + self.d * s * s * s

    def get_grad(self, s: float):
        return self.b + 2 * self.c * s + 3 * self.d * s * s

    # def approximate_linear(self, eps: float, s_start: float, s_end: float) -> List[float]:
    #     if s_start == s_end:
    #         return []


# plt.show()

if __name__ == "__main__":
    open_drive_map = OpenDriveMap("map07.xodr")
    open_drive_map.render()
