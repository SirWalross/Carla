from time import time
from typing import Dict, List, Optional, Tuple
import untangle
import matplotlib.pyplot as plt
import numpy as np
import sys


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

    def next_lane(self, lane: "Lane") -> Optional["Lane"]:
        assert lane.lane_section.road == self
        travelDir = lane.direction
        lanes = [lane_section.lanes[lane.id] for lane_section in list(self.lane_sections.values())]

        index = lanes.index(lane)
        if travelDir == "forward":
            if len(lanes) - 1 != index:
                return lanes[index + 1]
            else:
                return None
        else:
            if index != 0:
                return lanes[index - 1]
            else:
                return None


class LaneLink:
    def __init__(self, from_id: int, to_id: int):
        self.from_id = from_id
        self.to_id = to_id


class Connection:
    def __init__(self, id: int, incomingRoad: int, connectingRoad: int, contactPoint: str):
        self.id = id
        self.incomingRoad = incomingRoad
        self.connectionRoad = connectingRoad
        self.contactPoint = contactPoint
        self.lane_links: Dict[int, LaneLink] = {}


class Junction:
    def __init__(self, id: int):
        self.id = id
        self.connections: Dict[int, Connection] = {}


class RefLine:
    def __init__(self, length: float) -> None:
        self.length = length
        self.geometries: Dict[float, RoadGeometry] = {}  # from s0 to road geometry


class RoadLink:
    def __init__(self, elementId: int, elementType: str, contactPoint: str) -> None:
        self.id = elementId
        self.type = elementType
        self.contactPoint = contactPoint


class RoadGeometry:
    def __init__(self, s: float, x: float, y: float, hdg: float, length: float) -> None:
        self.s = s
        self.x = x
        self.y = y
        self.hdg = hdg
        self.length = length

    def get_xy(self, s: float) -> Tuple[float, float]:
        raise NotImplementedError()

    def get_grad(self, s: float) -> Tuple[float, float]:
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

    def get_grad(self, s: float) -> Tuple[float, float]:
        return (
            np.sin((np.pi / 2) - self.curvature * (s - self.s) - self.hdg),
            np.cos((np.pi / 2) - self.curvature * (s - self.s) - self.hdg),
        )


class Line(RoadGeometry):
    def __init__(self, s: float, x: float, y: float, hdg: float, length: float) -> None:
        super().__init__(s, x, y, hdg, length)

    def get_xy(self, s: float) -> Tuple[float, float]:
        return (np.cos(self.hdg) * (s - self.s) + self.x, np.sin(self.hdg) * (s - self.s) + self.y)

    def get_grad(self, s: float) -> Tuple[float, float]:
        return (np.cos(self.hdg), np.sin(self.hdg))


class LaneSection:
    def __init__(self, s: float, length: float) -> None:
        self.s = s
        self.road: Road = None
        self.lanes: Dict[int, Lane] = {}  # from id to lane
        self.length = length

    def find_lanes(self, id: int) -> List["Lane"]:
        lanes: List[Lane] = []

        for lane_id, lane in self.lanes.items():
            if np.sign(lane_id) == np.sign(id):
                lanes.append(lane)

        return lanes


class Lane:
    def __init__(self, id: int, level: bool, type: str, direction: str) -> None:
        self.id = id
        self.level = level
        self.type = type
        self.direction = direction
        self.road: Road = None
        self.widths: List[Dict[float, Poly3]] = []  # list of widths of the current and previous lanes
        self.predecessor: int = -1
        self.successor: int = -1
        self.lane_section: LaneSection = None
    
    def get_width(self, s: np.ndarray, middle_lane: bool = False) -> np.ndarray:
        width = np.zeros_like(s)
        if self.id > 0:
            for i, s0 in enumerate(s):
                width[i] = sum(
                    [
                        width[
                            list(width.keys())[np.searchsorted(list(width.keys()), s0, side="right") - 1]
                        ].get(s0) * (0.5 if middle_lane and (j == len(self.widths) - 1) else 1)
                        for j, width in enumerate(self.widths)
                    ]
                )
        else:
            for i, s0 in enumerate(s):
                width[i] = sum(
                    [
                        width[list(width.keys())[np.searchsorted(list(width.keys()), s0, side="right") - 1]]
                        .negate()
                        .get(s0) * (0.5 if middle_lane and (j == len(self.widths) - 1) else 1)
                        for j, width in enumerate(self.widths)
                    ]
                )
        return width


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

    def negate(self) -> "Poly3":
        return Poly3(self.s, -self.a, -self.b, -self.c, -self.d)


class OpenDriveMap:
    def __init__(self, xodr_file: str):
        reverse = lambda l: l[::-1] if isinstance(l, list) else l
        self.possible_paths: List[List[Lane]] = []

        self.xodr = untangle.parse(xodr_file)

        self.roads: List[Road] = []
        for road_node in self.xodr.OpenDRIVE.road:
            road = Road(float(road_node["length"]), int(road_node["id"]), int(road_node["junction"]), road_node["name"])

            # parse road links
            for link in ["predecessor", "successor"]:
                try:
                    road_link_node = getattr(road_node.link, link)
                    road_link = RoadLink(
                        int(road_link_node["elementId"]), road_link_node["elementType"], road_link_node["contactPoint"]
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
            for i, lane_section_node in enumerate(road_node.lanes.laneSection):
                s = float(lane_section_node["s"])
                lane_section = LaneSection(s, road.length - s)

                if i != 0:
                    old_lane_section = road.lane_sections[list(road.lane_sections.keys())[-1]]
                    old_lane_section.length = s - old_lane_section.s

                lane_section.road = road
                road.lane_sections[s] = lane_section

                lane_nodes = []
                try:
                    lane_nodes.extend(reverse(lane_section_node.left.lane))
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

                poly: Tuple[int, List[Dict[float, Poly3]]] = (1, [])

                for lane_node in lane_nodes:
                    lane_id = int(lane_node["id"])
                    lane_type = lane_node["type"]
                    level = bool(lane_node["level"])

                    if lane_type != "driving":
                        continue
                    direction = (
                        lane_node.userData.vectorLane[0]["travelDir"]
                        if isinstance(lane_node.userData.vectorLane, list)
                        else lane_node.userData.vectorLane["travelDir"]
                    )
                    lane = Lane(lane_id, level, lane_type, direction)
                    lane.road = road
                    lane_section.lanes[lane_id] = lane
                    lane.lane_section = lane_section

                    if np.sign(lane_id) != np.sign(poly[0]):
                        assert np.sign(lane_id) == -1
                        poly = (-1, [])

                    try:
                        widths: Dict[float, Poly3] = {}
                        for lane_width_node in lane_node.width:
                            s_offset = float(lane_width_node["sOffset"])
                            a = float(lane_width_node["a"])
                            b = float(lane_width_node["b"])
                            c = float(lane_width_node["c"])
                            d = float(lane_width_node["d"])
                            widths[s + s_offset] = Poly3(s + s_offset, a, b, c, d)
                        poly[1].append(widths)
                        lane.widths = [*poly[1]]
                    except AttributeError:
                        pass

                    try:
                        lane.predecessor = int(lane_node.link.predecessor["id"])
                    except AttributeError:
                        pass
                    try:
                        lane.successor = int(lane_node.link.successor["id"])
                    except AttributeError:
                        pass
            if all([not bool(lane_section.lanes) for _, lane_section in road.lane_sections.items()]):
                continue
            self.roads.append(road)

        self.junctions: List[Junction] = []
        for junction_node in self.xodr.OpenDRIVE.junction:
            junction = Junction(int(junction_node["id"]))

            # parse connections
            for connection_node in junction_node.connection:
                id = int(connection_node["id"])
                connection = Connection(
                    id,
                    int(connection_node["incomingRoad"]),
                    int(connection_node["connectingRoad"]),
                    connection_node["contactPoint"],
                )

                # parse lane links
                for lane_link_node in connection_node.laneLink:
                    lane_link = LaneLink(int(lane_link_node["from"]), int(lane_link_node["to"]))

                    connection.lane_links[lane_link.from_id] = lane_link

                junction.connections[id] = connection

            self.junctions.append(junction)

    def _highlight_lanes(self, lanes: List[Lane], color="orange", eps: float = 0.1):
        roads = [lane.lane_section.road for lane in lanes]
        # highlight lanes
        for road in roads:
            ref_line_xy = [[], [], [], [], []]  # x, y, dx, dy
            for _, geometry in road.ref_line.geometries.items():
                s = np.arange(geometry.s, geometry.s + geometry.length, eps)
                get_xy = np.vectorize(geometry.get_xy)
                get_grad = np.vectorize(geometry.get_grad)
                xy = get_xy(s)
                grad = get_grad(s)
                ref_line_xy[0].extend(s)
                ref_line_xy[1].extend(xy[0])
                ref_line_xy[2].extend(xy[1])
                ref_line_xy[3].extend(grad[0])
                ref_line_xy[4].extend(grad[1])

                if isinstance(geometry, Arc):
                    pass
            ref_line_xy = np.array(ref_line_xy)
            for _, lane_section in road.lane_sections.items():
                s = np.arange(lane_section.s, lane_section.s + lane_section.length, eps)
                keys = list(road.offset.keys())
                offset = [road.offset[keys[np.searchsorted(keys, s0, side="right") - 1]].get(s0) for s0 in s]
                ref_line_section_xy = np.array(
                    [ref_line_xy[1:, np.searchsorted(ref_line_xy[0], s0, side="right") - 1] for s0 in s]
                )
                for _, lane in lane_section.lanes.items():
                    if lane not in lanes:
                        continue
                    s = np.arange(0, lane_section.length, eps)
                    width = lane.get_width(s, middle_lane=True)
                    xy = np.array(
                        [
                            (
                                ref_line_section_xy[i, 0] - ref_line_section_xy[i, 3] * width[i],
                                ref_line_section_xy[i, 1] + ref_line_section_xy[i, 2] * width[i],
                            )
                            for i, _ in enumerate(s)
                        ]
                    )
                    plt.plot(xy[:, 0], xy[:, 1], c=color)

    def find_route(self, road1: str, section1: int, lane1: int, road2: str, section2: int, lane2: int):
        road1: Road = next(road for road in self.roads if road.name == road1)
        section1: LaneSection = list(road1.lane_sections.values())[section1]
        lane1: Lane = next(lane for lane in section1.lanes.values() if lane.id == lane1)
        road2: Road = next(road for road in self.roads if road.name == road2)
        section2: LaneSection = list(road2.lane_sections.values())[section2]
        lane2: Lane = next(lane for lane in section2.lanes.values() if lane.id == lane2)

        curr = time()
        try:
            self._dfs([], lane1, [lane1], lane2)
        except ValueError:
            if len(self.possible_paths) == 0:
                raise ValueError("Didnt find possible path")
        print(f"Search for route took {time() - curr:.2f}s, with {len(self.possible_paths)} possible routes")
        shortest_path = sorted(self.possible_paths, key=lambda path: sum([lane.lane_section.length for lane in path]))[
            0
        ]

        self._highlight_lanes(shortest_path)
        self._highlight_lanes([lane1, lane2], color="black")

    def _dfs(self, visited_lanes: List[Lane], current_lane: Lane, path: List[Lane], goal: Lane):
        if len(self.possible_paths) > 1000:
            raise ValueError("Found enough paths")
        current_road = current_lane.lane_section.road

        next_lanes = []
        next_lane = current_road.next_lane(current_lane)
        if next_lane is None:
            # already at last lane_section
            link = "successor" if current_lane.direction == "forward" else "predecessor"
            next_road_link = getattr(current_road, link)
            if next_road_link is None:
                raise ValueError("Dead end")
            elif next_road_link.type == "road":
                # next road is of type road
                next_road = self._find_road(next_road_link.id)

                if (next_road_link.contactPoint != "start") != (link != "successor"):
                    # flip next road
                    next_lanes.extend(list(next_road.lane_sections.values())[-1].find_lanes(-current_lane.id))
                else:
                    next_lanes.extend(next_road.lane_sections[0].find_lanes(current_lane.id))
            else:
                # next road is of type junction
                junction = self._find_junction(next_road_link.id)

                # find next roads
                next_roads = []
                for connection in junction.connections.values():
                    if connection.incomingRoad == current_road.id:
                        try:
                            next_roads.append((self._find_road(connection.connectionRoad), connection.lane_links))
                        except StopIteration:
                            continue

                for next_road, lane_links in next_roads:
                    # find next lane
                    try:
                        next_lanes.append(next_road.lane_sections[0].lanes[lane_links[current_lane.id].to_id])
                    except KeyError:
                        # lane_link didnt connect the current_lane id to another road
                        pass
        else:
            next_lanes.append(next_lane)

        for next_lane in next_lanes:
            if goal == next_lane:
                self.possible_paths.append([*path, next_lane])
            if next_lane in path:
                continue
            else:
                try:
                    new_path = self._dfs([*visited_lanes, next_lane], next_lane, [*path, next_lane], goal)
                    return new_path
                except ValueError:
                    continue
        raise ValueError("No more lanes to explore")

    def _find_road(self, roadId: int) -> Road:
        return next(road for road in self.roads if road.id == roadId)

    def _find_junction(self, junctionId: int) -> Junction:
        return next(junction for junction in self.junctions if junction.id == junctionId)

    def render(self, eps: float = 0.1):
        # plt.ion()
        # plt.show()
        for road in self.roads:
            ref_line_xy = [[], [], [], [], []]  # x, y, dx, dy
            for _, geometry in road.ref_line.geometries.items():
                s = np.arange(geometry.s, geometry.s + geometry.length, eps)
                get_xy = np.vectorize(geometry.get_xy)
                get_grad = np.vectorize(geometry.get_grad)
                xy = get_xy(s)
                grad = get_grad(s)
                ref_line_xy[0].extend(s)
                ref_line_xy[1].extend(xy[0])
                ref_line_xy[2].extend(xy[1])
                ref_line_xy[3].extend(grad[0])
                ref_line_xy[4].extend(grad[1])

                if isinstance(geometry, Arc):
                    pass
            plt.plot(ref_line_xy[1], ref_line_xy[2], "b")
            ref_line_xy = np.array(ref_line_xy)
            for _, lane_section in road.lane_sections.items():
                s = np.arange(lane_section.s, lane_section.s + lane_section.length, eps)
                keys = list(road.offset.keys())
                offset = [road.offset[keys[np.searchsorted(keys, s0, side="right") - 1]].get(s0) for s0 in s]
                ref_line_section_xy = np.array(
                    [ref_line_xy[1:, np.searchsorted(ref_line_xy[0], s0, side="right") - 1] for s0 in s]
                )
                for _, lane in lane_section.lanes.items():
                    s = np.arange(0, lane_section.length, eps)
                    width = lane.get_width(s)
                    xy = np.array(
                        [
                            (
                                ref_line_section_xy[i, 0] - ref_line_section_xy[i, 3] * width[i],
                                ref_line_section_xy[i, 1] + ref_line_section_xy[i, 2] * width[i],
                            )
                            for i, _ in enumerate(s)
                        ]
                    )
                    plt.plot(xy[:, 0], xy[:, 1], "g")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        plt.ion()
    plt.show()
    open_drive_map = OpenDriveMap("OpenDriveMaps/map07.xodr")
    open_drive_map.render()
    open_drive_map.find_route("Road 0", 0, -1, "Road 11", 0, -1)
    plt.show()
