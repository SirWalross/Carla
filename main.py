import argparse
import math
import random
from typing import List, Tuple
import numpy as np
import carla
import cv2
import pygame
from pygame.locals import *
from trafficsign import TrafficSignType, detect_traffic_sign, load_model

from carla import ColorConverter as cc

pygame.init()
screen = pygame.display.set_mode((1280, 720))

world = None
vehicle = None

current_speed = 0.0  # m/s
prev_sensor_data = None
steering_delta = 0
steering = 0
speed_delta = 0
throttle = 0
brake = 0
distance_waypoint = 0
waypoint = None
traffic_light = None
traffic_sign = None
detected_red_traffic_light = False
waypoint_deadzone = 0

# CONSTANTS
ROAD_WIDTH = 3.0
TRAFFIC_LIGHT_SENSITIVITY = 0.4
LIDAR_DISTANCE = 50.0


class LidarData:
    def __init__(
        self,
        lidar_data,
    ):
        self.point_cloud = self._convert_to_point_cloud(lidar_data)
        self.timestamp = lidar_data.timestamp

        # Calculate as only x-axis, should do distance in forward direction, but oh well

        # self.distances = np.linalg.norm(self.point_cloud["position"], axis=1)
        self.distances = self.point_cloud["position"][:, 0]
        self.object_indices = np.unique(self.point_cloud["object_index"])

    @staticmethod
    def valid_points(points: np.ndarray, tags: np.ndarray) -> np.ndarray:
        if np.abs(steering) <= 1e-4:
            return np.logical_and.reduce(
                np.logical_or.reduce((tags == 4, tags == 10)),
                (points[:, 0] < LIDAR_DISTANCE, points[:, 0] > 1, points[:, 1] < 2, points[:, 1] > -2),
            )
        else:
            r = 8 * (1 / np.abs(steering)) * (0.8 + throttle * 0.2)
            r1 = r - (ROAD_WIDTH / 2) * (0.8 + throttle * 0.2)
            r2 = r + (ROAD_WIDTH / 2) * (0.8 + throttle * 0.2)
            return np.logical_and.reduce(
                np.logical_or.reduce((tags == 4, tags == 10)),
                points[:, 0] > 1, 
                np.linalg.norm(points[:, :2], axis=1) >= r1,
                np.linalg.norm(points[:, :2], axis=1) <= r2,
            )

    @staticmethod
    def _convert_to_point_cloud(lidar_data):
        dtype = np.dtype(
            [("position", np.float32, (3,)), ("cos_angle", np.float32), ("object_index", np.uint32), ("tag", np.uint32)]
        )
        p_cloud = np.frombuffer(lidar_data.raw_data, dtype=dtype)
        tags = p_cloud["tag"]
        positions = np.array(p_cloud["position"])
        return p_cloud[LidarData.valid_points(positions, tags)]

    def query_object_index(self, object_index) -> Tuple[float, np.ndarray]:
        """Return the distance to the nearest point with object index `object_index` and the position"""
        object_indices = self.point_cloud["object_index"] != object_index
        distances = np.copy(self.distances)
        distances[object_indices] = np.inf
        if len(distances) > 0:
            i = np.argmin(distances)
            return distances[i], self.point_cloud["position"][i, :]
        else:
            return np.inf, self.point_cloud["position"][0, :]


obstacles: List[Tuple[float, float, np.ndarray]] = []
last_lidar_data: LidarData = None


def lidar_sensor(lidar_data):
    global obstacles, last_lidar_data
    lidar_data = LidarData(lidar_data)
    if last_lidar_data is not None:
        obstacles = []
        for object_index in lidar_data.object_indices:
            dist_new, point = lidar_data.query_object_index(object_index)
            dist_old, _ = last_lidar_data.query_object_index(object_index)
            if dist_old != np.inf and dist_new != np.inf:
                obstacles.append(
                    (dist_new, (dist_new - dist_old) / (lidar_data.timestamp - last_lidar_data.timestamp), point)
                )
        obstacles.sort(key=lambda obstacle: obstacle[0])
    if len(obstacles) > 0:
        world.debug.draw_point(
            waypoint.transform.location(*obstacles[0][2]), size=0.1, color=carla.Color(255, 0, 0), life_time=0.1
        )
    last_lidar_data = lidar_data


def rgb_sensor(image):
    global detected_red_traffic_light
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    # convert from rgb to bgr
    array = np.array(array[:, :, ::-1])

    if traffic_light is not None:
        area = cv2.contourArea(traffic_light)

        mask = np.zeros_like(array[:, :, 0])
        cv2.drawContours(mask, [traffic_light], -1, (255, 255, 255), -1)
        image = cv2.bitwise_and(array, array, mask=mask)

        # average red color in image
        average_colours = np.array([np.average(image[:, :, 0]), np.average(image[:, :, 1]), np.average(image[:, :, 2])])
        average_colours = average_colours / np.sum(average_colours)

        point = (max(traffic_light[:, 0, 0]), min(traffic_light[:, 0, 1]))
        if average_colours[0] > TRAFFIC_LIGHT_SENSITIVITY:
            cv2.drawContours(array, [traffic_light], -1, (255, 0, 0), 1)
            cv2.putText(array, f"{average_colours[0]:.2f},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        else:
            cv2.drawContours(array, [traffic_light], -1, (0, 255, 0), 1)
            cv2.putText(array, f"{average_colours[0]:.2f},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        if area > 1800 and average_colours[0] > TRAFFIC_LIGHT_SENSITIVITY:
            detected_red_traffic_light = True
        else:
            detected_red_traffic_light = False
    else:
        detected_red_traffic_light = False

    if traffic_sign is not None:
        area = cv2.contourArea(traffic_sign)

        x, y, w, h = cv2.boundingRect(traffic_sign)
        prediction = detect_traffic_sign(array[y:y+h, x:x+w])

        point = (max(traffic_sign[:, 0, 0]), min(traffic_sign[:, 0, 1]))
        cv2.drawContours(array, [traffic_sign], -1, (255, 0, 0), 1)
        cv2.putText(array, f"{prediction.name},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def segmentation_sensor(image):
    global traffic_light, traffic_sign
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # Traffic light
    traffic_image = np.copy(array)
    # preparing the mask to overlay
    mask = cv2.inRange(traffic_image, np.array([249, 169, 29]), np.array([251, 171, 31]))

    base_colour = np.full_like(traffic_image, np.array([255, 255, 255]))
    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # convert image to greyscale
    traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_light = max(contours, key=cv2.contourArea)
    else:
        traffic_light = None

    # TODO: @henry
    # Traffic sign
    traffic_image = np.copy(array)
    # preparing the mask to overlay
    mask = cv2.inRange(traffic_image, np.array([219, 219, 0]), np.array([221, 221, 1]))

    base_colour = np.full_like(traffic_image, np.array([255, 255, 255]))
    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # convert image to greyscale
    traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_sign = max(contours, key=cv2.contourArea)
    else:
        traffic_sign = None


def gnss_sensor(sensor_data):
    global prev_sensor_data, current_speed, distance_waypoint
    if prev_sensor_data is not None:
        dx = sensor_data.transform.location.distance(prev_sensor_data.transform.location)
        dt = sensor_data.timestamp - prev_sensor_data.timestamp
        current_speed = dx / dt
        prev_sensor_data = sensor_data
    else:
        prev_sensor_data = sensor_data
    if waypoint is not None:
        distance_waypoint = sensor_data.transform.location.distance(waypoint.transform.location)


def vehicle_control(waypoint_transform, vehicle_transform, target_speed) -> Tuple[float, float, float]:
    global steering_delta, speed_delta, throttle, steering, brake

    # calulate steering delta
    forward_vector = vehicle_transform.get_forward_vector()
    forward_vector = np.array([forward_vector.x, forward_vector.y, 0.0])
    waypoint_vector = np.array(
        [
            waypoint_transform.location.x - vehicle_transform.location.x,
            waypoint_transform.location.y - vehicle_transform.location.y,
            0.0,
        ]
    )
    wv_linalg = np.linalg.norm(waypoint_vector) * np.linalg.norm(forward_vector)
    if wv_linalg == 0:
        _dot = 1
    else:
        _dot = math.acos(np.clip(np.dot(waypoint_vector, forward_vector) / (wv_linalg), -1.0, 1.0))

    _cross = np.cross(forward_vector, waypoint_vector)
    if _cross[2] < 0:
        _dot *= -1.0
    steering_delta = _dot
    speed_delta = target_speed - current_speed

    if len(obstacles) > 0:
        speed_delta = min(speed_delta, obstacles[0][1] + (obstacles[0][0] - 3) / 2)

    K_P_T = 0.5
    K_P_S = 0.5

    steering = np.clip(steering_delta * K_P_S, -1, 1)
    if speed_delta > 0:
        throttle = np.clip(speed_delta * K_P_T, 0, 0.5) / (1 + abs(steering))
        brake = 0
    else:
        throttle = 0
        brake = np.clip(-speed_delta * K_P_T, 0, 1.0)

    if detected_red_traffic_light:
        return 0.0, steering, 1.0
    return throttle, steering, brake


def visualize_path():
    t = np.arange(0, LIDAR_DISTANCE / 5, 0.1)
    points = np.zeros((t.shape[0], 9))

    if np.abs(steering) <= 1e-4:
        points[:, [0, 3, 6]] = np.array([np.copy(t) * 5, np.copy(t) * 5, np.copy(t) * 5]).T
        points[:, [1, 4, 7]] = np.full_like(points[:, :3], [-ROAD_WIDTH / 2, 0, ROAD_WIDTH / 2])
    else:
        r = 8 * (1 / np.abs(steering)) * (0.8 + throttle * 0.2)
        points[:, 0] = r * np.sin(t / r * np.pi)
        points[:, 1] = -r * (np.cos(t / r * np.pi) - 1) * np.sign(steering)
        r = r - ROAD_WIDTH / 2
        points[:, 3] = r * np.sin(t / r * np.pi)
        points[:, 4] = -r * (np.cos(t / r * np.pi) - 1) * np.sign(steering) - ROAD_WIDTH / 2
        r = r + ROAD_WIDTH
        points[:, 6] = r * np.sin(t / r * np.pi)
        points[:, 7] = -r * (np.cos(t / r * np.pi) - 1) * np.sign(steering) + ROAD_WIDTH / 2

    points[:, [2, 5, 8]] = np.full_like(points[:, :3], -1.8)

    points = points.reshape((points.shape[0] * 3, 3))

    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T

    lidar_2_world = lidar.get_transform().get_matrix()

    # Transform the points from lidar space to world space.
    points = np.dot(lidar_2_world, points)

    # convert to global coordinate coordinate system

    for i in range(points.shape[1] - 3):
        world.debug.draw_line(
            carla.Location(x=points[0, i], y=points[1, i], z=points[2, i]),
            carla.Location(x=points[0, i + 3], y=points[1, i + 3], z=points[2, i + 3]),
            0.02,
            carla.Color(255, 0, 0),
            0.15,
        )


def main(ip: str):
    global waypoint, waypoint_deadzone, lidar, world, vehicle
    
    load_model()

    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")
        # world = client.get_world()
        # world.unload_map_layer(carla.MapLayer.Buildings)
        # world.unload_map_layer(carla.MapLayer.Decals)
        # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # world.unload_map_layer(carla.MapLayer.Foliage)
        # world.unload_map_layer(carla.MapLayer.Walls)
        # world.unload_map_layer(carla.MapLayer.Props)

        # Settings
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
        while True:
            try:
                spawn_point = random.choice(spawn_points)
                spawn_point.z = 10
                spawn_point.location = world.get_random_location_from_navigation()
                vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                continue
        print("Spawned vehicle")

        # RGB camera
        rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "1280")
        rgb_camera_bp.set_attribute("image_size_y", "720")
        rgb_camera_bp.set_attribute("fov", "60")
        rgb_camera_bp.set_attribute("sensor_tick", "0.02")
        relative_transform = carla.Transform(carla.Location(x=1.2, y=0, z=1.7), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(rgb_sensor)

        # Segmentation camera
        segmentation_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        segmentation_bp.set_attribute("image_size_x", "1280")
        segmentation_bp.set_attribute("image_size_y", "720")
        segmentation_bp.set_attribute("fov", "60")
        segmentation_bp.set_attribute("sensor_tick", "0.02")
        segmentation = world.spawn_actor(segmentation_bp, relative_transform, vehicle)
        segmentation.listen(segmentation_sensor)

        # Lidar sensor
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
        lidar_bp.set_attribute("upper_fov", "30.0")
        lidar_bp.set_attribute("lower_fov", "-25.0")
        lidar_bp.set_attribute("channels", "64.0")
        lidar_bp.set_attribute("range", "100.0")
        lidar_bp.set_attribute("points_per_second", "100000.0")
        lidar = world.spawn_actor(lidar_bp, relative_transform, vehicle)
        lidar.listen(lidar_sensor)

        # GNSS sensor
        gnss = blueprint_library.find("sensor.other.gnss")
        gnss = world.spawn_actor(gnss, carla.Transform(), vehicle)
        gnss.listen(gnss_sensor)

        # Lidar sensor
        lidar_transform = carla.Transform(carla.Location(z=1.9))
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("upper_fov", "30.0")
        lidar_bp.set_attribute("lower_fov", "-25.0")
        lidar_bp.set_attribute("channels", "64.0")
        lidar_bp.set_attribute("range", str(LIDAR_DISTANCE))
        lidar_bp.set_attribute("points_per_second", "100000.0")
        lidar_bp.set_attribute("rotation_frequency", "10")
        lidar = world.spawn_actor(lidar_bp, lidar_transform, vehicle)
        lidar.listen(lidar_sensor)

        waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        vehicle.set_transform(waypoint.transform)
        while 1:
            throttle, steering, brake = vehicle_control(waypoint.transform, vehicle.get_transform(), 10)
            control = carla.VehicleControl(
                throttle=throttle, steer=steering, brake=brake, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)

            visualize_path()

            pygame.display.flip()
            pygame.display.update()
            world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            if distance_waypoint <= 2 and waypoint_deadzone <= 0:
                waypoint = random.choice(waypoint.next(3))
                world.debug.draw_point(waypoint.transform.location)
                waypoint_deadzone = 2
            elif waypoint_deadzone != 0:
                waypoint_deadzone -= 1
            print(
                f"throttle: {throttle:.3f}, steering: {steering:.3f}, brake: {brake:.3f}, speed:"
                f" {current_speed:.3f} m/s, distance waypoint: {distance_waypoint:.3f}, obstacles: {obstacles}",
                end="\033[0K\r",
            )
    finally:
        try:
            vehicle.destroy()
            rgb_camera.destroy()
            segmentation.destroy()
            gnss.destroy()
        except UnboundLocalError:
            pass
        pygame.quit()
        print("\nCleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
