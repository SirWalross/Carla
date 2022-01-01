import argparse
import math
import random
from typing import List, Tuple
import numpy as np
import carla
import cv2
import pygame
from pygame.locals import *

from carla import ColorConverter as cc

from pid import PID
from trafficsign import TrafficSignType, load_model, detect_traffic_sign
from spawn_road_borders import spawn_road_borders
from generate_traffic import spawn_traffic, destroy_traffic

random.seed(42)


# CONSTANTS
WIDTH = 1280
HEIGHT = 720
ROAD_WIDTH = 3.0
TRAFFIC_LIGHT_SENSITIVITY = 0.37
LIDAR_DISTANCE = 47.0
ROAD_OFFSET = 50
BORDER = 0.1  # 10% border around image
TRAFFIC_SIGN_DETECTION_RANGE = (500, 1000)  # min and max area of sign
MAX_FRAME = 200000
FRAME_SKIP = 20

# traffic signs
speed_30_sign = cv2.imread("speed_signs/speed_30_sign.png", -1)
speed_60_sign = cv2.imread("speed_signs/speed_60_sign.png", -1)
speed_90_sign = cv2.imread("speed_signs/speed_90_sign.png", -1)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

world = None
vehicle = None

current_speed = 0.0  # m/s
prev_sensor_data = None
steering = 0
throttle = 0
brake = 0
distance_waypoint = 0
waypoint = None
traffic_light = None
traffic_sign = None
road_contour = None
detected_red_traffic_light = False
waypoint_deadzone = 0
target_speed = 30  # km/h
frame = 0
obstacles: List[Tuple[float, float, np.ndarray]] = []
last_lidar_data: "LidarData" = None

# controllers
steering_pid = PID(2.0, 0, 0, (-1, 1))
throttle_pid = PID(1.2, 0, 0, (-1, 1))

# enable/disable certain features
traffic_sign_detection = True
traffic_light_detection = True
collision_detection = True
display_image = True
write_to_file = True


class LidarData:
    def __init__(self, lidar_data):
        self.point_cloud = self._convert_to_point_cloud(lidar_data)
        self.timestamp = lidar_data.timestamp

        # FIXME Calculate as only x-axis, should do distance in forward direction

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


def lidar_sensor(lidar_data):
    global obstacles, last_lidar_data
    if collision_detection:
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
    global detected_red_traffic_light, target_speed, frame
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    # convert from rgb to bgr
    array = np.array(array[:, :, ::-1])
    raw_image = np.copy(array)

    if road_contour is not None:
        cv2.drawContours(array, [road_contour], -1, (0, 0, 0), 1)

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

        if area >= TRAFFIC_SIGN_DETECTION_RANGE[0] and area <= TRAFFIC_SIGN_DETECTION_RANGE[1]:
            x, y, w, h = cv2.boundingRect(traffic_sign)
            x1 = int(np.clip(x - w * BORDER, 0, WIDTH))
            x2 = int(np.clip(x + w * (1 + BORDER), 0, WIDTH))
            y1 = int(np.clip(y - h * BORDER, 0, HEIGHT))
            y2 = int(np.clip(y + h * (1 + BORDER), 0, HEIGHT))
            prediction = detect_traffic_sign(raw_image[y1:y2, x1:x2, :])

            point = (max(traffic_sign[:, 0, 0]), min(traffic_sign[:, 0, 1]))
            color = (255, 255, 255) if prediction != TrafficSignType.INVALID_SIGN else (255, 0, 0)
            cv2.drawContours(array, [traffic_sign], -1, color, 1)
            cv2.putText(array, f"{prediction.name},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

            if prediction == TrafficSignType.SPEED_30_SIGN:
                target_speed = 30
            elif prediction == TrafficSignType.SPEED_60_SIGN:
                target_speed = 60
            elif prediction == TrafficSignType.SPEED_90_SIGN:
                target_speed = 90

    # overlay current speed sign
    current_sign = speed_30_sign if target_speed == 30 else (speed_60_sign if target_speed == 60 else speed_90_sign)
    x1, x2 = 0, current_sign.shape[1]
    y1, y2 = HEIGHT - current_sign.shape[0], HEIGHT
    array[y1:y2, x1:x2] = current_sign[:, :, 2::-1]

    # display image
    if display_image:
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
    if write_to_file:
        if frame % FRAME_SKIP == 0:
            cv2.imwrite(f"images/traffic{frame}.png", array[:, :, ::-1])
    frame += 1


def segmentation_sensor(image):
    global traffic_light, traffic_sign, road_contour
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # Road Outline
    road_image = np.copy(array)
    mask_road_line = cv2.inRange(road_image, np.array([156, 233, 49]), np.array([158, 235, 51]))
    mask_road = cv2.inRange(road_image, np.array([127, 63, 127]), np.array([129, 65, 129]))
    mask = cv2.bitwise_or(mask_road_line, mask_road)

    base_colour = np.full_like(road_image, np.array([100, 100, 100]))
    road_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # create a zero array
    stencil = np.zeros_like(road_image[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[200, HEIGHT], [200, HEIGHT - 200], [WIDTH - 200, HEIGHT - 200], [WIDTH - 200, HEIGHT]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 255)

    # apply stencil
    road_image = cv2.bitwise_and(road_image, road_image, mask=stencil)
    road_image = cv2.cvtColor(road_image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(road_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        road_contour = max(contours, key=cv2.contourArea)
    else:
        road_contour = None

    # Traffic light
    if traffic_light_detection:
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

    # Traffic sign
    if traffic_sign_detection:
        traffic_image = np.copy(array)
        # preparing the mask to overlay
        mask = cv2.inRange(traffic_image, np.array([219, 219, 0]), np.array([221, 221, 1]))

        base_colour = np.full_like(traffic_image, np.array([255, 255, 255]))
        traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

        # create a zero array
        stencil = np.zeros_like(traffic_image[:, :, 0])

        # specify coordinates of the polygon
        polygon = np.array([[int(WIDTH / 2), 200], [int(WIDTH / 2), HEIGHT], [WIDTH, HEIGHT], [WIDTH, 200]])
        cv2.fillConvexPoly(stencil, polygon, 255)

        # convert image to greyscale
        traffic_image = cv2.bitwise_and(traffic_image, traffic_image, mask=stencil)
        traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

        # contour detection
        contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            traffic_sign = max(contours, key=cv2.contourArea)
            # cv2.drawContours(traffic_image, [traffic_sign], -1, (255, 0, 0), 1)
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


def vehicle_control() -> Tuple[float, float, float]:

    # calulate steering value
    if road_contour is not None:
        moment = cv2.moments(road_contour)
        cX = int(moment["m10"] / moment["m00"])

        diff = cX - WIDTH / 2 + ROAD_OFFSET
        steering = steering_pid(diff / 400, 0)
    else:
        steering = 0

    # calculate speed delta
    speed_delta = target_speed / 3.6 - current_speed

    # detected obstacle in path
    if len(obstacles) > 0:
        speed_delta = min(speed_delta, obstacles[0][1] + (obstacles[0][0] - 3) / 2)

    # calculate throttle and brake from speed_delta
    throttle = throttle_pid(speed_delta, 0, (0, 1 / (1 + abs(steering))))
    brake = throttle if throttle < 0 else 0
    throttle = throttle if throttle > 0 else 0

    if brake > 0:
        steering = 0

    # detected red traffic light
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


def main(
    ip: str,
    enable_path_visualization: bool,
    env_information: bool,
    road_borders: bool,
    generate_traffic: bool,
    number_of_vehicles: int,
    number_of_walkers: int,
):
    global waypoint, waypoint_deadzone, lidar, world, vehicle

    if traffic_sign_detection:
        load_model()

    client = carla.Client(ip, 2000)
    try:
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
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        spawn_point = carla.Transform(carla.Location(-5.38, 280.0, 1.0), carla.Rotation(yaw=90.0))
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Spawned vehicle")

        if road_borders:
            spawn_road_borders(world, blueprint_library)

        if generate_traffic:
            spawn_traffic(client, number_of_vehicles, number_of_walkers)

        # RGB camera
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", "1280")
        rgb_bp.set_attribute("image_size_y", "720")
        rgb_bp.set_attribute("fov", "60")
        rgb_bp.set_attribute("sensor_tick", "0.02")
        relative_transform = carla.Transform(carla.Location(x=1.2, y=0, z=1.7), carla.Rotation())
        rgb = world.spawn_actor(rgb_bp, relative_transform, vehicle)
        rgb.listen(rgb_sensor)

        # Segmentation camera
        segmentation_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        segmentation_bp.set_attribute("image_size_x", "1280")
        segmentation_bp.set_attribute("image_size_y", "720")
        segmentation_bp.set_attribute("fov", "60")
        segmentation_bp.set_attribute("sensor_tick", "0.02")
        segmentation = world.spawn_actor(segmentation_bp, relative_transform, vehicle)
        segmentation.listen(segmentation_sensor)

        # GNSS sensor
        gnss = blueprint_library.find("sensor.other.gnss")
        gnss = world.spawn_actor(gnss, carla.Transform(), vehicle)
        gnss.listen(gnss_sensor)

        # Lidar sensor
        if collision_detection:
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

        while frame < MAX_FRAME:
            throttle, steering, brake = vehicle_control()
            control = carla.VehicleControl(
                throttle=throttle, steer=steering, brake=brake, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)

            if enable_path_visualization:
                visualize_path()

            pygame.display.flip()
            pygame.display.update()
            world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            if env_information:
                print(
                    f"throttle: {throttle:.3f}, steering: {steering:.3f}, brake: {brake:.3f}, speed:"
                    f" {current_speed * 3.6:.3f} km/h, target speed {target_speed:3f} km/h, obstacles: {obstacles}",
                    end="\033[0K\r",
                )
    finally:
        if generate_traffic:
            destroy_traffic(client)

        try:
            gnss.destroy()
            segmentation.destroy()
            rgb.destroy()
            lidar.destroy()
            vehicle.destroy()
        except UnboundLocalError:
            pass
        except NameError:
            pass

        pygame.quit()
        print("\nCleaned up")
        quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    parser.add_argument("--visualize_path", nargs="?", default=False, help="Enable path visualization")
    parser.add_argument("--collision_detection", nargs="?", default=False, help="Enable collision detection")
    parser.add_argument("--traffic_sign_detection", nargs="?", default=True, help="Enable detection of traffic signs")
    parser.add_argument("--traffic_light_detection", nargs="?", default=True, help="Enable detection of traffic lights")
    parser.add_argument("--env_information", nargs="?", default=True, help="Wether to print enviroment information")
    parser.add_argument("--spawn_road_borders", nargs="?", default=True, help="Wether to spawn road borders")
    parser.add_argument("--spawn_traffic", nargs="?", default=True, help="Wether to spawn traffic")
    parser.add_argument("--number-of-vehicles", default=30, type=int, help="Number of vehicles (default: 30)")
    parser.add_argument("--number-of-walkers", default=10, type=int, help="Number of walkers (default: 10)")
    parser.add_argument("--write_to_file", nargs="?", default=False, help="Enable writing of image to file")
    parser.add_argument("--display_image", nargs="?", default=True, help="Enable displaying of image")
    args = parser.parse_args()
    collision_detection = args.collision_detection
    traffic_sign_detection = args.traffic_sign_detection
    traffic_light_detection = args.traffic_light_detection
    write_to_file = args.write_to_file
    display_image = args.display_image
    main(
        args.host,
        args.visualize_path,
        args.env_information,
        args.spawn_road_borders,
        args.spawn_traffic,
        args.number_of_vehicles,
        args.number_of_walkers,
    )
