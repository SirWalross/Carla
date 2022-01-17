import argparse
import math
import random
from typing import List, Optional, Tuple
import numpy as np
import carla
import cv2
import threading
import time
import traceback


from carla import ColorConverter as cc

from pid import PID
from trafficsign import TrafficSignType, load_model, detect_traffic_sign

random.seed(42)


# CONSTANTS
WIDTH = 1280
"""width of the image sensors"""
HEIGHT = 720
"""height of the image sensors"""
ROAD_WIDTH = 3.0
"""width of a lane for collision detection"""
TRAFFIC_LIGHT_SENSITIVITY = 0.37
""" average red color for traffic light to be classified as red"""
LIDAR_DISTANCE = 47.0
"""maximum distance for lidar sensor"""
ROAD_OFFSET = 50
"""offset from the middle of the road for right turn steering"""
ROAD_OFFSET_LEFT = 175
"""offset from the middle of the road for left turn steering"""
BORDER = 0.1
"""border around traffic sign image in percent"""
TRAFFIC_SIGN_DETECTION_RANGE = (1000, 2000)
"""min and max area of sign"""
MAX_FRAME = 20000
"""maximum number of frames to run agent for"""
FRAME_SKIP = 50
"""number of frames to skip before outputting one"""
ROAD_SIGN_DETECTION_RANGE = (WIDTH / 3, WIDTH * 2 / 3)
"""values for classifying road signs in left, middle or right"""
ROAD_SIGN_DETECTION_AREA = (7000, 12000, 7000)
"""min areas of road signs for left, middle and right"""
ROAD_DETECTION_AREA = 40000
"""minium area of road to be classifyied as an right turn"""
FRAME_COOLDOWN = 65
"""number of frames to be in a road sign mode"""

# traffic signs
SPEED_30_SIGN = cv2.imread("speed_signs/speed_30_sign.png", -1)
SPEED_60_SIGN = cv2.imread("speed_signs/speed_60_sign.png", -1)
SPEED_90_SIGN = cv2.imread("speed_signs/speed_90_sign.png", -1)

screen = None

world = None
vehicle = None

current_speed = 0.0  # m/s
prev_sensor_data = None
steering = 0
throttle = 0
brake = 0
traffic_light = None
traffic_sign = None
road_sign_case = None
road_contour = None
detected_red_traffic_light = False
target_speed = 30  # km/h
frame = 0
current_cooldown = 0
obstacles: List[Tuple[float, float, np.ndarray]] = []
last_lidar_data: "LidarData" = None

# threading
render_barrier = threading.Barrier(4)
tick_event = threading.Event()

# controllers
steering_pid = PID(1.8, 0, 0, (-1, 1))
throttle_pid = PID(0.8, 0, 0, (-1, 1))

# enable/disable certain features
traffic_sign_detection = True
traffic_light_detection = True
collision_detection = True
display_image = True
write_to_file = True


class LidarData:
    def __init__(self, lidar_data):
        """Converts the lidar data to a point cloud.

        Args:
            lidar_data (carla.SemanticLidarMeasurement): The lidar data from the sensor.
        """

        self.point_cloud = self._convert_to_point_cloud(lidar_data)
        self.timestamp = lidar_data.timestamp

        # FIXME Calculate as only x-axis, should do distance in forward direction

        # self.distances = np.linalg.norm(self.point_cloud["position"], axis=1)
        self.distances = self.point_cloud["position"][:, 0]
        self.object_indices = np.unique(self.point_cloud["object_index"])

    @staticmethod
    def _valid_points(points: np.ndarray, tags: np.ndarray) -> np.ndarray:
        if np.abs(steering) <= 1e-4:
            return np.logical_and.reduce(
                [
                    np.logical_or.reduce((tags == 4, tags == 10)),
                    points[:, 0] < LIDAR_DISTANCE,
                    points[:, 0] > 1,
                    points[:, 1] < ROAD_WIDTH / 2,
                    points[:, 1] > -ROAD_WIDTH / 2,
                ]
            )
        else:
            r = 8 * (1 / np.abs(steering)) * (0.8 + throttle * 0.2)
            r1 = r - (ROAD_WIDTH / 2) * (0.8 + throttle * 0.2)
            r2 = r + (ROAD_WIDTH / 2) * (0.8 + throttle * 0.2)
            return np.logical_and.reduce(
                [
                    np.logical_or.reduce((tags == 4, tags == 10)),
                    points[:, 0] > 1,
                    np.linalg.norm(points[:, :2], axis=1) >= r1,
                    np.linalg.norm(points[:, :2], axis=1) <= r2,
                ]
            )

    @staticmethod
    def _convert_to_point_cloud(lidar_data):
        dtype = np.dtype([("position", np.float32, (3,)), ("cos_angle", np.float32), ("object_index", np.uint32), ("tag", np.uint32)])
        p_cloud = np.frombuffer(lidar_data.raw_data, dtype=dtype)
        tags = p_cloud["tag"]
        positions = np.array(p_cloud["position"])
        return p_cloud[LidarData._valid_points(positions, tags)]

    def query_object_index(self, object_index: int) -> Tuple[float, Optional[np.ndarray]]:
        """Return the distance to the nearest point with object index `object_index` and its position.

        Args:
            object_index (int): The object index of the object to query to distances to.

        Returns:
            Tuple[float, np.ndarray]: Returns the closest distance and the position of the object.
                If no point of the object is found, a distance of `np.inf` is returned.
        """

        object_indices = self.point_cloud["object_index"] != object_index
        distances = np.copy(self.distances)
        distances[object_indices] = np.inf
        if len(distances) > 0:
            i = np.argmin(distances)
            return distances[i], self.point_cloud["position"][i]
        else:
            try:
                return np.inf, self.point_cloud["position"][0]
            except IndexError:
                return np.inf, None


def lidar_sensor(lidar_data):
    """Iterates over all lidar data, to check for possible collisions with vehicles or walkers.

    Appends any potential collisions to `obstacles`.

    Args:
        lidar_data (carla.SemanticLidarMeasurement): The lidar data from the semantic lidar sensor
    """

    global obstacles, last_lidar_data

    if collision_detection:
        lidar_data = LidarData(lidar_data)
        if last_lidar_data is not None:
            obstacles = []
            for object_index in lidar_data.object_indices:
                dist_new, point = lidar_data.query_object_index(object_index)
                dist_old, _ = last_lidar_data.query_object_index(object_index)
                if dist_old != np.inf and dist_new != np.inf:
                    obstacles.append((dist_new, (dist_new - dist_old) / (lidar_data.timestamp - last_lidar_data.timestamp), point))
            obstacles.sort(key=lambda obstacle: obstacle[0])
        last_lidar_data = lidar_data

    render_barrier.wait()


def rgb_sensor(image):
    """Displays the sensor data or output it to a file.

    Checks for traffic lights, if enabled and sets `detected_red_traffic_light` if a red traffic light was detected.
    Also checks for traffic signs, if enabled and updates `target_speed` with a new target_speed.

    Args:
        image (carla.Image): The image data from the rgb sensor
    """

    global detected_red_traffic_light, target_speed

    render_barrier.wait()

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

        if area > 1800 and average_colours[0] > TRAFFIC_LIGHT_SENSITIVITY and traffic_light_detection:
            detected_red_traffic_light = True
        else:
            detected_red_traffic_light = False
    else:
        detected_red_traffic_light = False

    if traffic_sign is not None:
        area = cv2.contourArea(traffic_sign)

        if area >= TRAFFIC_SIGN_DETECTION_RANGE[0] and area <= TRAFFIC_SIGN_DETECTION_RANGE[1]:
            x, y, w, h = cv2.boundingRect(traffic_sign)
            if (
                w > 40
                and h > 40
                and x - w * BORDER > 20
                and x + w * (1 + BORDER) < WIDTH - 20
                and y - h * BORDER > 20
                and y + h * (1 + BORDER) < HEIGHT - 20
            ):
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
    current_sign = SPEED_30_SIGN if target_speed == 30 else (SPEED_60_SIGN if target_speed == 60 else SPEED_90_SIGN)
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

    render_barrier.reset()
    tick_event.set()

def road_sign_detection(image):
    global traffic_light, traffic_sign, road_contour, road_sign_case, current_cooldown
    base_colour = np.full_like(image, np.array([255, 255, 255]))
    stencil = np.zeros_like(image[:, :, 0])

    # Road Sign Detection
    mask = cv2.inRange(image, np.array([169, 119, 49]), np.array([171, 121, 51]))

    road_sign = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # convert image to greyscale
    road_sign = cv2.cvtColor(road_sign, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(road_sign, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if current_cooldown > 0:
        if brake == 0:
            current_cooldown -= 1
    elif len(contours) > 0 and road_sign_case == 0:
        road_sign = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(road_sign)

        moment = cv2.moments(road_sign)
        cX = int(moment["m10"] / (moment["m00"] + 0.001))
        if cX <= ROAD_SIGN_DETECTION_RANGE[0] and area >= ROAD_SIGN_DETECTION_AREA[0]:
            road_sign_case = 1
            current_cooldown = FRAME_COOLDOWN
        elif cX <= ROAD_SIGN_DETECTION_RANGE[1] and area >= ROAD_SIGN_DETECTION_AREA[1]:
            mask = cv2.inRange(image, np.array([127, 63, 127]), np.array([129, 65, 129]))

            road_decision = cv2.bitwise_and(base_colour, base_colour, mask=mask)

            # specify coordinates of the polygon
            polygon = np.array([[5 * WIDTH // 8, HEIGHT], [5 * WIDTH // 8, HEIGHT - 400], [WIDTH, HEIGHT - 400], [WIDTH, HEIGHT]])

            # fill polygon with ones
            cv2.fillConvexPoly(stencil, polygon, 255)

            # apply stencil
            road_decision = cv2.bitwise_and(road_decision, road_decision, mask=stencil)
            road_decision = cv2.cvtColor(road_decision, cv2.COLOR_BGR2GRAY)

            area = np.average(road_decision) * WIDTH * HEIGHT / 255
            if area >= ROAD_DETECTION_AREA:
                road_sign_case = 1
            else:
                road_sign_case = 3

            current_cooldown = FRAME_COOLDOWN
            cv2.imwrite(f"images/road_decision{frame}-{area}-{'left' if area >= ROAD_DETECTION_AREA else 'right'}.png", road_decision)

        elif cX > ROAD_SIGN_DETECTION_RANGE[1] and area >= ROAD_SIGN_DETECTION_AREA[2]:
            road_sign_case = 3
            current_cooldown = FRAME_COOLDOWN
        else:
            road_sign_case = 0
    else:
        road_sign_case = 0

def road_outline(image):
    global traffic_light, traffic_sign, road_contour, road_sign_case, current_cooldown
    
    masks = [
        cv2.inRange(image, np.array([156, 233, 49]), np.array([158, 235, 51])),
        cv2.inRange(image, np.array([127, 63, 127]), np.array([129, 65, 129])),
        # cv2.inRange(array, np.array([0, 0, 141]), np.array([1, 1, 143])),
        # cv2.inRange(array, np.array([(219, 19, 59)]), np.array([(221, 21, 61)])),
    ]
    mask = masks[0]
    for m in masks[1:]:
        cv2.bitwise_or(mask, m, mask)

    base_colour = np.full_like(image, np.array([255, 255, 255]))
    stencil = np.zeros_like(image[:, :, 0])

    road_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # specify coordinates of the polygon
    if road_sign_case == 3:
        polygon = np.array([[0, HEIGHT], [0, HEIGHT - 200], [WIDTH - 400, HEIGHT - 200], [WIDTH - 400, HEIGHT]])
    else:
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

def detect_traffic_light(image):
    global traffic_light, traffic_sign, road_contour, road_sign_case, current_cooldown

    base_colour = np.full_like(image, np.array([255, 255, 255]))
    
    # preparing the mask to overlay
    mask = cv2.inRange(image, np.array([249, 169, 29]), np.array([251, 171, 31]))

    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # convert image to greyscale
    traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_light = max(contours, key=cv2.contourArea)
    else:
        traffic_light = None
        
def detect_traffic_sign(image):
    global traffic_light, traffic_sign, road_contour, road_sign_case, current_cooldown

    base_colour = np.full_like(image, np.array([255, 255, 255]))
    stencil = np.zeros_like(image[:, :, 0])

    # preparing the mask to overlay
    mask = cv2.inRange(image, np.array([219, 219, 0]), np.array([221, 221, 1]))
    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

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

def segmentation_sensor(image):
    """Detects the road contour for steering and detects traffic signs and lights if enabled.

    Writes the current road contour into `road_countour`, as well as the current traffic sign and light into
    `traffic_sign` and `traffic_light` respectively, if enabled.

    Args:
        image (carla.Image): The image data from the semantic segmentation sensor.
    """
    global traffic_light, traffic_sign, road_contour, road_sign_case, current_cooldown

    # now = time.time()

    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    functions = [road_sign_detection, road_outline, detect_traffic_light, detect_traffic_sign]
    threads = []
    for function in functions:
        thread = threading.Thread(target=function, args=(array,))
        threads.append(thread)
        thread.start()
    
    for index, thread in enumerate(threads):
        thread.join()
        # print(f"Thread {index} took {(time.time() - now) * 1000:.3f}ms")

    render_barrier.wait()


def gnss_sensor(sensor_data):
    """Reads the current data from the gnss sensor and updates the `current_speed` of the vehicle.

    Args:
        sensor_data (carla.GnssMeasurement): The measurement data from the gnss sensor.
    """

    global prev_sensor_data, current_speed

    if prev_sensor_data is not None:
        dx = sensor_data.transform.location.distance(prev_sensor_data.transform.location)
        dt = sensor_data.timestamp - prev_sensor_data.timestamp
        current_speed = dx / dt
        prev_sensor_data = sensor_data
    else:
        prev_sensor_data = sensor_data

    render_barrier.wait()


def vehicle_control() -> Tuple[float, float, float]:
    """Performs the vehicle control based on the `road_contour`, the `target_speed` and `detected_red_traffic_light`.

    Uses two pid controllers to perform control of the vehicle.

    Returns:
        Tuple[float, float, float]: The throttle, steering and braking outputs of the controller.
    """

    # calulate steering value
    if road_contour is not None:
        moment = cv2.moments(road_contour)
        cX = int(moment["m10"] / moment["m00"])

        if road_sign_case == 3:
            diff = cX - WIDTH / 2 + ROAD_OFFSET_LEFT
            steering = steering_pid(diff / 490, 0)
        else:
            diff = cX - WIDTH / 2 + ROAD_OFFSET
            steering = steering_pid(diff / 420, 0)
    else:
        steering = 0

    # calculate speed delta
    new_target_speed = target_speed if traffic_sign_detection else 30.0
    if len(obstacles) > 0:
        speed_delta = min(new_target_speed / 3.6, obstacles[0][1] + (obstacles[0][0] - 7) / 2) - current_speed
    else:
        speed_delta = new_target_speed / 3.6 - current_speed

    # calculate throttle and brake from speed_delta
    throttle = throttle_pid(speed_delta, 0, (-1, 1 / (1 + abs(steering))))
    brake = -throttle if throttle < 0 else 0
    throttle = throttle if throttle > 0 else 0

    if brake > 0:
        steering = 0

    # detected red traffic light
    if detected_red_traffic_light:
        return 0.0, steering, 1.0
    return throttle, steering, brake


def visualize_path():
    """Visualizes the path of the vehicle based on its current throttle and steering values."""
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
    telemetry_info: bool,
    road_borders: bool,
    generate_traffic: bool,
    number_of_vehicles: int,
    number_of_walkers: int,
):
    """Main method

    Args:
        ip (str): IP adress of the carla server.
        enable_path_visualization (bool): Wether to enable path visualization of the vehicle.
        telemetry_info (bool): Wether to output telemetry info of the vehicle.
        road_borders (bool): Wether to spawn road borders.
        generate_traffic (bool): Wether to generate vehicle and walker traffic.
        number_of_vehicles (int): Number of vehicles to spawn.
        number_of_walkers (int): Number of walkers to spawn.
    """
    global waypoint, waypoint_deadzone, lidar, world, vehicle, screen, brake, throttle, steering, frame

    load_model()

    if display_image:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

    client = carla.Client(ip, 2000)

    gnss = None
    rgb = None
    segmentation = None
    lidar = None
    vehicle = None

    try:
        client.set_timeout(10.0)
        world = client.load_world("Town02_Opt")
        # world = client.get_world()
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Decals)
        # world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # world.unload_map_layer(carla.MapLayer.Foliage)
        # world.unload_map_layer(carla.MapLayer.Walls)
        world.unload_map_layer(carla.MapLayer.Props)

        # Settings
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        spawn_point = carla.Transform(carla.Location(191.76, 273.54, 1.0), carla.Rotation(yaw=90.0))
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

        world.tick()

        while frame < MAX_FRAME:
            throttle, steering, brake = vehicle_control()
            control = carla.VehicleControl(
                throttle=throttle, steer=steering, brake=brake, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)

            if enable_path_visualization:
                visualize_path()

            tick_event.wait()
            tick_event.clear()
            frame += 1
            world.tick()

            if display_image:
                pygame.display.flip()
                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

            if telemetry_info:
                print(
                    f"frame: {frame:04}, throttle: {throttle:.03f}, steering: {steering:.03f}, brake: {brake:.03f}, road sign: {road_sign_case},"
                    f" speed:  {current_speed * 3.6:.03f} km/h, target speed {target_speed:.03f} km/h, obstacles: {len(obstacles)}",
                    end="\033[0K\r",
                )
    except KeyboardInterrupt:
        pass
    finally:
        traceback.print_exc()
        if generate_traffic:
            destroy_traffic(client)

        if gnss is not None:
            gnss.destroy()
        if segmentation is not None:
            segmentation.destroy()
        if rgb is not None:
            rgb.destroy()
        if lidar is not None:
            lidar.destroy()
        if vehicle is not None:
            vehicle.destroy()

        try:
            pygame.quit()
        except NameError:
            pass

        print("\nCleaned up")
        quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server")
    parser.add_argument("--path-visualization", dest="visualize_path", action="store_true", help="Enable path visualization")
    parser.add_argument("--no-collision-detection", dest="collision_detection", action="store_false", help="Disable collision detection")
    parser.add_argument("--no-sign-detection", dest="sign_detection", action="store_false", help="Disable traffic sign detection")
    parser.add_argument("--no-light-detection", dest="light_detection", action="store_false", help="Disable traffic lights detection")
    parser.add_argument("--no-telemetry", dest="telemetry_info", action="store_false", help="Disable telemetry info")
    parser.add_argument("--no-road-borders", dest="spawn_road_borders", action="store_false", help="Disable spawning of road borders")
    parser.add_argument("--no-traffic", dest="spawn_traffic", action="store_false", help="Disable spawning of traffic")
    parser.add_argument("--output-to-file", dest="write_to_file", action="store_true", help="Enable writing of output image to file")
    parser.add_argument("--no-display", dest="display_image", action="store_false", help="Disable displaying of image on screen")
    parser.add_argument("--number-of-vehicles", default=30, type=int, help="Number of vehicles")
    parser.add_argument("--number-of-walkers", default=10, type=int, help="Number of walkers")

    args = parser.parse_args()

    # import needed modules
    if args.spawn_traffic:
        from generate_traffic import spawn_traffic, destroy_traffic

    if args.spawn_road_borders:
        from spawn_road_borders import spawn_road_borders

    if args.display_image:
        import pygame
        from pygame.locals import *

    collision_detection = args.collision_detection
    traffic_sign_detection = args.sign_detection
    traffic_light_detection = args.light_detection
    write_to_file = args.write_to_file
    display_image = args.display_image

    main(
        args.host,
        args.visualize_path,
        args.telemetry_info,
        args.spawn_road_borders,
        args.spawn_traffic,
        args.number_of_vehicles,
        args.number_of_walkers,
    )
