import argparse
import math
import random
from typing import List, Tuple
import numpy as np
import carla
import cv2
import os
import shortuuid

from carla import ColorConverter as cc

random.seed(42)

written_images = 0
MAX_IMAGES = 10000
WIDTH = 1280
HEIGHT = 720
BORDER = 0.1
traffic_light = None
traffic_light = []
traffic_sign = None
traffic_signs = []
vehicle = None


def get_yaw_diff(traffic_sign_vector, vehicle_vector) -> int:
    traffic_sign_vector = np.array([traffic_sign_vector.x - vehicle_vector.x, traffic_sign_vector.y - vehicle_vector.y, 0.0])
    vehicle_vector = np.array([vehicle_vector.x, vehicle_vector.y, 0.0])
    wv_linalg = np.linalg.norm(traffic_sign_vector) * np.linalg.norm(vehicle_vector)
    if wv_linalg == 0:
        _dot = 1
    else:
        _dot = math.acos(np.clip(np.dot(traffic_sign_vector, vehicle_vector) / (wv_linalg), -1.0, 1.0))

    _cross = np.cross(vehicle_vector, traffic_sign_vector)
    if _cross[2] < 0:
        _dot *= -1.0
    return int(_dot * 180 / np.pi)


def get_traffic_sign_type() -> Tuple[int, int]:
    # offset vehicle location by 10 m in forward direction and 1 m to the right
    location = vehicle.get_location() + vehicle.get_transform().get_forward_vector() * 10 + vehicle.get_transform().get_right_vector()

    distances = [sign.get_location().distance(location) for sign in traffic_signs]
    index = np.argmin(distances)
    yaw_diff = get_yaw_diff(traffic_signs[index].get_transform().get_forward_vector(), vehicle.get_transform().get_forward_vector())

    if abs(yaw_diff) < 180:
        return int(traffic_signs[index].type_id[20:]), yaw_diff
    else:
        return -1, yaw_diff


def get_traffic_light_type() -> Tuple[int, int]:
    # offset vehicle location by 10 m in forward direction and 1 m to the right
    location = vehicle.get_location() + vehicle.get_transform().get_forward_vector() * 10 + vehicle.get_transform().get_right_vector()

    distances = [light.get_location().distance(location) for light in traffic_lights]
    index = np.argmin(distances)
    yaw_diff = get_yaw_diff(traffic_lights[index].get_transform().get_forward_vector(), vehicle.get_transform().get_forward_vector())

    if abs(yaw_diff) < 180:
        return traffic_lights[index].state, yaw_diff
    else:
        return carla.TrafficLightState.Off, yaw_diff


def rgb_sensor(image):
    global written_images
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    array = np.array(array)
    raw_image = np.copy(array)

    if traffic_light is not None and len(traffic_light) > 0:
        x, y, w, h = cv2.boundingRect(traffic_light)
        x1 = int(np.clip(x - w * BORDER, 0, WIDTH))
        x2 = int(np.clip(x + w * (1 + BORDER), 0, WIDTH))
        y1 = int(np.clip(y - h * BORDER, 0, HEIGHT))
        y2 = int(np.clip(y + h * (1 + BORDER), 0, HEIGHT))
        image = raw_image[y1:y2, x1:x2, :]

        light_type, angle = get_traffic_light_type()
        if x + w <= WIDTH - 2 * w * BORDER and w > 16:
            os.makedirs(f"traffic_light/{light_type}", exist_ok=True)
            cv2.imwrite(f"traffic_light/{light_type}/{written_images}.{angle:.2f}.{x + w}.png", image)
            written_images += 1

    if traffic_sign is not None:
        x, y, w, h = cv2.boundingRect(traffic_sign)
        x1 = int(np.clip(x - w * BORDER, 0, WIDTH))
        x2 = int(np.clip(x + w * (1 + BORDER), 0, WIDTH))
        y1 = int(np.clip(y - h * BORDER, 0, HEIGHT))
        y2 = int(np.clip(y + h * (1 + BORDER), 0, HEIGHT))
        image = raw_image[y1:y2, x1:x2, :]

        sign_type, angle = get_traffic_sign_type()
        if x + w <= WIDTH - 2 * w * BORDER and w > 16:
            os.makedirs(f"traffic_sign/{sign_type}", exist_ok=True)
            cv2.imwrite(f"traffic_sign/{sign_type}/{written_images}.{angle:.2f}.{x + w}.png", image)
            written_images += 1


def segmentation_sensor(image):
    global traffic_light, traffic_sign
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = np.array(array[:, :, ::-1], dtype=np.dtype("uint8"))

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
        print("detected light")
    else:
        traffic_light = None

    # Traffic sign
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
        print(f"detected sign")
    else:
        traffic_sign = None


def main(ip: str):
    global vehicle, traffic_lights, traffic_signs
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")
        # world = client.get_world()

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = map.get_spawn_points()
        while True:
            try:
                spawn_point = random.choice(spawn_points)
                spawn_point.location.z += 2
                vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                pass
        print("Spawned vehicle")

        # RGB camera
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", f"{WIDTH}")
        rgb_bp.set_attribute("image_size_y", f"{HEIGHT}")
        rgb_bp.set_attribute("fov", "60")
        rgb_bp.set_attribute("sensor_tick", "0.02")
        relative_transform = carla.Transform(carla.Location(x=1.2, y=0, z=1.7), carla.Rotation(yaw=0))
        rgb = world.spawn_actor(rgb_bp, relative_transform, vehicle)
        rgb.listen(rgb_sensor)

        # Segmentation camera
        segmentation_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        segmentation_bp.set_attribute("image_size_x", f"{WIDTH}")
        segmentation_bp.set_attribute("image_size_y", f"{HEIGHT}")
        segmentation_bp.set_attribute("fov", "60")
        segmentation_bp.set_attribute("sensor_tick", "0.02")
        segmentation = world.spawn_actor(segmentation_bp, relative_transform, vehicle)
        segmentation.listen(segmentation_sensor)

        traffic_signs = [world.get_traffic_sign(landmark) for landmark in map.get_all_landmarks_of_type("274")]
        traffic_lights = [world.get_traffic_light(landmark) for landmark in map.get_all_landmarks_of_type("1000001")]

        vehicle.set_autopilot()

        while written_images < MAX_IMAGES:
            pass
    finally:
        try:
            segmentation.destroy()
            rgb.destroy()
            vehicle.destroy()
        except UnboundLocalError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
