import argparse
import math
import random
from typing import List, Tuple
import numpy as np
import carla
import cv2
import os
import shortuuid
import time

from carla import ColorConverter as cc

random.seed(42)

MAX_IMAGES = 10000
WIDTH = 1280
HEIGHT = 720
BORDER = 0.1
TRAFFIC_SIGN_DETECTION_RANGE = (500, 1200)  # min and max area of sign
traffic_sign = None
vehicle = None

signs = []
current_speed = None

def write_signs_to_disk(sign_type: float):
    sign_type = int(3.6 * sign_type)
    global signs
    for sign in signs:
        os.makedirs(f"sign/{sign_type}", exist_ok=True)
        cv2.imwrite(f"sign/{sign_type}/{shortuuid.uuid()}.png", sign)
    print(f"Written {len(signs)} speed {sign_type} signs")
    signs = []


def rgb_sensor(image):
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    array = np.array(array)
    raw_image = np.copy(array)

    if traffic_sign is not None:
        area = cv2.contourArea(traffic_sign)

        if area >= TRAFFIC_SIGN_DETECTION_RANGE[0] and area <= TRAFFIC_SIGN_DETECTION_RANGE[1]:
            x, y, w, h = cv2.boundingRect(traffic_sign)
            x1 = int(np.clip(x - w * BORDER, 0, WIDTH))
            x2 = int(np.clip(x + w * (1 + BORDER), 0, WIDTH))
            y1 = int(np.clip(y - h * BORDER, 0, HEIGHT))
            y2 = int(np.clip(y + h * (1 + BORDER), 0, HEIGHT))
            image = raw_image[y1:y2, x1:x2, :]

            if x + w <= WIDTH - 2 * w * BORDER and w > 16:
                signs.append(image)


def segmentation_sensor(image):
    global traffic_sign
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # Traffic sign

    # preparing the mask to overlay
    mask = cv2.inRange(array, np.array([219, 219, 0]), np.array([221, 221, 1]))

    base_colour = np.full_like(array, np.array([255, 255, 255]))
    array = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # create a zero array
    stencil = np.zeros_like(array[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[int(WIDTH / 2), 200], [int(WIDTH / 2), HEIGHT], [WIDTH, HEIGHT], [WIDTH, 200]])
    cv2.fillConvexPoly(stencil, polygon, 255)

    # convert image to greyscale
    array = cv2.bitwise_and(array, array, mask=stencil)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_sign = max(contours, key=cv2.contourArea)
    else:
        traffic_sign = None


def main(ip: str):
    global vehicle, current_speed
    now = time.time()
    segmentation = None
    rgb = None
    vehicle = None
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")
        # world = client.get_world()
        
        # Settings
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

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
        current_speed = vehicle.get_speed_limit()

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

        vehicle.set_autopilot()

        while time.time() - now > 60.0 * 1:
            speed = vehicle.get_speed_limit()
            if current_speed != speed:
                current_speed = speed
                write_signs_to_disk(current_speed)
            world.tick()
        raise ValueError()
    finally:
        if segmentation is not None:
            segmentation.destroy()
        if rgb is not None:
            rgb.destroy()
        if vehicle is not None:
            vehicle.destroy()
        print("Cleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
