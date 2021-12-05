import random
import argparse
import time
import numpy as np
import pygame
from pygame.locals import *
import cv2
import sys
from pid import PID

import carla
from carla import ColorConverter as cc

pygame.init()

WIDTH = 1920
HEIGHT = 720
STEERING_OFFSET = 100  # in pixels
FOV = 130
STEERING_MAX = 0.05
DISTANCE_THRESHHOLD = (10, 10, 20)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

frame = 0
throttle = 0.4
steering = 0
crossing = []  # l, f, r
depth_buffer = None
pid = PID(0.5, 0, 0.1)


def depth_sensor(image):
    global depth_buffer
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    depth_buffer = 1e9 * (array[:, :, 0] + array[:, :, 1] * 256 + array[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)

    # print(depth_buffer)

    
    # surface = pygame.surfarray.make_surface(depth_buffer.swapaxes(0, 1))
    # screen.blit(surface, (0, 0))


def rgb_sensor(image):
    global detected_red_traffic_light
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    # surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    # screen.blit(surface, (0, 0))


def semantic_sensor(image):
    global throttle
    global steering
    global lines
    global frame
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    raw_image = np.copy(array)
    kreuzungen = [np.copy(array), np.copy(array), np.copy(array)]

    # preparing the mask to overlay
    mask_road = cv2.inRange(array, np.array([0, 0, 0]), np.array([0, 0, 0]))

    base_colour = np.full_like(array, np.array([100, 100, 100]))
    array = cv2.bitwise_and(base_colour, base_colour, mask=mask_road)

    # create a zero array
    stencil = np.zeros_like(array[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[400, 720], [400, 500], [1520, 500], [1520, 720]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 255)

    # apply stencil
    array = cv2.bitwise_and(array, array, mask=stencil)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(array.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.drawContours(raw_image, [contour], -1, (255, 0, 0), 1)

        diff = cX - WIDTH / 2 + STEERING_OFFSET
        steering = pid(diff / 400, 0)
        print(steering)
    else:
        pass

    polygons = [
        [[0, 200], [400, 200], [400, 500], [0, 500]],
        [[1920, 200], [1520, 200], [1520, 500], [1920, 500]],
        [[710, 400], [710, 300], [1110, 300], [1110, 400]],
    ]

    for i in range(3):
        # preparing the mask to overlay
        mask_road = cv2.inRange(kreuzungen[i], np.array([0, 0, 0]), np.array([0, 0, 0]))

        base_colour = np.full_like(kreuzungen[i], np.array([100, 100, 100]))
        kreuzungen[i] = cv2.bitwise_and(base_colour, base_colour, mask=mask_road)

        # create a zero array
        stencil = np.zeros_like(kreuzungen[i][:, :, 0])

        # specify coordinates of the polygon
        polygon = np.array(polygons[i])

        # fill polygon with ones
        cv2.fillConvexPoly(stencil, polygon, 255)

        # apply stencil
        kreuzungen[i] = cv2.bitwise_and(kreuzungen[i], kreuzungen[i], mask=stencil)
        kreuzungen[i] = cv2.cvtColor(kreuzungen[i], cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(kreuzungen[i].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0 and depth_buffer is not None:
            contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(depth_buffer).astype("uint8")
            cv2.drawContours(mask, [contour], -1, (255), 1)
            distance = cv2.bitwise_and(depth_buffer, depth_buffer, mask=mask)
            # print(distance)
            distance[np.where((distance==0).all(axis=1))] = 255
            # print(distance)
            if np.min(distance) <= DISTANCE_THRESHHOLD[i]:
                cv2.drawContours(raw_image, [contour], -1, (255, 0, 0), 1)
                cv2.putText(raw_image, f"{np.min(distance):.5f}", contour[0][0], cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        else:
            pass

    surface = pygame.surfarray.make_surface(raw_image.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    # frame += 1


def main(ip: str):
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        with open("OpenDriveMaps/map02.xodr", encoding="utf-8") as od_file:
            try:
                data = od_file.read()
            except OSError:
                print("file could not be readed.")
                sys.exit()
        vertex_distance = 2.0  # in meters
        max_road_length = 500.0  # in meters
        wall_height = 0.0  # in meters
        extra_width = 0.6  # in meters
        world = client.generate_opendrive_world(
            data,
            carla.OpendriveGenerationParameters(
                vertex_distance=vertex_distance,
                max_road_length=max_road_length,
                wall_height=wall_height,
                additional_width=extra_width,
                smooth_junctions=True,
                enable_mesh_visibility=True,
            ),
        )

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = map.get_spawn_points()

        vehicle = None
        while True:
            try:
                spawn_point = spawn_points[0]
                # spawn_point.location = world.get_random_location_from_navigation()
                vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                continue
        print("Spawned vehicle")

        relative_transform = carla.Transform(carla.Location(x=2.5, y=0, z=1.7), carla.Rotation(yaw=0))

        camera_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_bp.set_attribute("image_size_x", f"{WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{HEIGHT}")
        camera_bp.set_attribute("fov", f"{FOV}")
        camera_bp.set_attribute("sensor_tick", "0.05")
        camera = world.spawn_actor(camera_bp, relative_transform, vehicle)
        camera.listen(semantic_sensor)

        rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", f"{WIDTH}")
        rgb_camera_bp.set_attribute("image_size_y", f"{HEIGHT}")
        rgb_camera_bp.set_attribute("fov", f"{FOV}")
        rgb_camera_bp.set_attribute("sensor_tick", "0.02")
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(rgb_sensor)

        rgb_camera_bp = blueprint_library.find("sensor.camera.depth")
        rgb_camera_bp.set_attribute("image_size_x", f"{WIDTH}")
        rgb_camera_bp.set_attribute("image_size_y", f"{HEIGHT}")
        rgb_camera_bp.set_attribute("fov", f"{FOV}")
        rgb_camera_bp.set_attribute("sensor_tick", "0.02")
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(depth_sensor)

        while 1:
            control = carla.VehicleControl(
                throttle=throttle, steer=steering, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)
            pygame.display.flip()
            pygame.display.update()
            time.sleep(0.05)
    finally:
        try:
            vehicle.destroy()
        except NameError:
            pass
        pygame.quit()
        print("\nCleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
