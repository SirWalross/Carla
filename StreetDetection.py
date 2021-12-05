import random
import argparse
import time
import numpy as np
import pygame
from pygame.locals import *
import cv2
from pid import PID

import carla
from carla import ColorConverter as cc

pygame.init()
screen = pygame.display.set_mode((1920, 720))

frame = 0
throttle = 0.4
steering = 0
STEERING_MAX = 0.05
crossing = []  # l, f, r
pid = PID(0.5, 0, 0.2)


def rgb_sensor(image):
    global detected_red_traffic_light
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]


def display_image(image):
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
    mask_road_line = cv2.inRange(array, np.array([156, 233, 49]), np.array([158, 235, 51]))
    mask_road = cv2.inRange(array, np.array([127, 63, 127]), np.array([129, 65, 129]))
    mask = cv2.bitwise_or(mask_road_line, mask_road)

    base_colour = np.full_like(array, np.array([100, 100, 100]))
    array = cv2.bitwise_and(base_colour, base_colour, mask=mask)

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

        diff = cX - 1920 / 2 + 100
        steering = pid(diff / 400, 0)
        print(steering)
    else:
        pass

    polygons = [
        [[0, 200], [400, 200], [400, 500], [0, 500]],
        [[1920, 200], [1520, 200], [1520, 500], [1920, 500]],
        [[910, 400], [910, 300], [1010, 300], [1010, 400]],
    ]

    for i in range(3):
        # preparing the mask to overlay
        mask_road_line = cv2.inRange(kreuzungen[i], np.array([156, 233, 49]), np.array([158, 235, 51]))
        mask_road = cv2.inRange(kreuzungen[i], np.array([127, 63, 127]), np.array([129, 65, 129]))
        mask = cv2.bitwise_or(mask_road_line, mask_road)

        base_colour = np.full_like(kreuzungen[i], np.array([100, 100, 100]))
        kreuzungen[i] = cv2.bitwise_and(base_colour, base_colour, mask=mask)

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

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(raw_image, [contour], -1, (255, 0, 0), 1)
        else:
            pass

    surface = pygame.surfarray.make_surface(raw_image.swapaxes(0, 1))
    screen.blit(surface, (0, 0))

    # surface = pygame.surfarray.make_surface(raw_image.swapaxes(0, 1))
    # screen.blit(surface, (0, 0))
    # frame += 1


def main(ip: str):
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town07_Opt")

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = map.get_spawn_points()

        # print([str(spawn_point.location) for spawn_point in spawn_points])
        vehicle = None
        while True:
            try:
                spawn_point = spawn_points[0]
                spawn_point.location = carla.Location(-2, -50, 1)
                # spawn_point.location = world.get_random_location_from_navigation()
                vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                continue
        print("Spawned vehicle")

        camera_bp = blueprint_library.find("sensor.camera.semantic_segmentation")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute("image_size_x", "1920")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "150")
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute("sensor_tick", "0.05")

        rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "1920")
        rgb_camera_bp.set_attribute("image_size_y", "720")
        rgb_camera_bp.set_attribute("fov", "150")
        rgb_camera_bp.set_attribute("sensor_tick", "0.02")
        relative_transform = carla.Transform(carla.Location(x=2.5, y=0, z=1.7), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(rgb_sensor)

        relative_transform = carla.Transform(carla.Location(x=2.5, y=0, z=1.7), carla.Rotation(yaw=0))
        camera = world.spawn_actor(camera_bp, relative_transform, vehicle)
        camera.listen(display_image)
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
