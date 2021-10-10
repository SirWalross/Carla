import random
import argparse
import time
import numpy as np
import pygame
from pygame.locals import *
import cv2

import carla
from carla import ColorConverter as cc

pygame.init()
screen = pygame.display.set_mode((1280, 720))


def closest(values, Number):
    aux = []
    for value in values:
        aux.append(abs(Number - value))

    return aux.index(sorted(aux)[0]), aux.index(sorted(aux)[1])


def display_image(image):
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    raw_image = np.copy(array)

    # preparing the mask to overlay
    mask_road_line = cv2.inRange(array, np.array([156, 233, 49]), np.array([158, 235, 51]))
    mask_side_walk = cv2.inRange(array, np.array([243, 34, 231]), np.array([245, 36, 233]))

    # array = cv2.bitwise_or(cv2.bitwise_and(array, array, mask = mask_road_line), cv2.bitwise_and(array, array, mask = mask_side_walk))
    array = cv2.bitwise_and(array, array, mask=mask_side_walk)

    # create a zero array
    stencil = np.zeros_like(array[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[-100, 720], [430, 400], [860, 400], [1430, 720]])
    # polygon = np.array([[100,100], [1180,100], [1180,620], [100,620]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 255)

    # apply stencil
    # array = cv2.bitwise_and(array, array, mask = stencil)
    base_colour = np.full_like(array, np.array([244, 35, 232]))
    cv2.fillConvexPoly(base_colour, polygon, (0, 0, 0))
    array = cv2.bitwise_or(array, base_colour)

    # edge detection
    edges = cv2.Canny(array, 100, 200)

    # Line detection
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = np.copy(edges) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    x_midpoints = [x1 + (x2 - x1) * 0.5 for x1, _, x2, _ in [line[0] for line in lines]]
    indices = closest(x_midpoints, image.width / 2)
    lines = [line[0] for index, line in enumerate(lines) if index in indices]

    for x1, y1, x2, y2 in lines:
        cv2.line(raw_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

    surface = pygame.surfarray.make_surface(raw_image.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    return lines


def main(ip: str):
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_point = carla.Transform(
            carla.Location(x=122.972397, y=238.584183, z=0.405664),
            carla.Rotation(pitch=0.000000, yaw=-179.999634, roll=0.000000),
        )
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        print(vehicle_bp)
        print(spawn_point)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        camera_bp = blueprint_library.find("sensor.camera.semantic_segmentation")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "110")
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute("sensor_tick", "0.1")

        relative_transform = carla.Transform(carla.Location(x=1.2, y=-0.5, z=1.7), carla.Rotation(yaw=0))
        camera = world.spawn_actor(camera_bp, relative_transform, vehicle)
        camera.listen(display_image)
        while 1:
            control = carla.VehicleControl(
                throttle=0.5, steer=0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)
            pygame.display.flip()
            pygame.display.update()
            time.sleep(0.02)
    finally:
        vehicle.destroy()
        pygame.quit()
        print("Cleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
