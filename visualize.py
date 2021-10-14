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

frame = 0
throttle = 0
steering = 0
change_amount = 0.1
lines = []

sign = lambda x: (1, -1)[x<0]


def closest(values, Number):
    array = []
    for value in values:
        array.append(abs(Number - value))

    return array.index(sorted(array)[0]), array.index(sorted(array)[1])


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

    # preparing the mask to overlay
    mask_road_line = cv2.inRange(array, np.array([156, 233, 49]), np.array([158, 235, 51]))
    mask_road = cv2.inRange(array, np.array([127, 63, 127]), np.array([129, 65, 129]))
    mask = cv2.bitwise_or(mask_road_line, mask_road)

    base_colour = np.full_like(array, np.array([100, 100, 100]))
    array = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # create a zero array
    stencil = np.zeros_like(array[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[-100, 720], [430, 400], [860, 400], [1430, 720]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 255)

    # apply stencil
    array = cv2.bitwise_and(array, array, mask=stencil)

    # edge detection
    edges = cv2.Canny(array, 100, 200)

    # Line detection
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    try:
        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        detected_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        x_midpoints = [x1 + (x2 - x1) * 0.5 for x1, _, x2, _ in [line[0] for line in detected_lines]]
        indices = closest(x_midpoints, image.width / 2)
        lines = [line[0] for index, line in enumerate(detected_lines) if index in indices]

        for x1, y1, x2, y2 in lines:
            cv2.line(raw_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

        if len(lines) == 2:
            slope = []
            for x1, y1, x2, y2 in lines:
                slope.append((x2 - x1)/(y2 - y1))
            steering_value = np.clip(slope[1] + slope[0], -2, 2) / 2
            diff = steering - steering_value
            if abs(diff) > change_amount:
                steering -= sign(diff) * change_amount
            else:
                steering = steering_value
            # throttle = np.clip(abs(slope[0]), 0.2, 1)
            throttle = np.clip(0.2, 0.2, 1)
    except IndexError:
        pass
    except TypeError:
        pass

    surface = pygame.surfarray.make_surface(raw_image.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    frame += 1


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
        camera_bp.set_attribute("sensor_tick", "0.05")

        relative_transform = carla.Transform(carla.Location(x=1.2, y=-0.5, z=1.7), carla.Rotation(yaw=0))
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

            if len(lines) == 2:
                lines_output = f"[({lines[0][0]}, {lines[0][1]}), ({lines[0][2]}, {lines[0][3]})], [({lines[1][0]}, {lines[1][1]}), ({lines[1][2]}, {lines[1][3]})]"
                print(f"frame {frame}, steering: {steering}, throttle: {throttle}, lines: {lines_output}", end="\r")
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
