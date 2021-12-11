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
TRAFFIC_LIGHT_SENSITIVITY = 0.35
crossing = []  # l, f, r
pid = PID(1.3, 0, 0.2)
traffic_light = None
detected_red_traffic_light = False


def rgb_sensor(image):
    global detected_red_traffic_light
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
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
        print(f"{average_colours[0]}, {area}")
        if average_colours[0] > TRAFFIC_LIGHT_SENSITIVITY:
            cv2.drawContours(array, [traffic_light], -1, (255, 0, 0), 1)
            cv2.putText(array, f"{average_colours[0]:.2f},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        else:
            cv2.drawContours(array, [traffic_light], -1, (0, 255, 0), 1)
            cv2.putText(array, f"{average_colours[0]:.2f},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        if area > 40 and average_colours[0] > TRAFFIC_LIGHT_SENSITIVITY and steering <= 0.1:
            detected_red_traffic_light = True
        else:
            detected_red_traffic_light = False
    else:
        detected_red_traffic_light = False
        

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def display_image(image):
    global throttle, steering, lines, frame, traffic_light
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
    else:
        pass

    # Traffic light
    traffic_image = np.copy(raw_image)
    # preparing the mask to overlay
    mask = cv2.inRange(traffic_image, np.array([249, 169, 29]), np.array([251, 171, 31]))

    base_colour = np.full_like(traffic_image, np.array([255, 255, 255]))
    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # create a zero array
    stencil = np.zeros_like(traffic_image[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[930, 0], [930, 360], [990, 360], [990, 0]])
    cv2.fillConvexPoly(stencil, polygon, 255)

    # convert image to greyscale
    traffic_image = cv2.bitwise_and(traffic_image, traffic_image, mask=stencil)
    traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_light = max(contours, key=cv2.contourArea)
        cv2.drawContours(raw_image, [traffic_light], -1, (255, 0, 0), 1)
    else:
        traffic_light = None


    # Signs #############################################################################################
    sign_image = np.copy(raw_image)
    # preparing the mask to overlay
    mask = cv2.inRange(traffic_image, np.array([249, 169, 29]), np.array([251, 171, 31]))

    base_colour = np.full_like(traffic_image, np.array([255, 255, 255]))
    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # create a zero array
    stencil = np.zeros_like(traffic_image[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[930, 0], [930, 360], [990, 360], [990, 0]])
    cv2.fillConvexPoly(stencil, polygon, 255)

    # convert image to greyscale
    traffic_image = cv2.bitwise_and(traffic_image, traffic_image, mask=stencil)
    traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_light = max(contours, key=cv2.contourArea)
        cv2.drawContours(raw_image, [traffic_light], -1, (255, 0, 0), 1)
    else:
        traffic_light = None



    # surface = pygame.surfarray.make_surface(raw_image.swapaxes(0, 1))
    # screen.blit(surface, (0, 0))
    # frame += 1


def main(ip: str):
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town07_Opt")
        
        # Settings
        # settings = world.get_settings()
        # settings.synchronous_mode = True  # Enables synchronous mode
        # settings.fixed_delta_seconds = 0.5
        # world.apply_settings(settings)

        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = map.get_spawn_points()

        # print([str(spawn_point.location) for spawn_point in spawn_points])
        vehicle = None
        while True:
            try:
                spawn_point = spawn_points[0]
                spawn_point.location = carla.Location(50, -2, 1)
                spawn_point.rotation.yaw -= 90
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
            if detected_red_traffic_light:
                control = carla.VehicleControl(
                    throttle=0, steer=steering, brake=1.0, hand_brake=False, reverse=False, manual_gear_shift=False
                )
            else:   
                control = carla.VehicleControl(
                    throttle=throttle, steer=steering, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False
                )
            vehicle.apply_control(control)
            pygame.display.flip()
            pygame.display.update()
            # world.tick()
            time.sleep(0.01)
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
