import argparse
import math
import random
import time
from typing import Tuple
import numpy as np
import carla
import cv2
import pygame
from pygame.locals import *

from carla import ColorConverter as cc

pygame.init()
screen = pygame.display.set_mode((1280, 720))

current_speed = 0.0 # m/s
prev_sensor_data = None
steering_delta = 0
speed_delta = 0
distance_waypoint = 0
waypoint = None
detected_traffic_light = None
detected_red_traffic_light = False

def display_image(image):
    global detected_red_traffic_light
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    # convert from rgb to bgr
    array = array[:, :, ::-1]
    
    if detected_traffic_light is not None:
        mask = np.zeros_like(array[:, :, 0])
        cv2.drawContours(mask, detected_traffic_light, -1,255,3)

        img = cv2.bitwise_and(array, array, mask=mask)

        # average red color in image
        average_red = np.average(img[:, :, 2])
        if average_red > 10:
            detected_red_traffic_light = True
        else:
            detected_red_traffic_light = False
    else:
        detected_red_traffic_light = False


def detect_traffic_light(image):
    global detected_traffic_light
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    # preparing the mask to overlay
    mask = cv2.inRange(array, np.array([249, 169, 29]), np.array([251, 171, 31]))

    base_colour = np.full_like(array, np.array([100, 100, 100]))
    array = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # create a zero array
    stencil = np.zeros_like(array[:, :, 0])

    # specify coordinates of the polygon
    polygon = np.array([[-100, 0], [430, 320], [860, 320], [1430, 0]])

    # fill polygon with ones
    cv2.fillConvexPoly(stencil, polygon, 255)

    # apply stencil
    array = cv2.bitwise_and(array, array, mask=stencil)

    # contour detection
    # contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # if len(contours) > 0:
    #     # detected traffic light
    #     detected_traffic_light = contours[0]
    # else:
    #     detected_traffic_light = None
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))



def gnss_sensor(sensor_data):
    global prev_sensor_data, current_speed, distance_waypoint
    if prev_sensor_data is not None:
        dx = sensor_data.transform.location.distance(prev_sensor_data.transform.location)
        dt = sensor_data.timestamp - prev_sensor_data.timestamp
        current_speed = dx/dt
        prev_sensor_data = sensor_data
    else:
        prev_sensor_data = sensor_data
    if waypoint is not None:
        distance_waypoint = sensor_data.transform.location.distance(waypoint.transform.location)


def vehicle_control(waypoint_transform, vehicle_transform, target_speed) -> Tuple[float, float, float]:
    global steering_delta, speed_delta

    # calulate steering delta
    forward_vector = vehicle_transform.get_forward_vector()
    forward_vector = np.array([forward_vector.x, forward_vector.y, 0.0])
    waypoint_vector = np.array([waypoint_transform.location.x - vehicle_transform.location.x, waypoint_transform.location.y - vehicle_transform.location.y, 0.0])
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

    K_P_T = 0.5
    K_P_S = 0.5

    throttle = np.clip(speed_delta * K_P_T, 0, 1)
    steering = np.clip(steering_delta * K_P_S, -1, 1)
    return throttle, steering, 0.0


def main(ip: str):
    global waypoint
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(5.0)
        world = client.load_world("Town01")
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
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
        
        # RGB camera
        rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "1280")
        rgb_camera_bp.set_attribute("image_size_y", "720")
        rgb_camera_bp.set_attribute("fov", "110")
        rgb_camera_bp.set_attribute("sensor_tick", "0.02")
        relative_transform = carla.Transform(carla.Location(x=1.2, y=-0.5, z=1.7), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(display_image)

        # Segmentation camera
        segmentation_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        segmentation_bp.set_attribute("image_size_x", "1280")
        segmentation_bp.set_attribute("image_size_y", "720")
        segmentation_bp.set_attribute("fov", "110")
        segmentation_bp.set_attribute("sensor_tick", "1")
        segmentation = world.spawn_actor(segmentation_bp, relative_transform, vehicle)
        segmentation.listen(detect_traffic_light)

        # GNSS sensor
        gnss = blueprint_library.find("sensor.other.gnss")
        gnss = world.spawn_actor(gnss, carla.Transform(), vehicle)
        gnss.listen(gnss_sensor)

        waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        vehicle.set_transform(waypoint.transform)
        while 1:
            throttle, steering, brake = vehicle_control(waypoint.transform, vehicle.get_transform(), 10)
            control = carla.VehicleControl(
                throttle=throttle, steer=steering, brake=brake, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)
            # vehicle.set_transform(waypoint.transform)
            # if cv2.waitKey(20) & 0xFF == ord('q'):
            #     break
            pygame.display.flip()
            pygame.display.update()
            time.sleep(0.05)
            if distance_waypoint <= 5:
                waypoint = random.choice(waypoint.next(10))
                world.debug.draw_point(waypoint.transform.location)
            print(
                f"Position ({waypoint.transform.location.x:.3f}, {waypoint.transform.location.y:.3f},"
                f" {waypoint.transform.location.z:.3f}), Speed: {current_speed:.3f} m/s, speed delta: {speed_delta:.3f}, "
                f"steering delta: {steering_delta:.3f}, distance waypoint: {distance_waypoint}",
                end="\033[0K\r",
            )
    finally:
        try:
            vehicle.destroy()
        except UnboundLocalError:
            pass
        cv2.destroyAllWindows()
        print("\nCleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
