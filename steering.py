import argparse
import math
import random
import time
from typing import Tuple
import numpy as np
import carla
import cv2
from numpy.lib.function_base import average
import pygame
from pygame.locals import *

from carla import ColorConverter as cc

pygame.init()
screen = pygame.display.set_mode((1280, 720))

current_speed = 0.0  # m/s
prev_sensor_data = None
steering_delta = 0
speed_delta = 0
distance_waypoint = 0
waypoint = None
traffic_light = None
traffic_sign = None
detected_red_traffic_light = False
waypoint_deadzone = 0
lidar = None

last_image = None

def lidar_sensor(lidar_data): 
    p_cloud_size = len(lidar_data)
    p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    intensity = np.array(p_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(p_cloud[:, :3]).T

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
    
    print(local_lidar_points.T[0])

    # This (4, 4) matrix transforms the points from lidar space to world space.
    lidar_2_world = lidar.get_transform().get_matrix()

    # Transform the points from lidar space to world space.
    world_points = np.dot(lidar_2_world, local_lidar_points)


def rgb_sensor(image):
    global detected_red_traffic_light, last_image
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    # convert from rgb to bgr
    array = np.array(array[:, :, ::-1])
    last_image = np.copy(array)

    if traffic_light is not None:
        area = cv2.contourArea(traffic_light)

        mask = np.zeros_like(array[:, :, 0])
        cv2.drawContours(mask, [traffic_light], -1, (255, 255, 255), -1)
        image = cv2.bitwise_and(array, array, mask=mask)

        # average red color in image
        average_colours = np.array([np.average(image[:, :, 0]), np.average(image[:, :, 1]), np.average(image[:, :, 2])])
        average_colours = average_colours / np.sum(average_colours)

        point = (max(traffic_light[:, 0, 0]), min(traffic_light[:, 0, 1]))
        if average_colours[0] > 0.37:
            cv2.drawContours(array, [traffic_light], -1, (255, 0, 0), 1)
            cv2.putText(array, f"{average_colours[0]:.2f},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        else:
            cv2.drawContours(array, [traffic_light], -1, (0, 255, 0), 1)
            cv2.putText(array, f"{average_colours[0]:.2f},{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        if area > 1800 and average_colours[0] > 0.37:
            detected_red_traffic_light = True
        else:
            detected_red_traffic_light = False
    else:
        detected_red_traffic_light = False

    if traffic_sign is not None:
        area = cv2.contourArea(traffic_sign)

        mask = np.zeros_like(array[:, :, 0])
        cv2.drawContours(mask, [traffic_sign], -1, (255, 255, 255), -1)
        image = cv2.bitwise_and(array, array, mask=mask)

        # TODO: use image to read traffic sign

        point = (max(traffic_sign[:, 0, 0]), min(traffic_sign[:, 0, 1]))
        cv2.drawContours(array, [traffic_sign], -1, (255, 0, 0), 1)
        cv2.putText(array, f"30,{area:.0f}", point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def segmentation_sensor(image):
    global traffic_light, traffic_sign
    image.convert(cc.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

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
    else:
        traffic_light = None

    # Traffic sign
    traffic_image = np.copy(array)
    # preparing the mask to overlay
    mask = cv2.inRange(traffic_image, np.array([219, 219, 0]), np.array([221, 221, 1]))

    base_colour = np.full_like(traffic_image, np.array([255, 255, 255]))
    traffic_image = cv2.bitwise_and(base_colour, base_colour, mask=mask)

    # convert image to greyscale
    traffic_image = cv2.cvtColor(traffic_image, cv2.COLOR_BGR2GRAY)

    # contour detection
    contours, _ = cv2.findContours(traffic_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        traffic_sign = max(contours, key=cv2.contourArea)
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


def vehicle_control(waypoint_transform, vehicle_transform, target_speed) -> Tuple[float, float, float]:
    if detected_red_traffic_light:
        return 0.0, 0.0, 1.0
    global steering_delta, speed_delta

    # calulate steering delta
    forward_vector = vehicle_transform.get_forward_vector()
    forward_vector = np.array([forward_vector.x, forward_vector.y, 0.0])
    waypoint_vector = np.array(
        [
            waypoint_transform.location.x - vehicle_transform.location.x,
            waypoint_transform.location.y - vehicle_transform.location.y,
            0.0,
        ]
    )
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

    throttle = np.clip(speed_delta * K_P_T, 0, 0.5)
    steering = np.clip(steering_delta * K_P_S, -1, 1)
    return throttle, steering, 0.0


def main(ip: str):
    global waypoint, waypoint_deadzone, lidar
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02_Opt", carla.MapLayer.NONE)
        world.unload_map_layer(carla.MapLayer.Buildings)
        world.unload_map_layer(carla.MapLayer.Decals)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Foliage)
        world.unload_map_layer(carla.MapLayer.Walls)
        world.unload_map_layer(carla.MapLayer.Props)
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
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
        print("Spawned vehicle")
        # RGB camera
        rgb_camera_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_camera_bp.set_attribute("image_size_x", "1280")
        rgb_camera_bp.set_attribute("image_size_y", "720")
        rgb_camera_bp.set_attribute("fov", "60")
        rgb_camera_bp.set_attribute("sensor_tick", "0.02")
        relative_transform = carla.Transform(carla.Location(x=1.2, y=-0.5, z=1.7), carla.Rotation(yaw=0))
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(rgb_sensor)

        # Segmentation camera
        segmentation_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        segmentation_bp.set_attribute("image_size_x", "1280")
        segmentation_bp.set_attribute("image_size_y", "720")
        segmentation_bp.set_attribute("fov", "60")
        segmentation_bp.set_attribute("sensor_tick", "0.02")
        segmentation = world.spawn_actor(segmentation_bp, relative_transform, vehicle)
        segmentation.listen(segmentation_sensor)

        # Lidar sensor
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_bp.set_attribute('upper_fov', '30.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('points_per_second', '100000.0')
        lidar = world.spawn_actor(lidar_bp, relative_transform, vehicle)
        lidar.listen(lidar_sensor)

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
            pygame.display.flip()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            if distance_waypoint <= 1 and waypoint_deadzone <= 0:
                waypoint = random.choice(waypoint.next(2))
                world.debug.draw_point(waypoint.transform.location)
                waypoint_deadzone = 2
            elif waypoint_deadzone != 0:
                waypoint_deadzone -= 1
            print(
                f"throttle: {throttle:.3f}, steering: {steering:.3f}, brake: {brake:.3f}, speed:"
                f" {current_speed:.3f} m/s, speed delta: {speed_delta:.3f}, steering delta: {steering_delta:.3f},"
                f" distance waypoint: {distance_waypoint:.3f}",
                end="\033[0K\r",
            )
    finally:
        try:
            vehicle.destroy()
            rgb_camera.destroy()
            segmentation.destroy()
            gnss.destroy()
        except UnboundLocalError:
            pass
        pygame.quit()
        print("\nCleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
