import argparse
import math
import random
import time
from typing import List, Tuple
import numpy as np
import carla
import cv2
from numpy.lib.function_base import average
import pygame
from pygame.locals import *
from sensorclasses import LidarData

from carla import ColorConverter as cc

pygame.init()
screen = pygame.display.set_mode((1280, 720))
obstacles: List[Tuple[float, float]] = []
last_lidar_data: LidarData = None       


def lidar_sensor(lidar_data):
    global obstacles, last_lidar_data
    lidar_data = LidarData(lidar_data)
    if last_lidar_data is not None:
        obstacles = []
        for object_index in lidar_data.object_indices:
            dist_new = lidar_data.query_object_index(object_index)
            dist_old = last_lidar_data.query_object_index(object_index)
            print(f"index: {object_index}, dist_old: {dist_old}, dist_new: {dist_new}")
            if dist_old != np.inf and dist_new != np.inf:
                obstacles.append((dist_new, (dist_new - dist_old)/(lidar_data.timestamp - last_lidar_data.timestamp)))
    last_lidar_data = lidar_data


def rgb_sensor(image):
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]

    # convert from rgb to bgr
    array = np.array(array[:, :, ::-1])

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def main(ip: str):
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        # world = client.load_world("Town01_Opt", carla.MapLayer.NONE)
        world = client.get_world()

        # Settings
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

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
        rgb_camera_bp.set_attribute("sensor_tick", "0.1")
        relative_transform = carla.Transform(carla.Location(x=-6.0, y=0.0, z=3.0), carla.Rotation(pitch=-15.0))
        rgb_camera = world.spawn_actor(rgb_camera_bp, relative_transform, vehicle)
        rgb_camera.listen(rgb_sensor)

        # Lidar sensor
        lidar_transform = carla.Transform(carla.Location(z=1.9))
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("upper_fov", "30.0")
        lidar_bp.set_attribute("lower_fov", "-25.0")
        lidar_bp.set_attribute("channels", "64.0")
        lidar_bp.set_attribute("range", "40.0")
        lidar_bp.set_attribute("points_per_second", "100000.0")
        lidar_bp.set_attribute("rotation_frequency", "10")
        lidar = world.spawn_actor(lidar_bp, lidar_transform, vehicle)
        lidar.listen(lidar_sensor)

        waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        vehicle.set_transform(waypoint.transform)
        while 1:
            control = carla.VehicleControl(
                throttle=0.1, steer=-0.1, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)
            world.tick()
            pygame.display.flip()
            pygame.display.update()
    finally:
        try:
            vehicle.destroy()
            rgb_camera.destroy()
            lidar.destroy()
        except UnboundLocalError:
            pass
        pygame.quit()
        print("\nCleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
