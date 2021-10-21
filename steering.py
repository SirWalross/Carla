import argparse
import math
import random
import time
from typing import Tuple
import numpy as np
import carla
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


def display_image(image):
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))


def imu_sensor(sensor_data):
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
        world = client.load_world("Town05")
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        while True:
            try:
                spawn_point = random.choice(spawn_points)
                spawn_point.z = 10
                spawn_point.location = world.get_random_location_from_navigation()
                vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
                print(vehicle_bp)
                print(spawn_point)
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                continue
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "110")
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute("sensor_tick", "0.02")

        relative_transform = carla.Transform(carla.Location(x=1.2, y=-0.5, z=1.7), carla.Rotation(yaw=0))
        camera = world.spawn_actor(camera_bp, relative_transform, vehicle)
        camera.listen(display_image)

        imu = blueprint_library.find("sensor.other.imu")
        imu = world.spawn_actor(imu, carla.Transform(), vehicle)
        imu.listen(imu_sensor)

        waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        vehicle.set_transform(waypoint.transform)
        while 1:
            throttle, steering, brake = vehicle_control(waypoint.transform, vehicle.get_transform(), 10)
            control = carla.VehicleControl(
                throttle=throttle, steer=steering, brake=brake, hand_brake=False, reverse=False, manual_gear_shift=False
            )
            vehicle.apply_control(control)
            # vehicle.set_transform(waypoint.transform)
            pygame.display.flip()
            pygame.display.update()
            time.sleep(0.02)
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
        pygame.quit()
        print("\nCleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
