import carla
import random
import pygame, sys, os
import numpy as np
from pygame.locals import *
from carla import ColorConverter as cc
pygame.init()
screen = pygame.display.set_mode((1280, 720))

def do_something(image):
    image.convert(cc.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    screen.blit(surface, (0, 0))

try:
    client = carla.Client('172.23.48.1', 2000)
    client.set_timeout(10.0)
    world = client.load_world('Town02')
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    spawn_point.z = 10
    spawn_point.location = world.get_random_location_from_navigation()
    vehicle_bp = blueprint_library.find('vehicle.mercedes.sprinter')
    print(vehicle_bp)
    print(spawn_point)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    camera_bp = blueprint_library.find('sensor.camera.rgb')

    # Modify the attributes of the blueprint to set image resolution and field of view.
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    camera_bp.set_attribute('sensor_tick', '0.1')

    relative_transform = carla.Transform(carla.Location(x=1, y=-0.5, z=1.7), carla.Rotation(yaw=0))
    camera = world.spawn_actor(camera_bp, relative_transform, vehicle)
    camera.listen(lambda data: do_something(data))
    while 1:
        pygame.display.flip()
        pygame.display.update()
except KeyboardInterrupt as e:
    vehicle.destroy()
    pygame.quit()
    print("\nFinished")
    raise e