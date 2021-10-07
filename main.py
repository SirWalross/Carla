import carla
import random

client = carla.Client('localhost', 2000)
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
# camera_bp = blueprint_library.find('sensor.camera.rgb')
# relative_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0))
# camera = world.spawn_actor(camera_bp, relative_transform, attach_to=vehicle, attachment=carla.AttachmentType.Rigid)
