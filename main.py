import carla
import random
import argparse


def do_something(data):
    print(data)


def main(ip: str):
    try:
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")
        map = world.get_map()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        spawn_point.z = 10
        spawn_point.location = world.get_random_location_from_navigation()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        print(vehicle_bp)
        print(spawn_point)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "110")
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute("sensor_tick", "10.0")

        relative_transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(yaw=0))
        camera = world.spawn_actor(camera_bp, relative_transform, vehicle)
        camera.listen(lambda data: do_something(data))

        waypoint_list = map.generate_waypoints(2.0)
        while 1:
            pass
    finally:
        vehicle.destroy()
        print("Cleaned up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
