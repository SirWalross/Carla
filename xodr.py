import argparse
import carla
import sys


def main(ip):
    client = carla.Client(ip, 2000)
    client.set_timeout(10.0)

    # world = client.load_world("Town02_Opt")
    # world.get_map().save_to_disk('map.xodr')

    with open("OpenDriveMaps/map07.xodr", encoding="utf-8") as od_file:
        try:
            data = od_file.read()
        except OSError:
            print("file could not be readed.")
            sys.exit()
    vertex_distance = 2.0  # in meters
    max_road_length = 500.0  # in meters
    wall_height = 1.0  # in meters
    extra_width = 0.6  # in meters
    world = client.generate_opendrive_world(
        data,
        carla.OpendriveGenerationParameters(
            vertex_distance=vertex_distance,
            max_road_length=max_road_length,
            wall_height=wall_height,
            additional_width=extra_width,
            smooth_junctions=True,
            enable_mesh_visibility=True,
        ),
    )
    # world = client.load_world("Town02")
    # client = carla.Client('localhost', 2000)

    map = world.get_map()
    while True:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
