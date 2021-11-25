import argparse
import carla


def main(ip: str):
    client = carla.Client(ip, 2000)
    client.set_timeout(10.0)
    world = client.load_world("Town07")
    world.get_map().save_to_disk('map07.xodr')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", nargs="?", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)")
    args = parser.parse_args()
    main(args.host)
