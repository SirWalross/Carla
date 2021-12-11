import argparse
import carla


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world("Town02_Opt")
    world.get_map().save_to_disk('map.xodr')


if __name__ == "__main__":
    main()