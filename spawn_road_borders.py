import argparse
import carla


def spawn_road_borders(world: carla.World, blueprint_library: carla.BlueprintLibrary) -> None:
    """
    Spawn traffic warnings to define route the car has to follow.

    Args:
        world (carla.World): The carla world
        blueprint_library (carla.BlueprintLibrary): The carla blueprint library
    """
    spawn_points = [
        (-5.4, 184.0, 180.0),
        (184.58, 239.0, 90.0),
        (143.0, 239.0, -90.0),
        (128.5, 189.5, 90.0),
        (191.6, 181.31, 180.0),
        (43.86, 244.53, 0.0),
        (48.92, 189.46, -90.0),
        (44.0, 298.5, 180.0),
    ]

    for point in spawn_points:
        spawn_point = carla.Transform(carla.Location(*point[:2], 0.22), carla.Rotation(yaw=point[2]))
        sign_bp = blueprint_library.find("static.prop.trafficwarning")
        world.spawn_actor(sign_bp, spawn_point)
