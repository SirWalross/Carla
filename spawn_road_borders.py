import carla


def spawn_road_borders(world: carla.World, blueprint_library: carla.BlueprintLibrary) -> None:
    """
    Spawn traffic warnings to define route the car has to follow.

    Args:
        world (carla.World): The carla world
        blueprint_library (carla.BlueprintLibrary): The carla blueprint library
    """
    spawn_points = [
        (-5.4, 199.25, 0.0),
        (52.73, 189.60, -90.0),
        (52.02, 238.84, -90.0),
        (35.89, 304.55, 90.0),
        (191.71, 230.34, 180.0),
        (124.97, 238.92, 90.0),
        (125.66, 189.82, 90.0),
        (191.71, 198.21, 0.0)
    ]

    sign_bp = blueprint_library.find("static.prop.trafficwarning")
    for point in spawn_points:
        spawn_point = carla.Transform(carla.Location(*point[:2], 0.22), carla.Rotation(yaw=point[2]))
        world.spawn_actor(sign_bp, spawn_point)
