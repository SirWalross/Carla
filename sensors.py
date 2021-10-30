import numpy as np

class LidarData:
    def __init__(self, lidar_data):
        self.point_cloud = self._convert_to_point_cloud(lidar_data)
        self.timestamp = lidar_data.timestamp

        # Calculate as only x-axis

        # self.distances = np.linalg.norm(self.point_cloud["position"], axis=1)
        self.distances = self.point_cloud["position"][:, 0]
        self.object_indices = np.unique(self.point_cloud["object_index"])

    @staticmethod
    def _convert_to_point_cloud(lidar_data):
        dtype = np.dtype(
            [("position", np.float32, (3,)), ("cos_angle", np.float32), ("object_index", np.uint32), ("tag", np.uint32)]
        )
        p_cloud = np.frombuffer(lidar_data.raw_data, dtype=dtype)
        tags = p_cloud["tag"]
        positions = np.array(p_cloud["position"])
        return p_cloud[
            np.logical_and.reduce(
                (
                    np.logical_or.reduce((tags == 4, tags == 10)),
                    positions[:, 0] > 1,
                    positions[:, 1] < 2,
                    positions[:, 1] > -2,
                )
            )
        ]
    
    def query_object_index(self, object_index) -> float:
        """ Return the distance to the nearest point with object index `object_index` """
        object_indices = self.point_cloud['object_index'] != object_index
        distances = np.copy(self.distances)
        distances[object_indices] = np.inf
        if len(distances) > 0:
            return distances[np.argmin(distances)]
        else:
            return np.inf