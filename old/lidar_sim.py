# lidar_sim.py
import numpy as np
from config import LIDAR_MAX_RANGE, LIDAR_MIN_RANGE, LIDAR_ANGULAR_RES, LIDAR_NOISE_STD, MAP_SIZE

class Lidar:
    def __init__(self):
        self.max_range = LIDAR_MAX_RANGE
        self.min_range = LIDAR_MIN_RANGE
        self.angular_resolution = LIDAR_ANGULAR_RES
        self.noise_std = LIDAR_NOISE_STD
        # Realistic environment: walls + rectangular obstacles
        self.environment = [
            # Walls: (x1, y1, x2, y2, type='line')
            ((-MAP_SIZE/2, -MAP_SIZE/2, -MAP_SIZE/2, MAP_SIZE/2), 'line'),  # Left wall
            ((-MAP_SIZE/2, MAP_SIZE/2, MAP_SIZE/2, MAP_SIZE/2), 'line'),    # Top wall
            ((MAP_SIZE/2, MAP_SIZE/2, MAP_SIZE/2, -MAP_SIZE/2), 'line'),    # Right wall
            ((-MAP_SIZE/2, -MAP_SIZE/2, MAP_SIZE/2, -MAP_SIZE/2), 'line'),  # Bottom wall
            # Rectangular obstacles: (x_min, y_min, x_max, y_max, type='rect')
            ((2, 2, 3, 3), 'rect'),
            ((-2, -2, -1, -1), 'rect'),
        ]

    def get_scan(self, robot_pose, environment):
        x, y, theta = robot_pose
        angles = np.deg2rad(np.arange(-180, 180, self.angular_resolution))
        ranges = np.full(len(angles), self.max_range)

        for i, angle in enumerate(angles):
            ray_angle = theta + angle
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)
            min_dist = self.max_range

            for obj, obj_type in self.environment:
                if obj_type == 'line':
                    x1, y1, x2, y2 = obj
                    denom = ray_dx * (y2 - y1) - ray_dy * (x2 - x1)
                    if abs(denom) > 1e-6:
                        t = ((x1 - x) * (y2 - y1) - (y1 - y) * (x2 - x1)) / denom
                        u = -((x1 - x) * ray_dy - (y1 - y) * ray_dx) / denom
                        if 0 <= t <= 1 and u > 0:
                            dist = u
                            min_dist = min(min_dist, dist)
                elif obj_type == 'rect':
                    x_min, y_min, x_max, y_max = obj
                    edges = [
                        (x_min, y_min, x_min, y_max),  # Left
                        (x_min, y_max, x_max, y_max),  # Top
                        (x_max, y_max, x_max, y_min),  # Right
                        (x_min, y_min, x_max, y_min)   # Bottom
                    ]
                    for edge in edges:
                        x1, y1, x2, y2 = edge
                        denom = ray_dx * (y2 - y1) - ray_dy * (x2 - x1)
                        if abs(denom) > 1e-6:
                            t = ((x1 - x) * (y2 - y1) - (y1 - y) * (x2 - x1)) / denom
                            u = -((x1 - x) * ray_dy - (y1 - y) * ray_dx) / denom
                            if 0 <= t <= 1 and u > 0:
                                dist = u
                                min_dist = min(min_dist, dist)

            ranges[i] = max(min(min_dist, self.max_range), self.min_range)

        ranges += np.random.normal(0, self.noise_std, len(ranges))
        ranges = np.clip(ranges, self.min_range, self.max_range)
        return ranges, angles

    def _get_real_scan(self):
        # Placeholder for real LiDAR
        pass
