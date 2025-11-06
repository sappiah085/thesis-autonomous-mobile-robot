# slam.py
import numpy as np
from config import MAP_SIZE, MAP_RESOLUTION, PARTICLE_COUNT, LIDAR_MIN_RANGE

class SLAM:
    def __init__(self):
        self.map_size = int(MAP_SIZE / MAP_RESOLUTION)
        self.map = np.full((self.map_size, self.map_size), 0.5)  # Start with unknown (0.5)
        self.particles = [(0.0, 0.0, 0.0)] * PARTICLE_COUNT  # (x, y, theta)
        self.weights = np.ones(PARTICLE_COUNT) / PARTICLE_COUNT
        self.min_range = LIDAR_MIN_RANGE  # Fixed the attribute error

    def update(self, robot_pose, lidar_ranges, lidar_angles):
        """Update map and particles with new LiDAR scan."""
        # Update particles (simplified motion model with noise)
        for i in range(PARTICLE_COUNT):
            x, y, theta = self.particles[i]
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            theta += np.random.normal(0, 0.01)
            self.particles[i] = (x, y, theta)

        # Resample particles based on weights (simplified)
        self.weights /= np.sum(self.weights)

        # Update map with best particle
        best_idx = np.argmax(self.weights)
        x, y, theta = self.particles[best_idx]
        for r, angle in zip(lidar_ranges, lidar_angles):
            if self.min_range < r < MAP_SIZE:
                global_angle = theta + angle
                px = x + r * np.cos(global_angle)
                py = y + r * np.sin(global_angle)
                mx = int((px + MAP_SIZE / 2) / MAP_RESOLUTION)
                my = int((py + MAP_SIZE / 2) / MAP_RESOLUTION)
                if 0 <= mx < self.map_size and 0 <= my < self.map_size:
                    self.map[mx, my] += 0.1  # Increment occupancy for hit

                # Mark free space along the ray
                for dist in np.linspace(0, r, int(r / MAP_RESOLUTION)):
                    fx = x + dist * np.cos(global_angle)
                    fy = y + dist * np.sin(global_angle)
                    fm_x = int((fx + MAP_SIZE / 2) / MAP_RESOLUTION)
                    fm_y = int((fy + MAP_SIZE / 2) / MAP_RESOLUTION)
                    if 0 <= fm_x < self.map_size and 0 <= fm_y < self.map_size:
                        self.map[fm_x, fm_y] -= 0.05  # Decrement for free space

        # Clip map values to [0, 1]
        self.map = np.clip(self.map, 0, 1)

    def is_free(self, x, y):
        """Check if a point is free in the map."""
        mx = int((x + MAP_SIZE / 2) / MAP_RESOLUTION)
        my = int((y + MAP_SIZE / 2) / MAP_RESOLUTION)
        if 0 <= mx < self.map_size and 0 <= my < self.map_size:
            return self.map[mx, my] < 0.5
        return True
