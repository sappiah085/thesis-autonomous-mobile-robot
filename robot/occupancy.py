from .config import RobotConfig
from typing import Tuple
import numpy as np

# ============================================================================
# OCCUPANCY GRID MAP - FIXED COORDINATE SYSTEM
# ============================================================================
class OccupancyGrid:
    """
    2D occupancy grid for mapping obstacles

    Coordinate Convention:
    - World frame: theta=0 points in +Y direction (North), increases CCW
    - This matches your LIDAR convention where angle=0 is forward (+Y)
    """
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.resolution = cfg.grid_resolution
        self.size = cfg.grid_size

        # Calculate grid dimensions
        self.cells = int(self.size / self.resolution)
        self.grid = np.zeros((self.cells, self.cells), dtype=np.float32)

        # Origin at center
        self.origin_x = self.size / 2
        self.origin_y = self.size / 2

        print(f"✓ Occupancy Grid: {self.cells}x{self.cells} cells, {self.resolution}m resolution")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        gx = int((x + self.origin_x) / self.resolution)
        gy = int((y + self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates"""
        x = gx * self.resolution - self.origin_x
        y = gy * self.resolution - self.origin_y
        return x, y

    def is_valid(self, gx: int, gy: int) -> bool:
        """Check if grid coordinates are valid"""
        return 0 <= gx < self.cells and 0 <= gy < self.cells

    def update_from_scan(self, robot_x: float, robot_y: float, robot_theta: float,
                        ranges: np.ndarray, angles: np.ndarray):
        """
        Update occupancy grid from LIDAR scan

        CRITICAL FIX: Uses correct coordinate convention where theta=0 points in +Y
        - x = r * sin(world_angle)  [rightward displacement]
        - y = r * cos(world_angle)  [forward displacement]
        """
        # Decay existing occupancy (forgetting factor for dynamic environments)
        self.grid *= 0.95

        for r, angle in zip(ranges, angles):
            if r < 0.1 or r > self.cfg.max_lidar_range:
                continue

            # FIXED: Correct coordinate transformation
            # angle is in LIDAR frame (0 = forward = +Y)
            # robot_theta is robot heading in world frame
            world_angle = robot_theta + angle

            # Convert polar to Cartesian with correct convention:
            # theta=0 points in +Y direction (North/Forward)
            obs_x = robot_x + r * np.sin(world_angle)  # Rightward component
            obs_y = robot_y + r * np.cos(world_angle)  # Forward component

            gx, gy = self.world_to_grid(obs_x, obs_y)

            if self.is_valid(gx, gy):
                # Mark as occupied (incremental update with saturation)
                self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + 0.3)

    def is_occupied(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """Check if position is occupied (for DWA collision checking)"""
        gx, gy = self.world_to_grid(x, y)
        if not self.is_valid(gx, gy):
            return True  # Out of bounds = occupied
        return self.grid[gy, gx] > threshold

    def get_min_dist_to_obstacle(self, x: float, y: float,
                                  search_radius: float = 1.0) -> float:
        """Get distance to closest obstacle (for DWA clearance evaluation)"""
        gx, gy = self.world_to_grid(x, y)
        search_cells = int(search_radius / self.resolution)
        min_dist = search_radius

        for dx in range(-search_cells, search_cells + 1):
            for dy in range(-search_cells, search_cells + 1):
                nx, ny = gx + dx, gy + dy
                if self.is_valid(nx, ny) and self.grid[ny, nx] > 0.5:
                    wx, wy = self.grid_to_world(nx, ny)
                    dist = np.sqrt((x - wx)**2 + (y - wy)**2)
                    min_dist = min(min_dist, dist)

        return min_dist

    def draw(self, ax):
        """Draw probability map (black = occupied, white = free)"""
        x0, y0 = self.grid_to_world(0, 0)
        x1, y1 = self.grid_to_world(self.cells - 1, self.cells - 1)

        ax.imshow(
            self.grid,
            cmap="gray_r",
            origin="lower",
            extent=[x0, x1, y0, y1],
            vmin=0,
            vmax=1,
            alpha=0.6,
            interpolation="nearest",
        )
