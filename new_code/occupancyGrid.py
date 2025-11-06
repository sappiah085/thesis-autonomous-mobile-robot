# ============================================================================
# OCCUPANCY GRID MAP
# ============================================================================
import math
from typing import List, Tuple
from .configuration import RobotConfig
import numpy as np

class OccupancyGrid:
    """2D occupancy grid for mapping obstacles"""
    def __init__(self, cfg: RobotConfig):
        self.resolution = cfg.map_resolution
        self.size = cfg.map_size
        self.data = np.ones(cfg.map_size) * 0.5  # Unknown = 0.5
        self.origin = np.array([cfg.map_size[0] // 2, cfg.map_size[1] // 2])
        print(f"✓ Occupancy grid initialized: {cfg.map_size[0]}x{cfg.map_size[1]} cells")

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        grid_x = int(x / self.resolution) + self.origin[0]
        grid_y = int(y / self.resolution) + self.origin[1]
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates"""
        x = (grid_x - self.origin[0]) * self.resolution
        y = (grid_y - self.origin[1]) * self.resolution
        return x, y

    def update_from_scan(self, robot_x: float, robot_y: float, robot_theta: float,
                        ranges: np.ndarray, angles: np.ndarray):
        """
        Update map using ray tracing from LIDAR scan

        CRITICAL: angles are already in robot frame (from LIDAR)
        We need to transform to world frame using robot pose
        """
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)

        for r, angle in zip(ranges, angles):
            # Filter invalid readings
            if r < 0.1 or r > 4.5 or not np.isfinite(r):
                continue

            # FIXED: Transform angle from robot frame to world frame
            # angle is already in radians from LIDAR [-pi, pi]
            # robot_theta is robot's orientation in world frame
            world_angle = robot_theta + angle

            # Calculate endpoint in world coordinates
            # Use sin for X and cos for Y (standard robot coordinate frame)
            end_x = robot_x + r * np.sin(world_angle)
            end_y = robot_y + r * np.cos(world_angle)

            end_gx, end_gy = self.world_to_grid(end_x, end_y)

            # Mark endpoint as occupied (higher confidence)
            if 0 <= end_gx < self.size[0] and 0 <= end_gy < self.size[1]:
                self.data[end_gy, end_gx] = min(0.98, self.data[end_gy, end_gx] + 0.2)

            # Ray tracing: mark cells along the ray as free
            ray_cells = self._bresenham(robot_gx, robot_gy, end_gx, end_gy)
            for rx, ry in ray_cells[:-1]:  # Exclude endpoint (it's occupied)
                if 0 <= rx < self.size[0] and 0 <= ry < self.size[1]:
                    self.data[ry, rx] = max(0.01, self.data[ry, rx] - 0.08)

    def _bresenham(self, x0: int, y0: int, x1: int, y1: int) -> list:
        """Bresenham's line algorithm for ray tracing"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def is_occupied(self, x: float, y: float, threshold: float = 0.7) -> bool:
        """Check if world position is occupied"""
        gx, gy = self.world_to_grid(x, y)
        if not (0 <= gx < self.size[0] and 0 <= gy < self.size[1]):
            return True  # Out of bounds = occupied
        return self.data[gy, gx] > threshold

    def draw(self, ax):
        """
        Draw occupancy grid on matplotlib axis

        Args:
            ax: matplotlib axis object to draw on
        """
        # Get world coordinates for grid extent
        x_min, y_min = self.grid_to_world(0, 0)
        x_max, y_max = self.grid_to_world(self.size[0]-1, self.size[1]-1)

        # Draw grid (0=free, 0.5=unknown, 1=occupied)
        # CRITICAL: Need to flip the data array because imshow treats first row as top
        # but our grid treats first row as bottom (y increases upward in world frame)
        ax.imshow(self.data, cmap='gray_r', origin='lower',
          extent=[x_min, x_max, y_min, y_max],
          vmin=0, vmax=1, alpha=0.5, interpolation='nearest')

    def draw_debug(self, ax, robot_x: float = 0, robot_y: float = 0):
        """
        Draw occupancy grid with debug information
        Shows grid origin, robot position, and coordinate axes

        Args:
            ax: matplotlib axis object
            robot_x, robot_y: robot position in world frame
        """
        self.draw(ax)

        # Draw origin
        ax.plot(0, 0, 'g+', markersize=20, markeredgewidth=3, label='World Origin')

        # Draw grid origin in world coordinates
        origin_world_x, origin_world_y = self.grid_to_world(self.origin[0], self.origin[1])
        ax.plot(origin_world_x, origin_world_y, 'bx', markersize=15,
                markeredgewidth=2, label='Grid Origin')

        # Draw robot
        ax.plot(robot_x, robot_y, 'ro', markersize=10, label='Robot')

        # Draw coordinate axes
        ax.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.5)
        ax.text(1.2, 0, 'X', fontsize=12, color='red')
        ax.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.5)
        ax.text(0, 1.2, 'Y', fontsize=12, color='green')

        ax.legend()
