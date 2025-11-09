# ============================================================================
# OCCUPANCY GRID MAP – DWA-ready
# ============================================================================
import math
import numpy as np
from collections import deque
from typing import Tuple, List
from .config import RobotConfig


class OccupancyGrid:
    """2-D occupancy grid with fast distance-to-obstacle queries."""

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.res = cfg.map_resolution                     # metres per cell
        self.size = cfg.map_size                          # (width, height) cells
        self.origin = np.array([self.size[0] // 2, self.size[1] // 2])

        # probability grid – 0.0 = free, 0.5 = unknown, 1.0 = occupied
        self.prob = np.full(self.size, 0.5, dtype=np.float32)

        # pre-allocated distance map (metres)
        self.dist = np.full(self.size, np.inf, dtype=np.float32)

        print(f"Occupancy grid initialized: {self.size[0]}×{self.size[1]} cells @ {self.res:.2f} m/cell")

    # --------------------------------------------------------------------- #
    #  Coordinate conversion helpers
    # --------------------------------------------------------------------- #
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """World → grid index (i, j) – note: i = column, j = row."""
        i = int(round(x / self.res)) + self.origin[0]
        j = int(round(y / self.res)) + self.origin[1]
        return i, j

    def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
        """Grid index → world centre of the cell."""
        x = (i - self.origin[0]) * self.res
        y = (j - self.origin[1]) * self.res
        return x, y

    # --------------------------------------------------------------------- #
    #  Map update (ray-tracing + Bresenham)
    # --------------------------------------------------------------------- #
    def update_from_scan(self,
                         robot_x: float, robot_y: float, robot_theta: float,
                         ranges: np.ndarray, angles: np.ndarray):
        """
        Bayesian update from a single LIDAR sweep.

        * angles are already in the robot frame ([-π, π])
        * robot_theta is the robot heading in the world frame
        """
        robot_i, robot_j = self.world_to_grid(robot_x, robot_y)

        # -----------------------------------------------------------------
        # 1. Ray-trace every valid beam
        # -----------------------------------------------------------------
        for r, a in zip(ranges, angles):
            if not (0.10 <= r <= self.cfg.max_lidar_range):   # filter noise / out-of-range
                continue

            world_angle = robot_theta + a
            end_x = robot_x + r * math.sin(world_angle)
            end_y = robot_y + r * math.cos(world_angle)
            end_i, end_j = self.world_to_grid(end_x, end_y)

            # ---- occupied endpoint ------------------------------------------------
            if 0 <= end_i < self.size[0] and 0 <= end_j < self.size[1]:
                self.prob[end_j, end_i] = min(0.98, self.prob[end_j, end_i] + 0.20)

            # ---- free cells along the ray (exclude endpoint) --------------------
            ray = self._bresenham(robot_i, robot_j, end_i, end_j)
            for i, j in ray[:-1]:
                if 0 <= i < self.size[0] and 0 <= j < self.size[1]:
                    self.prob[j, i] = max(0.02, self.prob[j, i] - 0.08)

        # -----------------------------------------------------------------
        # 2. Re-compute the distance map (used by DWA clearance)
        # -----------------------------------------------------------------
        self._build_distance_map()

    # --------------------------------------------------------------------- #
    #  Bresenham line – returns list of (i, j) cells
    # --------------------------------------------------------------------- #
    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham line – integer cell coordinates."""
        points = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    # --------------------------------------------------------------------- #
    #  Distance-map (BFS from every occupied cell)
    # --------------------------------------------------------------------- #
    def _build_distance_map(self):
        """Fill `self.dist` with Euclidean distance (metres) to nearest occupied cell."""
        occ_thresh = 0.65                         # anything ≥ this is considered occupied
        self.dist.fill(np.inf)

        q = deque()
        # seed queue with all occupied cells
        occ = np.where(self.prob >= occ_thresh)
        for j, i in zip(occ[0], occ[1]):
            self.dist[j, i] = 0.0
            q.append((i, j))

        # 4-connectivity (up/down/left/right)
        neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while q:
            ci, cj = q.popleft()
            d = self.dist[cj, ci]

            for di, dj in neighbours:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.size[0] and 0 <= nj < self.size[1]:
                    new_d = d + self.res * (1.0 if di == 0 or dj == 0 else math.sqrt(2))
                    if new_d < self.dist[nj, ni]:
                        self.dist[nj, ni] = new_d
                        q.append((ni, nj))

    # --------------------------------------------------------------------- #
    #  Public queries used by the planner
    # --------------------------------------------------------------------- #
    def is_occupied(self, x: float, y: float, thresh: float = 0.7) -> bool:
        """Return True if the cell is considered occupied (or out of bounds)."""
        i, j = self.world_to_grid(x, y)
        if not (0 <= i < self.size[0] and 0 <= j < self.size[1]):
            return True
        return self.prob[j, i] > thresh

    def get_min_dist_to_obstacle(self, x: float, y: float) -> float:
        """Distance (metres) to nearest occupied cell – safe even outside map."""
        i, j = self.world_to_grid(x, y)
        if not (0 <= i < self.size[0] and 0 <= j < self.size[1]):
            # outside → distance to map border (conservative)
            border_i = max(0, min(i, self.size[0] - 1))
            border_j = max(0, min(j, self.size[1] - 1))
            bx, by = self.grid_to_world(border_i, border_j)
            return math.hypot(x - bx, y - by)
        return self.dist[j, i]

    # --------------------------------------------------------------------- #
    #  Visualisation
    # --------------------------------------------------------------------- #
    def draw(self, ax):
        """Draw probability map (black = occupied, white = free)."""
        x0, y0 = self.grid_to_world(0, 0)
        x1, y1 = self.grid_to_world(self.size[0] - 1, self.size[1] - 1)
        ax.imshow(
            self.prob,
            cmap="gray_r",
            origin="lower",
            extent=[x0, x1, y0, y1],
            vmin=0,
            vmax=1,
            alpha=0.6,
            interpolation="nearest",
        )

    def draw_debug(self, ax, robot_x: float = 0.0, robot_y: float = 0.0):
        """Full debug overlay (origin, axes, robot)."""
        self.draw(ax)

        # world origin
        ax.plot(0, 0, "g+", markersize=20, markeredgewidth=3, label="World Origin")

        # grid centre (0,0 in world)
        gx, gy = self.grid_to_world(self.origin[0], self.origin[1])
        ax.plot(gx, gy, "bx", markersize=15, markeredgewidth=2, label="Grid Origin")

        # robot
        ax.plot(robot_x, robot_y, "ro", markersize=10, label="Robot")

        # axes
        ax.arrow(0, 0, 1, 0, head_width=0.2, head_length=0.2, fc="red", ec="red")
        ax.text(1.2, 0, "X", fontsize=12, color="red")
        ax.arrow(0, 0, 0, 1, head_width=0.2, head_length=0.2, fc="green", ec="green")
        ax.text(0, 1.2, "Y", fontsize=12, color="green")

        ax.legend()
