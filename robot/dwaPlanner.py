from typing import Tuple
import numpy as np
import math
# ============================================================================
# DWA PLANNER
# ============================================================================
class DWAPlanner:
    """Dynamic Window Approach planner with fuzzy logic integration"""

    def __init__(self, cfg, fuzzy):
        self.cfg = cfg
        self.fuzzy = fuzzy
        print("✓ DWA Planner initialized with fuzzy integration")

    def plan(self, x: float, y: float, theta: float,
             cur_v: float, cur_w: float,
             goal_x: float, goal_y: float,
             occupancy_grid, ranges: np.ndarray) -> Tuple[float, float]:
        """Plan linear and angular velocity commands"""
        dist_obs = np.min(ranges) if len(ranges) > 0 else self.cfg.max_lidar_range
        goal_angle = math.atan2(goal_y - y, goal_x - x) - theta
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi

        factors = self.fuzzy.infer(dist_obs, goal_angle, cur_v)
        speed_factor = factors['speed_factor']
        turn_factor = factors['turn_factor']

        alpha = turn_factor
        beta = 1.0 - speed_factor
        gamma = speed_factor

        total = alpha + beta + gamma
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total
        else:
            alpha, beta, gamma = 0.33, 0.33, 0.34

        v_min, v_max = self._compute_v_bounds(cur_v)
        w_min, w_max = self._compute_w_bounds(cur_w)

        v_samples = np.linspace(v_min, v_max, self.cfg.v_samples)
        w_samples = np.linspace(w_min, w_max, self.cfg.w_samples)

        best_v, best_w = 0.0, 0.0
        best_score = -np.inf

        for v in v_samples:
            for w in w_samples:
                traj = self._simulate_trajectory(x, y, theta, v, w)

                if not self._is_admissible(traj, occupancy_grid):
                    continue

                heading = self._eval_heading(traj, goal_x, goal_y)
                clearance = self._eval_clearance(traj, occupancy_grid)
                velocity = self._eval_velocity(v)

                score = alpha * heading + beta * clearance + gamma * velocity

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        if best_score == -np.inf:
            print("⚠️ No admissible trajectory - attempting recovery turn")
            return 0.0, self.cfg.max_angular_speed * 0.5

        return best_v, best_w

    def _compute_v_bounds(self, cur_v: float) -> Tuple[float, float]:
        v_min = max(0.0, cur_v - self.cfg.max_accel * self.cfg.dt)
        v_max = min(self.cfg.max_speed, cur_v + self.cfg.max_accel * self.cfg.dt)
        return v_min, v_max

    def _compute_w_bounds(self, cur_w: float) -> Tuple[float, float]:
        w_min = max(-self.cfg.max_angular_speed,
                    cur_w - self.cfg.max_angular_accel * self.cfg.dt)
        w_max = min(self.cfg.max_angular_speed,
                    cur_w + self.cfg.max_angular_accel * self.cfg.dt)
        return w_min, w_max

    def _simulate_trajectory(self, x: float, y: float, theta: float,
                             v: float, w: float) -> np.ndarray:
        t = np.linspace(0, self.cfg.prediction_horizon, self.cfg.traj_samples)
        if abs(w) < 1e-6:
            traj_x = x + v * t * np.cos(theta)
            traj_y = y + v * t * np.sin(theta)
            traj_theta = theta * np.ones_like(t)
        else:
            traj_x = x + (v / w) * (np.sin(theta + w * t) - np.sin(theta))
            traj_y = y + (v / w) * (-np.cos(theta + w * t) + np.cos(theta))
            traj_theta = theta + w * t
        return np.column_stack((traj_x, traj_y, traj_theta))

    def _is_admissible(self, traj: np.ndarray, occupancy_grid) -> bool:
        for point in traj[:, :2]:
            if occupancy_grid.is_occupied(point[0], point[1]):
                return False
        return True

    def _eval_heading(self, traj: np.ndarray, goal_x: float, goal_y: float) -> float:
        final_x, final_y, final_theta = traj[-1]
        goal_angle = math.atan2(goal_y - final_y, goal_x - final_x)
        delta = abs((goal_angle - final_theta + math.pi) % (2 * math.pi) - math.pi)
        return 1.0 - (delta / math.pi)

    def _eval_clearance(self, traj: np.ndarray, occupancy_grid) -> float:
        min_dist = np.inf
        for point in traj[:, :2]:
            dist = occupancy_grid.get_min_dist_to_obstacle(point[0], point[1])
            min_dist = min(min_dist, dist)
        return min(min_dist / self.cfg.max_lidar_range, 1.0)

    def _eval_velocity(self, v: float) -> float:
        return v / self.cfg.max_speed
