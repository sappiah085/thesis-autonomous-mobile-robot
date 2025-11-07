# ============================================================================
# DYNAMIC WINDOW APPROACH PLANNER WITH FUZZY LOGIC INTEGRATION
# ============================================================================
# Based on research from:
# - "Local Path Planning for Mobile Robots Based on Fuzzy Dynamic Window Algorithm"
#   (https://pmc.ncbi.nlm.nih.gov/articles/PMC10575201/)
# - "Autonomous Robot Navigation Using Fuzzy Inference Based Dynamic Window Approach"
#   (https://www.researchgate.net/publication/367068862_Autonomous_Robot_Navigation_Using_Fuzzy_Inference_Based_Dynamic_Window_Approach)
#
# These papers integrate fuzzy logic to dynamically adjust the weights (alpha, beta, gamma)
# of the DWA evaluation function based on environmental factors like goal distance/orientation
# and nearest obstacle distance.
#
# Adaptation:
# - Use the provided FuzzyController's infer method (which uses min_obstacle_dist and goal_angle).
# - Map the outputs: alpha = turn_factor, gamma = speed_factor, beta = 1 - speed_factor
# - Normalize the weights to sum to 1 for relative importance.
# - This approximates the fuzzy adjustment: high beta when close to obstacles (low speed_factor),
#   high alpha when needing to align (high turn_factor), high gamma when clear (high speed_factor).
#
# Key Features:
# - Samples velocities within dynamic constraints.
# - Simulates trajectories and checks for collisions using occupancy grid.
# - Evaluates trajectories with adaptive weights from fuzzy controller.
# - Selects optimal admissible velocity command.
#
# Assumptions:
# - FuzzyController.infer(min_obstacle_dist, goal_angle, velocity) returns dict with 'speed_factor', 'turn_factor'.
# - Velocity input to infer is current v (though not used in provided implementation).
# - RobotState provides current hardware position/orientation (odometry updated externally from hardware, not simulated here).

import numpy as np
import math
from typing import Tuple

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
        """
        Plan linear and angular velocity commands.

        Parameters follow the commented signature in AutonomousRobot.
        Uses occupancy grid for collision checking and lidar ranges for min_obs.

        Returns: (v_cmd, w_cmd)
        """
        # Compute inputs for fuzzy
        dist_obs = np.min(ranges) if len(ranges) > 0 else self.cfg.max_lidar_range
        goal_angle = math.atan2(goal_y - y, goal_x - x) - theta
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

        # Get factors from fuzzy controller
        factors = self.fuzzy.infer(dist_obs, goal_angle, cur_v)
        speed_factor = factors['speed_factor']
        turn_factor = factors['turn_factor']

        # Map to adaptive weights (based on research adaptations)
        alpha = turn_factor  # Prioritize heading when turn_factor high
        beta = 1.0 - speed_factor  # Prioritize clearance when speed_factor low (close obstacles)
        gamma = speed_factor  # Prioritize velocity when speed_factor high

        # Normalize weights
        total = alpha + beta + gamma
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total
        else:
            alpha, beta, gamma = 0.33, 0.33, 0.34  # Fallback

        # Compute dynamic window bounds
        v_min, v_max = self._compute_v_bounds(cur_v)
        w_min, w_max = self._compute_w_bounds(cur_w)

        # Sample velocities (discretize the window)
        v_samples = np.linspace(v_min, v_max, self.cfg.v_samples)
        w_samples = np.linspace(w_min, w_max, self.cfg.w_samples)

        best_v, best_w = 0.0, 0.0
        best_score = -np.inf

        for v in v_samples:
            for w in w_samples:
                # Simulate trajectory
                traj = self._simulate_trajectory(x, y, theta, v, w)

                # Check if admissible (collision-free)
                if not self._is_admissible(traj, occupancy_grid):
                    continue

                # Compute evaluation scores
                heading = self._eval_heading(traj, goal_x, goal_y)
                clearance = self._eval_clearance(traj, occupancy_grid)
                velocity = self._eval_velocity(v)

                # Compute score with adaptive fuzzy weights
                score = alpha * heading + beta * clearance + gamma * velocity

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        # If no admissible trajectory, stop
        if best_score == -np.inf:
            print("⚠️ No admissible trajectory found - stopping")
            return 0.0, 0.0

        return best_v, best_w

    def _compute_v_bounds(self, cur_v: float) -> Tuple[float, float]:
        """Compute feasible linear velocity bounds"""
        v_min = max(0.0, cur_v - self.cfg.max_accel * self.cfg.dt)  # Assuming non-backward
        v_max = min(self.cfg.max_speed, cur_v + self.cfg.max_accel * self.cfg.dt)
        return v_min, v_max

    def _compute_w_bounds(self, cur_w: float) -> Tuple[float, float]:
        """Compute feasible angular velocity bounds"""
        w_min = max(-self.cfg.max_angular_speed, cur_w - self.cfg.max_angular_accel * self.cfg.dt)
        w_max = min(self.cfg.max_angular_speed, cur_w + self.cfg.max_angular_accel * self.cfg.dt)
        return w_min, w_max

    def _simulate_trajectory(self, x: float, y: float, theta: float,
                             v: float, w: float) -> np.ndarray:
        """Simulate circular arc trajectory over prediction horizon"""
        t = np.linspace(0, self.cfg.prediction_horizon, self.cfg.traj_samples)
        if abs(w) < 1e-6:  # Straight line
            traj_x = x + v * t * np.cos(theta)
            traj_y = y + v * t * np.sin(theta)
            traj_theta = theta * np.ones_like(t)
        else:  # Arc
            traj_x = x - (v / w) * (np.sin(theta + w * t) - np.sin(theta))
            traj_y = y + (v / w) * (np.cos(theta + w * t) - np.cos(theta))
            traj_theta = theta + w * t
        return np.column_stack((traj_x, traj_y, traj_theta))

    def _is_admissible(self, traj: np.ndarray, occupancy_grid) -> bool:
        """Check if trajectory collides with occupied cells"""
        for point in traj[:, :2]:
            if occupancy_grid.is_occupied(point[0], point[1]):
                return False
        return True

    def _eval_heading(self, traj: np.ndarray, goal_x: float, goal_y: float) -> float:
        """Evaluate alignment to goal (normalized 0-1)"""
        final_x, final_y, final_theta = traj[-1]
        goal_angle = math.atan2(goal_y - final_y, goal_x - final_x)
        delta = abs((goal_angle - final_theta + math.pi) % (2 * math.pi) - math.pi)
        return 1.0 - (delta / math.pi)  # 1 if aligned, 0 if opposite

    def _eval_clearance(self, traj: np.ndarray, occupancy_grid) -> float:
        """Evaluate min distance to obstacles along trajectory (normalized)"""
        min_dist = np.inf
        for point in traj[:, :2]:
            dist = occupancy_grid.get_min_dist_to_obstacle(point[0], point[1])
            min_dist = min(min_dist, dist)
        return min(min_dist / self.cfg.max_lidar_range, 1.0)

    def _eval_velocity(self, v: float) -> float:
        """Evaluate forward progress (normalized)"""
        return v / self.cfg.max_speed
