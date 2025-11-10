# ============================================================================
# DYNAMIC WINDOW APPROACH PLANNER WITH FUZZY LOGIC INTEGRATION
# ============================================================================
# Enhanced version with proper safety margins to prevent collisions
#
# Key Safety Improvements:
# - Robot footprint consideration (circular approximation)
# - Safety margin buffer around obstacles
# - Minimum clearance threshold for trajectory rejection
# - Conservative collision checking along entire trajectory
#
# Based on research from:
# - "Local Path Planning for Mobile Robots Based on Fuzzy Dynamic Window Algorithm"
# - "Autonomous Robot Navigation Using Fuzzy Inference Based Dynamic Window Approach"

import numpy as np
import math
from typing import Tuple

class DWAPlanner:
    """Dynamic Window Approach planner with fuzzy logic integration and safety margins"""

    def __init__(self, cfg, fuzzy):
        self.cfg = cfg
        self.fuzzy = fuzzy

        # Safety parameters (should be added to cfg in practice)
        # Robot radius - conservative estimate of robot's circular footprint
        self.robot_radius = getattr(cfg, 'robot_radius', 1)  # meters

        # Safety margin - additional buffer beyond robot radius
        self.safety_margin = getattr(cfg, 'safety_margin', 0.4)  # meters

        # Total clearance needed = robot radius + safety margin
        self.min_clearance = self.robot_radius + self.safety_margin

        # Number of points to check along trajectory for more thorough collision detection
        self.collision_check_resolution = getattr(cfg, 'collision_check_resolution', 0.05)  # meters

        print(f"✓ DWA Planner initialized with fuzzy integration")
        print(f"  - Robot radius: {self.robot_radius:.2f}m")
        print(f"  - Safety margin: {self.safety_margin:.2f}m")
        print(f"  - Minimum clearance: {self.min_clearance:.2f}m")

    def plan(self, x: float, y: float, theta: float,
             cur_v: float, cur_w: float,
             goal_x: float, goal_y: float,
             occupancy_grid, ranges: np.ndarray) -> Tuple[float, float]:
        """
        Plan linear and angular velocity commands with safety considerations.

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

                # Check if admissible (collision-free with safety margins)
                is_safe, min_traj_clearance = self._is_admissible(traj, occupancy_grid)
                if not is_safe:
                    continue

                # Compute evaluation scores
                heading = self._eval_heading(traj, goal_x, goal_y)
                clearance = self._eval_clearance(min_traj_clearance)
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

    def _is_admissible(self, traj: np.ndarray, occupancy_grid) -> Tuple[bool, float]:
        """
        Check if trajectory is collision-free with safety margins.

        Returns: (is_safe, min_clearance)
            is_safe: True if trajectory maintains minimum clearance from all obstacles
            min_clearance: Minimum distance to obstacles along the trajectory
        """
        min_clearance = np.inf

        # Check each point along trajectory
        for i in range(len(traj) - 1):
            # Get current and next point
            p1 = traj[i, :2]
            p2 = traj[i + 1, :2]

            # Calculate segment length
            segment_length = np.linalg.norm(p2 - p1)

            # Number of intermediate points to check
            num_checks = max(2, int(segment_length / self.collision_check_resolution))

            # Check points along the segment
            for j in range(num_checks):
                t = j / (num_checks - 1) if num_checks > 1 else 0
                check_point = p1 + t * (p2 - p1)

                # Get distance to nearest obstacle
                dist = occupancy_grid.get_min_dist_to_obstacle(check_point[0], check_point[1])
                min_clearance = min(min_clearance, dist)

                # Immediate rejection if too close to obstacle
                if dist < self.min_clearance:
                    return False, dist

        return True, min_clearance

    def _eval_heading(self, traj: np.ndarray, goal_x: float, goal_y: float) -> float:
        """Evaluate alignment to goal (normalized 0-1)"""
        final_x, final_y, final_theta = traj[-1]
        goal_angle = math.atan2(goal_y - final_y, goal_x - final_x)
        delta = abs((goal_angle - final_theta + math.pi) % (2 * math.pi) - math.pi)
        return 1.0 - (delta / math.pi)  # 1 if aligned, 0 if opposite

    def _eval_clearance(self, min_clearance: float) -> float:
        """
        Evaluate obstacle clearance (normalized 0-1).

        Uses the minimum clearance found during admissibility check.
        Rewards trajectories that maintain good distance from obstacles.
        """
        # Normalize clearance score
        # Full score if clearance >= max sensor range
        # Score increases non-linearly with clearance for better safety
        normalized = min(min_clearance / self.cfg.max_lidar_range, 1.0)

        # Apply non-linear scaling to further reward safer trajectories
        # This makes the planner prefer paths with more clearance
        return normalized ** 0.5  # Square root gives more weight to additional clearance

    def _eval_velocity(self, v: float) -> float:
        """Evaluate forward progress (normalized)"""
        return v / self.cfg.max_speed
