import numpy as np
import math
from typing import Tuple

class DWAPlanner:
    """Dynamic Window Approach planner with fuzzy logic integration and trajectory-specific safety"""

    def __init__(self, cfg, fuzzy):
        self.cfg = cfg
        self.fuzzy = fuzzy

        # Safety parameters (should be added to cfg in practice)
        # Robot radius - conservative estimate of robot's circular footprint
        self.robot_radius = getattr(cfg, 'robot_radius', 0.15)  # meters

        # Safety margin - additional buffer beyond robot radius
        self.safety_margin = getattr(cfg, 'safety_margin', 0.05)  # meters

        # Total clearance needed = robot radius + safety margin
        self.min_clearance = self.robot_radius + self.safety_margin

        # Store original clearance for adaptive behavior
        self.original_clearance = self.min_clearance

        # Adaptive safety - reduce clearance if consistently stuck
        self.enable_adaptive_clearance = getattr(cfg, 'enable_adaptive_clearance', True)
        self.stuck_counter = 0
        self.max_stuck_count = 5  # Reduce clearance after this many failures

        print(f"✓ DWA Planner initialized with fuzzy integration")
        print(f"  - Robot radius: {self.robot_radius:.2f}m")
        print(f"  - Safety margin: {self.safety_margin:.2f}m")
        print(f"  - Minimum clearance: {self.min_clearance:.2f}m")
        print(f"  - Adaptive clearance: {'enabled' if self.enable_adaptive_clearance else 'disabled'}")

    def plan(self, x: float, y: float, theta: float,
             cur_v: float, cur_w: float,
             goal_x: float, goal_y: float,
             occupancy_grid, ranges: np.ndarray) -> Tuple[float, float]:
        """
        Plan linear and angular velocity commands with trajectory-specific safety.

        Key principle: Only reject trajectories that actually pass near obstacles,
        not based on global minimum obstacle distance.

        Parameters:
            x, y: Current robot position (meters)
            theta: Current robot heading (radians)
            cur_v: Current linear velocity (m/s)
            cur_w: Current angular velocity (rad/s)
            goal_x, goal_y: Goal position (meters)
            occupancy_grid: OccupancyGrid object with is_occupied() and get_min_dist_to_obstacle()
            ranges: LIDAR range measurements (used for fuzzy inference only)

        Returns: (v_cmd, w_cmd)
            v_cmd: Commanded linear velocity (m/s)
            w_cmd: Commanded angular velocity (rad/s)
        """
        # Compute inputs for fuzzy (uses global min for fuzzy inference only)
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

        # Debug counters
        total_trajectories = 0
        rejected_trajectories = 0
        rejection_reasons = {'too_close': 0, 'collision': 0}

        for v in v_samples:
            for w in w_samples:
                total_trajectories += 1

                # Simulate trajectory
                traj = self._simulate_trajectory(x, y, theta, v, w)

                # CRITICAL FIX: Check safety along THIS specific trajectory
                # Not global obstacle distance!
                is_safe, min_traj_clearance, reason = self._is_admissible(
                    traj, occupancy_grid
                )

                if not is_safe:
                    rejected_trajectories += 1
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
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

        # If no admissible trajectory, provide diagnostic info
        if best_score == -np.inf:
            self.stuck_counter += 1

            print(f"⚠️ No admissible trajectory found! (attempt {self.stuck_counter})")
            print(f"   Total trajectories tested: {total_trajectories}")
            print(f"   Rejected: {rejected_trajectories} (collision={rejection_reasons.get('collision', 0)}, too_close={rejection_reasons.get('too_close', 0)})")
            print(f"   Min clearance needed: {self.min_clearance:.3f}m")
            print(f"   Global min obstacle dist: {dist_obs:.3f}m (for fuzzy inference only)")
            print(f"   Velocity bounds: v=[{v_min:.2f}, {v_max:.2f}], w=[{w_min:.2f}, {w_max:.2f}]")
            print(f"   Fuzzy weights: α={alpha:.2f} (heading), β={beta:.2f} (clearance), γ={gamma:.2f} (velocity)")

            # Adaptive response: temporarily reduce safety margin if stuck repeatedly
            if self.enable_adaptive_clearance and self.stuck_counter >= self.max_stuck_count:
                original = self.min_clearance
                self.min_clearance = max(self.robot_radius, self.min_clearance * 0.7)
                print(f"   🔧 Reducing min_clearance from {original:.3f}m to {self.min_clearance:.3f}m")
                self.stuck_counter = 0  # Reset counter after adjustment

            return 0.0, 0.0

        # Reset stuck counter on success
        if self.stuck_counter > 0:
            self.stuck_counter = 0
            # Restore original clearance
            self.min_clearance = self.original_clearance

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
        t = np.linspace(self.cfg.dt, self.cfg.prediction_horizon, self.cfg.traj_samples)
        if abs(w) < 1e-6:  # Straight line
            traj_x = x + v * t * np.cos(theta)
            traj_y = y + v * t * np.sin(theta)
            traj_theta = theta * np.ones_like(t)
        else:  # Arc
            traj_x = x - (v / w) * (np.sin(theta + w * t) - np.sin(theta))
            traj_y = y + (v / w) * (np.cos(theta + w * t) - np.cos(theta))
            traj_theta = theta + w * t
        return np.column_stack((traj_x, traj_y, traj_theta))

    def _is_admissible(self, traj: np.ndarray, occupancy_grid) -> Tuple[bool, float, str]:
        """
        Check if trajectory is collision-free with safety margins.

        CRITICAL: Only checks obstacles along THIS trajectory path,
        not global minimum distance to all obstacles.

        Returns: (is_safe, min_clearance, rejection_reason)
            is_safe: True if trajectory maintains minimum clearance
            min_clearance: Minimum distance to obstacles along this trajectory
            rejection_reason: 'collision', 'too_close', or 'safe'
        """
        min_clearance = np.inf

        # Check each point along THIS specific trajectory
        for i in range(len(traj)):
            point = traj[i, :2]

            # Get distance to nearest obstacle from THIS trajectory point
            dist = occupancy_grid.get_min_dist_to_obstacle(point[0], point[1])

            # Handle invalid distance (e.g., point outside map bounds)
            if dist is None or np.isnan(dist) or np.isinf(dist):
                return False, min_clearance, 'invalid_point'

            min_clearance = min(min_clearance, dist)

            # Check if this trajectory point violates safety clearance
            if dist < self.min_clearance:
                return False, dist, 'too_close'

            # Also check for direct collision (occupied cell)
            if occupancy_grid.is_occupied(point[0], point[1]):
                return False, 0.0, 'collision'

        # If min_clearance is still inf, no obstacles were found - assume safe
        if np.isinf(min_clearance):
            min_clearance = self.cfg.max_lidar_range

        return True, min_clearance, 'safe'

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
