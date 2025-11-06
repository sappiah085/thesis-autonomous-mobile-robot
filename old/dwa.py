# dwa.py
import numpy as np
from config import MAX_SPEED, MAX_ANGULAR_SPEED, MAX_ACCEL, MAX_ANGULAR_ACCEL, DT, GOAL_TOLERANCE, OBSTACLE_THRESHOLD, GOAL_WEIGHT, OBSTACLE_WEIGHT

class DWA:
    def __init__(self):
        self.v = 0.0  # Linear velocity
        self.w = 0.0  # Angular velocity

    def plan(self, robot_pose, goal, lidar_ranges, lidar_angles):
        """Compute optimal velocity commands using DWA."""
        x, y, theta = robot_pose
        gx, gy = goal

        # Dynamic window: possible velocities based on acceleration limits
        v_min = max(self.v - MAX_ACCEL * DT, 0)
        v_max = min(self.v + MAX_ACCEL * DT, MAX_SPEED)
        w_min = max(self.w - MAX_ANGULAR_ACCEL * DT, -MAX_ANGULAR_SPEED)
        w_max = min(self.w + MAX_ANGULAR_ACCEL * DT, MAX_ANGULAR_SPEED)

        best_score = -float('inf')
        best_v, best_w = self.v, self.w

        # Evaluate velocity combinations
        for v in np.linspace(v_min, v_max, 10):
            for w in np.linspace(w_min, w_max, 10):
                score = self._evaluate_trajectory(v, w, robot_pose, goal, lidar_ranges, lidar_angles)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        # Stuck detection: if low velocities, add random turn to escape local minimum
        if best_v < 0.1 and abs(best_w) < 0.1:
            best_w = np.random.uniform(-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        self.v, self.w = best_v, best_w
        return self.v, self.w

    def _evaluate_trajectory(self, v, w, robot_pose, goal, lidar_ranges, lidar_angles):
        """Evaluate a trajectory based on goal distance, heading, and obstacle distance."""
        x, y, theta = robot_pose
        gx, gy = goal

        # Predict next position
        x_new = x + v * np.cos(theta) * DT
        y_new = y + v * np.sin(theta) * DT
        theta_new = theta + w * DT

        # Goal cost
        goal_dist = np.sqrt((x_new - gx)**2 + (y_new - gy)**2)
        goal_cost = -GOAL_WEIGHT * goal_dist

        # Heading cost
        heading = np.arctan2(gy - y_new, gx - x_new)
        heading_error = abs(self._normalize_angle(heading - theta_new))
        heading_cost = -heading_error

        # Obstacle cost
        obstacle_dist = min(lidar_ranges) if lidar_ranges.size > 0 else float('inf')
        obstacle_cost = OBSTACLE_WEIGHT * (1.0 / obstacle_dist) if obstacle_dist < OBSTACLE_THRESHOLD else 0

        return goal_cost + heading_cost - obstacle_cost

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
