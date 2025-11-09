# ============================================================================
# MAIN NAVIGATION SYSTEM
# ============================================================================
import math
import matplotlib.pyplot as plt
from typing import Tuple
import time  # Changed from lgpio.time to standard time
from .configuration import RobotConfig
from .lidar import LidarInterface
from .motorController import MotorController
from .robotState import RobotState
from .occupancyGrid import OccupancyGrid
from .fuzzyLogic import FuzzyController
from .draw_map import initializeMap, drawMap
from .DWA import DWAPlanner
import numpy as np

class AutonomousRobot:
    """Main autonomous navigation system"""
    def __init__(self, cfg: RobotConfig, lidar: LidarInterface,
                 goal_x: float, goal_y: float):
        self.cfg = cfg
        self.lidar = lidar
        self.goal = (goal_x, goal_y)
        ax1, ax2 = initializeMap()
        self.ax1 = ax1
        self.ax2 = ax2
        # Initialize components
        self.motors = MotorController(cfg)
        self.state = RobotState(0.0, 0.0, 0.0)
        self.occupancy_grid = OccupancyGrid(cfg)
        self.fuzzy = FuzzyController(cfg)
        self.planner = DWAPlanner(cfg, self.fuzzy)
        self.previous_ranges = None
        self.previous_angles = None
        self._running = False
        print("✓ Autonomous robot initialized")
        print(f"  Goal: ({goal_x:.2f}, {goal_y:.2f})")

    def start(self):
        """Start the navigation loop"""
        print("\n🚀 Starting autonomous navigation...")
        self.lidar.start()
        self._running = True
        try:
            self._navigation_loop()
        except KeyboardInterrupt:
            print("\n⚠️  Navigation interrupted by user")
        finally:
            self.stop()

    def _navigation_loop(self):
        """Main navigation control loop"""
        # Get initial scan
        ranges, angles = self.lidar.get_scan()
        # Update initial map
        self.occupancy_grid.update_from_scan(
            self.state.x, self.state.y, self.state.theta, ranges, angles
        )
        # Initial draw
        self._draw_map()
        # Set previous
        self.previous_ranges = ranges
        self.previous_angles = angles

        step = 0

        while self._running and step < self.cfg.max_steps:
            step += 1

            # Plan motion
            v_cmd, w_cmd = self.planner.plan(
                self.state.x, self.state.y, self.state.theta,
                self.state.v, self.state.w,
                self.goal[0], self.goal[1],
                self.occupancy_grid, ranges
            )

            # Convert to wheel speeds (differential drive)
            left_speed, right_speed = self._velocities_to_wheels(v_cmd, w_cmd)

            # Send commands to motors
            self.motors.set_wheel_speeds(left_speed, right_speed)

            # Wait for motion
            time.sleep(self.cfg.dt)

            # Get new LIDAR scan
            ranges, angles = self.lidar.get_scan()

            # Estimate delta pose from scan matching
            delta_x, delta_y, delta_theta = self._estimate_delta_pose(
                self.previous_ranges, self.previous_angles,
                ranges, angles, v_cmd, w_cmd
            )

            # Update state with estimated delta (LIDAR odometry)
            old_theta = self.state.theta
            self.state.x += math.cos(old_theta) * delta_x - math.sin(old_theta) * delta_y
            self.state.y += math.sin(old_theta) * delta_x + math.cos(old_theta) * delta_y
            self.state.theta += delta_theta
            self.state.theta = (self.state.theta + math.pi) % (2 * math.pi) - math.pi  # Normalize

            # Update current v, w from estimation
            dist_moved = math.sqrt(delta_x**2 + delta_y**2)
            self.state.v = dist_moved / self.cfg.dt
            self.state.w = delta_theta / self.cfg.dt

            # Update previous scan
            self.previous_ranges = ranges
            self.previous_angles = angles

            # Update map
            self.occupancy_grid.update_from_scan(
                self.state.x, self.state.y, self.state.theta, ranges, angles
            )

            # Check goal reached
            goal_dist = math.sqrt((self.state.x - self.goal[0])**2 +
                                  (self.state.y - self.goal[1])**2)
            if goal_dist < self.cfg.goal_tolerance:
                print(f"\n🎯 Goal reached in {step} steps!")
                break

            # Draw
            self._draw_map()

            # Status logging
            if step % 10 == 0:
                min_obs = np.min(ranges) if len(ranges) > 0 else 5.0
                print(f"Step {step}: Pos=({self.state.x:.2f}, {self.state.y:.2f}), "
                      f"Goal={goal_dist:.2f}m, MinObs={min_obs:.2f}m, "
                      f"v={self.state.v:.2f}, w={self.state.w:.2f}")

        if step >= self.cfg.max_steps:
            print(f"\n⏱️  Max steps ({self.cfg.max_steps}) reached")

    def _draw_map(self):
        """Helper to draw maps"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        # Redraw settings
        self.ax1.set_xlim(-4, 4)
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True)
        self.ax1.set_title('LD19 LIDAR Raw Scan')

        self.ax2.set_xlim(-5, 5)
        self.ax2.set_ylim(-5, 5)
        self.ax2.set_aspect('equal')
        self.ax2.grid(True)
        self.ax2.set_title('Occupancy Grid Map')

        # Draw LIDAR scan (left)
        self.lidar.draw(self.ax1)
        self.ax1.plot(0, 0, 'bo', markersize=10, label='Robot')
        self.ax1.legend()
        # Draw occupancy grid (right)
        self.occupancy_grid.draw(self.ax2)
        self.ax2.plot(self.state.x, self.state.y, 'bo', markersize=10, label='Robot')
        self.ax2.legend()
        plt.pause(0.1)

    def _velocities_to_wheels(self, v: float, w: float) -> Tuple[float, float]:
        """Convert linear/angular velocity to differential wheel speeds"""
        # v = (v_left + v_right) / 2
        # w = (v_right - v_left) / wheel_base

        v_left = v - (w * self.cfg.wheel_base / 2)
        v_right = v + (w * self.cfg.wheel_base / 2)

        # Normalize to max speed and convert to -1 to 1 range
        max_wheel_speed = self.cfg.max_speed
        left_speed = np.clip(v_left / max_wheel_speed, -1.0, 1.0)
        right_speed = np.clip(v_right / max_wheel_speed, -1.0, 1.0)

        return left_speed, right_speed

    def _scan_to_points(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Convert polar LIDAR scan to cartesian points"""
        valid = (ranges > 0.01) & (ranges < self.cfg.max_lidar_range)
        x = ranges[valid] * np.cos(angles[valid])
        y = ranges[valid] * np.sin(angles[valid])
        return np.column_stack((x, y))

    def _estimate_delta_pose(self, prev_ranges: np.ndarray, prev_angles: np.ndarray,
                             curr_ranges: np.ndarray, curr_angles: np.ndarray,
                             v_cmd: float, w_cmd: float) -> Tuple[float, float, float]:
        """Estimate delta pose using ICP scan matching with commanded initial guess"""
        prev_points = self._scan_to_points(prev_ranges, prev_angles)
        curr_points = self._scan_to_points(curr_ranges, curr_angles)

        if len(prev_points) < 50 or len(curr_points) < 50:
            print("⚠️ Insufficient points for ICP - using commanded velocities")
            delta_theta = w_cmd * self.cfg.dt
            delta_x = v_cmd * self.cfg.dt
            delta_y = 0.0
            return delta_x, delta_y, delta_theta

        # Initial guess from commands
        delta_theta_init = w_cmd * self.cfg.dt
        delta_x_init = v_cmd * self.cfg.dt
        delta_y_init = 0.0
        R_init = np.array([
            [math.cos(delta_theta_init), -math.sin(delta_theta_init)],
            [math.sin(delta_theta_init), math.cos(delta_theta_init)]
        ])
        t_init = np.array([delta_x_init, delta_y_init])
        init_T = np.eye(3)
        init_T[:2, :2] = R_init
        init_T[:2, 2] = t_init

        # Perform ICP
        tx, ty, theta = self._icp(prev_points, curr_points, init_T)

        return tx, ty, theta

    def _icp(self, ref_points: np.ndarray, src_points: np.ndarray,
             initial_T: np.ndarray = np.eye(3), max_iter: int = 50,
             min_delta: float = 1e-6, max_cor_dist: float = 0.3) -> Tuple[float, float, float]:
        """2D ICP implementation using NumPy (aligns src to ref)"""
        # Add homogeneous coordinates
        N_src = len(src_points)
        src_hom = np.hstack((src_points, np.ones((N_src, 1))))
        ref_hom = np.hstack((ref_points, np.ones((len(ref_points), 1))))

        T = initial_T
        prev_err = np.inf

        for _ in range(max_iter):
            # Transform source points
            tf_src = src_hom @ T.T  # Nx3 @ 3x3

            # Brute-force nearest neighbors (for small N ~360)
            dist_matrix = np.sum((tf_src[:, :2][:, None, :] - ref_points[None, :, :]) ** 2, axis=-1)  # N_src x N_ref
            indices = np.argmin(dist_matrix, axis=1)
            dists = np.sqrt(dist_matrix[np.arange(N_src), indices])

            # Filter correspondences
            mask = dists < max_cor_dist
            if np.sum(mask) < 10:  # Minimum matches
                break

            matched_src = tf_src[mask, :2]
            matched_ref = ref_points[indices[mask]]

            # Compute centroids
            mu_src = np.mean(matched_src, axis=0)
            mu_ref = np.mean(matched_ref, axis=0)

            # Centered points
            src_cen = matched_src - mu_src
            ref_cen = matched_ref - mu_ref

            # Cross-covariance
            H = src_cen.T @ ref_cen  # 2x2

            # SVD
            u, s, vh = np.linalg.svd(H)

            # Rotation
            R = u @ vh
            if np.linalg.det(R) < 0:
                vh[-1, :] *= -1
                R = u @ vh

            # Translation
            t = mu_ref - R @ mu_src

            # New delta transformation
            new_T = np.eye(3)
            new_T[:2, :2] = R
            new_T[:2, 2] = t

            # Accumulate
            T = T @ new_T

            # Compute error
            err = np.mean(dists[mask] ** 2)
            if abs(prev_err - err) < min_delta:
                break
            prev_err = err

        # Extract delta_x, delta_y, delta_theta
        theta = math.atan2(T[1, 0], T[0, 0])
        tx = T[0, 2]
        ty = T[1, 2]

        return tx, ty, theta

    def stop(self):
        """Stop robot and cleanup"""
        print("\n🛑 Stopping robot...")
        self._running = False
        self.motors.stop()
        self.lidar.stop()
        self.motors.cleanup()
        print("✓ Cleanup complete")
