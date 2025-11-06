# ============================================================================
# MAIN NAVIGATION SYSTEM
# ============================================================================
import math
import matplotlib.pyplot as plt
from typing import Tuple
from lgpio import time
from .configuration import RobotConfig
from .lidar import LidarInterface
from .motorController import MotorController
from .robotState import RobotState
from .occupancyGrid import OccupancyGrid
from .fuzzyLogic import FuzzyController
from .dwa import DWAPlanner
from .draw_map import initializeMap, drawMap
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
        step = 0

        while self._running and step < self.cfg.max_steps:
            step += 1
            loop_start = time.time()

            # Check goal reached
            goal_dist = math.sqrt((self.state.x - self.goal[0])**2 +
                                 (self.state.y - self.goal[1])**2)
            if goal_dist < self.cfg.goal_tolerance:
                print(f"\n🎯 Goal reached in {step} steps!")
                break

            # Get LIDAR scan
            ranges, angles = self.lidar.get_scan()
            # Update map
            self.occupancy_grid.update_from_scan(
                self.state.x, self.state.y, self.state.theta, ranges, angles
            )

            # Plan motion
            v_cmd, w_cmd = self.planner.plan(
                self.state.x, self.state.y, self.state.theta,
                self.state.v, self.state.w,
                self.goal[0], self.goal[1],
                self.occupancy_grid, ranges
            )

            # Convert to wheel speeds (differential drive)
            # left_speed, right_speed = self._velocities_to_wheels(v_cmd, w_cmd)
            self.motors.set_motor_speed(v_cmd, w_cmd)

            # Send commands to motors
            # self.motors.set_wheel_speeds(left_speed, right_speed)

            # Update state (odometry)
            self.state.update_odometry(v_cmd, w_cmd, self.cfg.dt)
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
            # Status logging
            if step % 10 == 0:
                min_obs = np.min(ranges) if len(ranges) > 0 else 5.0
                print(f"Step {step}: Pos=({self.state.x:.2f}, {self.state.y:.2f}), "
                      f"Goal={goal_dist:.2f}m, MinObs={min_obs:.2f}m, "
                      f"v={v_cmd:.2f}, w={w_cmd:.2f}")

            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < self.cfg.dt:
                time.sleep(self.cfg.dt - elapsed)

        if step >= self.cfg.max_steps:
            print(f"\n⏱️  Max steps ({self.cfg.max_steps}) reached")


    def _velocities_to_wheels(self, v: float, w: float) -> Tuple[float, float]:
        print(w)
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

    def stop(self):
        """Stop robot and cleanup"""
        print("\n🛑 Stopping robot...")
        self._running = False
        self.motors.stop()
        self.lidar.stop()
        self.motors.cleanup()
        print("✓ Cleanup complete")
