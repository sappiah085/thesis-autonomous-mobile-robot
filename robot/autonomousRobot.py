from .motorController import MotorController
from .robotState import RobotState
from .occupancy import OccupancyGrid
from .fuzzy import FuzzyController
from .dwaPlanner import DWAPlanner
from .ICPScanMatcher import ICPScanMatcher
from .config import RobotConfig
from .plot import initialize_plot
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple
# ============================================================================
# MAIN NAVIGATION SYSTEM
# ============================================================================
class AutonomousRobot:
    """Complete autonomous navigation system with ICP scan matching"""
    def __init__(self, cfg: RobotConfig, lidar, goal_x: float, goal_y: float):
        self.cfg = cfg
        self.lidar = lidar
        self.goal = (goal_x, goal_y)

        # Initialize visualization
        self.fig, self.ax1, self.ax2 = initialize_plot()

        # Initialize components
        self.motors = MotorController(cfg)
        self.state = RobotState()
        self.grid = OccupancyGrid(cfg)
        self.fuzzy = FuzzyController(cfg)
        self.planner = DWAPlanner(cfg, self.fuzzy)
        self.scan_matcher = ICPScanMatcher(cfg)

        self._running = False

        print("✓ Autonomous robot initialized with ICP scan matching")
        print(f"  Goal: ({goal_x:.2f}, {goal_y:.2f})")
        print("  Position tracking: ICP scan matching (LIDAR-only odometry)")

    def start(self):
        """Start autonomous navigation"""
        print("\n🚀 Starting autonomous navigation with scan matching...")
        self.lidar.start()
        self._running = True

        time.sleep(0.5)

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

            # Get LIDAR scan
            ranges, angles = self.lidar.get_scan()

            # Estimate odometry using ICP scan matching
            dx, dy, dtheta = self.scan_matcher.estimate_odometry(
                ranges, angles,
                self.state.v, self.state.w
            )

            # Update robot pose
            new_x, new_y, new_theta = self.scan_matcher.update_pose(dx, dy, dtheta)
            self.state.update(new_x, new_y, new_theta)

            # Update occupancy grid with new pose
            self.grid.update_from_scan(
                self.state.x, self.state.y, self.state.theta,
                ranges, angles
            )

            # Plan motion using DWA + Fuzzy Logic
            v_cmd, w_cmd = self.planner.plan(
                self.state.x, self.state.y, self.state.theta,
                self.state.v, self.state.w,
                self.goal[0], self.goal[1],
                self.grid, ranges
            )
            print(v_cmd, np.degrees(w_cmd))
            if np.degrees(w_cmd) > 15:  # Turning in place - use your open-loop method
                turn_angle_deg = np.degrees(w_cmd * self.cfg.dt)  # Convert to degrees for one step
                self.motors.recovery_turn(turn_angle_deg)
                # Skip regular motor commands
            else:
                # Normal case: Convert to wheel speeds
                left_speed, right_speed = self._velocities_to_wheels(v_cmd, w_cmd)
                # Send commands to motors
                self.motors.set_motor_speed(max(left_speed, right_speed)* self.cfg.pwm_frequency)

            # Update velocity state
            self.state.v = v_cmd
            self.state.w = w_cmd

            # Check goal reached
            goal_dist = np.sqrt((self.state.x - self.goal[0])**2 +
                               (self.state.y - self.goal[1])**2)

            if goal_dist < self.cfg.goal_tolerance:
                print(f"\n🎯 Goal reached in {step} steps!")
                self.motors.stop()
                break

            # # Visualization
            if step % 1 == 0:
                self._draw_map(ranges, angles)

            # Status logging
            if step % 10 == 0:
                min_obs = np.min(ranges) if len(ranges) > 0 else 5.0
                print(f"Step {step:4d}: Pos=({self.state.x:5.2f}, {self.state.y:5.2f}, "
                      f"{np.degrees(self.state.theta):6.1f}°) "
                      f"Odom=(Δx={dx:+.3f}, Δy={dy:+.3f}, Δθ={np.degrees(dtheta):+.1f}°) "
                      f"Goal={goal_dist:4.2f}m, MinObs={min_obs:4.2f}m")

            # Maintain loop timing
            elapsed = time.time() - loop_start
            if elapsed < self.cfg.dt:
                time.sleep(self.cfg.dt - elapsed)

        if step >= self.cfg.max_steps:
            print(f"\n⏱️  Max steps ({self.cfg.max_steps}) reached")

    def _draw_map(self, ranges: np.ndarray, angles: np.ndarray):
        """Update visualization"""
        self.ax1.clear()
        self.ax2.clear()

        # Left plot: Raw LIDAR scan
        self.ax1.set_xlim(-4, 4)
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('LD19 LIDAR Raw Scan (Robot Frame)')

        # Use LD19's built-in draw method
        self.lidar.draw(self.ax1)
        self.ax1.plot(0, 0, 'bo', markersize=12, label='Robot')
        self.ax1.legend()

        # Right plot: Occupancy grid
        self.ax2.set_xlim(-5, 5)
        self.ax2.set_ylim(-5, 5)
        self.ax2.set_aspect('equal')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Occupancy Grid Map (World Frame)')

        # Draw grid
        self.grid.draw(self.ax2)

        # Draw robot (always at origin without odometry)
        self.ax2.plot(self.state.x, self.state.y, 'bo', markersize=12, label='Robot')

        # Draw robot heading
        arrow_len = 0.4
        dx = arrow_len * np.cos(self.state.theta)
        dy = arrow_len * np.sin(self.state.theta)
        self.ax2.arrow(self.state.x, self.state.y, dx, dy,
                      head_width=0.2, head_length=0.15, fc='blue', ec='blue')

        # Draw goal
        self.ax2.plot(self.goal[0], self.goal[1], 'g*',
                     markersize=20, label='Goal')

        # Draw trajectory history
        if len(self.state.trajectory_x) > 1:
            self.ax2.plot(self.state.trajectory_x, self.state.trajectory_y,
                         'b-', alpha=0.3, linewidth=1, label='Path')

        self.ax2.legend()

        plt.pause(0.01)

    def _velocities_to_wheels(self, v: float, w: float) -> Tuple[float, float]:
        """Convert linear/angular velocity to differential wheel speeds"""
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
