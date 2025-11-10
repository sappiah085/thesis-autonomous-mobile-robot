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
import math
import matplotlib.pyplot as plt
from typing import Tuple
from .debugger import CoordinateDebugger
from .caliberationLidar import LidarCalibrator
from .alignment import check_physical_alignment
# ============================================================================
# MAIN NAVIGATION SYSTEM
# ============================================================================
class AutonomousRobot:
    """Complete autonomous navigation system with ICP scan matching"""
    def __init__(self, cfg: RobotConfig, lidar, goal_x: float, goal_y: float,
                 initial_heading: float = 0.0):
        """
        Initialize autonomous robot

        Args:
            cfg: Robot configuration
            lidar: LIDAR sensor object
            goal_x: Goal x coordinate
            goal_y: Goal y coordinate
            initial_heading: Initial robot heading in radians (default: 0 = pointing along +X axis)
                           Set this to match your robot's actual physical orientation
        """
        self.cfg = cfg
        self.lidar = lidar
        self.goal = (goal_x, goal_y)

        # LIDAR coordinate frame correction
        # The LIDAR applies 0.475π offset to align its 0° with robot front
        self.lidar_front_offset = 0.475 * np.pi  # 85.5 degrees

        # Set initial robot heading in world frame
        self.initial_heading = initial_heading

        # Initialize visualization
        self.fig, self.ax1, self.ax2 = initialize_plot()

        # Initialize components
        self.motors = MotorController(cfg)
        self.state = RobotState()

        # Set initial pose with correct heading
        self.state.x = 0.0
        self.state.y = 0.0
        self.state.theta = initial_heading  # Set the initial heading

        self.grid = OccupancyGrid(cfg)
        self.fuzzy = FuzzyController(cfg)
        self.planner = DWAPlanner(cfg, self.fuzzy)

        # Initialize scan matcher with correct heading
        self.scan_matcher = ICPScanMatcher(cfg, initial_heading=initial_heading)

        self._running = False

        print("✓ Autonomous robot initialized with ICP scan matching")
        print(f"  Goal: ({goal_x:.2f}, {goal_y:.2f})")
        print(f"  Initial heading: {np.degrees(initial_heading):.1f}°")
        print(f"  LIDAR frame offset: {np.degrees(self.lidar_front_offset):.1f}°")
        print("  Position tracking: ICP scan matching (LIDAR-only odometry)")

    def start(self):

        """Start autonomous navigation"""
        print("\n🚀 Starting autonomous navigation with scan matching...")
        self.lidar.start()
        self._running = True
        time.sleep(0.5)

        try:
            # debugger = CoordinateDebugger(self.lidar, self.motors, self.cfg)
            # debugger.test_coordinate_system()
            # calibrator = LidarCalibrator(self.lidar)
            # offset_radians = calibrator.calibrate()
            # print(offset_radians)
            # measurements = check_physical_alignment(self.lidar)
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

            # Update robot pose (scan matcher maintains absolute world frame pose)
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

            if abs(np.degrees(w_cmd)) > 10:  # Turning in place - use your open-loop method
                turn_angle_deg = np.degrees(w_cmd)  # Convert to degrees for one step
                self.motors.recovery_turn(turn_angle_deg)
                # Skip regular motor commands
            else:
                # Normal case: Convert to wheel speeds
                left_speed, right_speed = self._velocities_to_wheels(v_cmd, w_cmd)
                # Send commands to motors
                self.motors.set_motor_speed(max(left_speed, right_speed) * self.cfg.pwm_frequency)

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

            # Visualization
            if step % 1 == 0:
                self._draw_map(ranges, angles)

            # Status logging
            if step % 10 == 0:
                min_obs = np.min(ranges) if len(ranges) > 0 else 5.0
                # Normalize theta to -180 to 180 for display
                theta_display = np.degrees(self.state.theta)
                theta_display = ((theta_display + 180) % 360) - 180
                print(f"Step {step:4d}: Pos=({self.state.x:5.2f}, {self.state.y:5.2f}, "
                      f"{theta_display:6.1f}°) "
                      f"Odom=(Δx={dx:+.3f}, Δy={dy:+.3f}, Δθ={np.degrees(dtheta):+.1f}°) "
                      f"Goal={goal_dist:4.2f}m, MinObs={min_obs:4.2f}m")

            # Maintain loop timing
            elapsed = time.time() - loop_start
            if elapsed < self.cfg.dt:
                time.sleep(self.cfg.dt - elapsed)

        if step >= self.cfg.max_steps:
            print(f"\n⏱️  Max steps ({self.cfg.max_steps}) reached")

    def _draw_map(self, ranges: np.ndarray, angles: np.ndarray):
      """
    Update visualization with corrected robot heading indicator

    FIXES:
    1. Map stays fixed in world frame (doesn't rotate with robot)
    2. Robot heading arrow correctly shows orientation at all angles including 180°
      """
      self.ax1.clear()
      self.ax2.clear()

    # ===================================================================
    # LEFT PLOT: Raw LIDAR scan in robot frame
    # ===================================================================
      self.ax1.set_xlim(-4, 4)
      self.ax1.set_ylim(-4, 4)
      self.ax1.set_aspect('equal')
      self.ax1.grid(True, alpha=0.3)
      self.ax1.set_title('LD19 LIDAR Scan (Robot Frame)')

    # Draw LIDAR scan points
      if len(ranges) > 0:
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)
        self.ax1.scatter(x, y, c='red', s=1, alpha=0.5, label='LIDAR')

    # Draw robot at center
      self.ax1.plot(0, 0, 'bo', markersize=12, label='Robot')

    # Draw robot front direction (in LIDAR frame, front is at angle=0)
      arrow_len = 0.8
      dx_front = arrow_len * np.sin(0)  # angle=0 in LIDAR frame
      dy_front = arrow_len * np.cos(0)
      self.ax1.arrow(0, 0, dx_front, dy_front,
                  head_width=0.25, head_length=0.2,
                  fc='green', ec='green', linewidth=2,
                  label='Front')

      self.ax1.legend(loc='upper right')

    # ===================================================================
    # RIGHT PLOT: Occupancy grid in FIXED world frame
    # ===================================================================
      self.ax2.set_xlim(-5, 5)
      self.ax2.set_ylim(-5, 5)
      self.ax2.set_aspect('equal')
      self.ax2.grid(True, alpha=0.3)

    # Normalize theta for display (-180° to +180°)
      theta_display = np.degrees(self.state.theta)
      theta_display = ((theta_display + 180) % 360) - 180
      self.ax2.set_title(f'World Map (Heading: {theta_display:.1f}°)')

    # Draw FIXED occupancy grid (doesn't rotate)
      self.grid.draw(self.ax2)

    # Draw goal
      self.ax2.plot(self.goal[0], self.goal[1], 'g*',
                 markersize=20, label='Goal', zorder=5)

    # Draw trajectory history
      if len(self.state.trajectory_x) > 1:
        self.ax2.plot(self.state.trajectory_x, self.state.trajectory_y,
                     'b-', alpha=0.3, linewidth=1, label='Path')

    # Draw robot position
      self.ax2.plot(self.state.x, self.state.y, 'bo',
                 markersize=12, label='Robot', zorder=10)

    # ===================================================================
    # FIXED: Draw robot heading arrow with correct orientation
    # ===================================================================
    # Convention: theta=0 points in +Y (North), increases CCW
    # Arrow components using sin/cos correctly:
      arrow_len = 0.5
      dx = arrow_len * np.sin(self.state.theta)  # Rightward component
      dy = arrow_len * np.cos(self.state.theta)  # Forward component

    # Draw the heading arrow
      self.ax2.arrow(
        self.state.x, self.state.y,  # Start position
        dx, dy,                        # Direction vector
        head_width=0.2,
        head_length=0.15,
        fc='green',
        ec='green',
        linewidth=3,
        label=f'Heading',
        zorder=11
      )

    # Optional: Draw a small orientation triangle for better visibility
    # This helps see the orientation even during 180° turns
      triangle_size = 0.3
    # Calculate triangle vertices in robot frame
      front = np.array([dx, dy]) / arrow_len * triangle_size
      left = np.array([
        -triangle_size * np.cos(self.state.theta) + dx * 0.3,
        triangle_size * np.sin(self.state.theta) + dy * 0.3
      ])
      right = np.array([
        triangle_size * np.cos(self.state.theta) + dx * 0.3,
        -triangle_size * np.sin(self.state.theta) + dy * 0.3
      ])

      triangle = np.array([
        [self.state.x + front[0], self.state.y + front[1]],
        [self.state.x + left[0], self.state.y + left[1]],
        [self.state.x + right[0], self.state.y + right[1]]
      ])

      from matplotlib.patches import Polygon
      robot_triangle = Polygon(triangle, facecolor='blue',
                            edgecolor='darkblue', alpha=0.5, zorder=10)
      self.ax2.add_patch(robot_triangle)

      self.ax2.legend(loc='upper right')

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
