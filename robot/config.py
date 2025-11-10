from dataclasses import dataclass
import math
from typing import Tuple
# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class RobotConfig:
    """Robot configuration parameters"""
    # Physical parameters
    wheel_base: float = 0.14  # meters between wheels
    max_speed: float = 0.3    # m/s (reduced for safety)
    max_angular_speed: float = math.pi/4  # rad/s

    # Control parameters
    dt: float = 0.1 # time step (seconds)
    goal_tolerance: float = 0.3  # meters
    max_steps: int = 1000

    # LIDAR parameters
    max_lidar_range: float = 8 # meters
    min_obstacle_distance: float = 0.3  # meters

    # Occupancy grid parameters
    grid_resolution: float = 0.05 # meters per cell
    grid_size: float = 8 # meters (total map size)

    # DWA parameters (from research paper)
    v_samples: int = 10  # velocity samples
    w_samples: int = 20  # angular velocity samples
    prediction_horizon: float = 1.5  # seconds to predict ahead
    traj_samples: int = 20  # points per trajectory
    max_accel: float = 0.5  # m/s^2
    max_angular_accel: float = math.pi/2  # rad/s^2

    # Fuzzy parameters
    obstacle_weight: float = 0.4
    goal_weight: float = 0.6

    # ICP Scan Matching parameters
    icp_max_iterations: int = 50
    icp_tolerance: float = 0.001  # convergence threshold
    icp_max_correspondence_dist: float = 0.5  # max distance for point matching
    icp_min_points: int = 10  # minimum points required for scan matching

    # Motor pins (L293D)
    motor_left_pins: Tuple[int, int, int] = (16, 25, 12)   # IN1, IN2, EN1
    motor_right_pins: Tuple[int, int, int] = (5, 26, 13)   # IN3, IN4, EN2
    pwm_frequency: int = 40                              # Hz
