# ============================================================================
# CONFIGURATION
# ============================================================================
from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class RobotConfig:
    """Robot hardware and navigation configuration"""
    # Motor pins (L293D)
    motor_left_pins: Tuple[int, int, int] = (16, 25, 12)   # IN1, IN2, EN1
    motor_right_pins: Tuple[int, int, int] = (5, 26, 13)   # IN3, IN4, EN2
    pwm_frequency: int = 30  # Hz

    # Robot physical parameters
    wheel_base: float = 0.14  # meters (distance between wheels)
    wheel_radius: float = 0.03  # meters
    max_speed: float = 0.62  # m/s
    max_angular_velocity: float = math.pi / 3  # rad/s

    # Navigation parameters
    goal_tolerance: float = 0.3  # meters
    dt: float = 0.1  # control loop time step
    max_steps: int = 5000

    # DWA parameters
    num_v_samples: int = 5
    num_w_samples: int = 7
    predict_time: float = 1.0

    # Map parameters
    map_resolution: float = 0.10  # meters per cell
    map_size: Tuple[int, int] = (200, 200)  # cells (10m x 10m world)

    # Safety parameters
    obstacle_threshold: float = 0.5  # meters
    stuck_threshold: float = 0.1  # meters movement in 10 steps

    # Fuzzy logic parameters
    fuzzy_enabled: bool = True
