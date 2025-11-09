# ============================================================================
# CONFIGURATION
# ============================================================================
from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class RobotConfig:
    """Robot hardware and navigation configuration"""

    # --------------------------------------------------------------------- #
    # Motor pins (L293D) – keep exactly as you wired them
    # --------------------------------------------------------------------- #
    motor_left_pins: Tuple[int, int, int] = (16, 25, 12)   # IN1, IN2, EN1
    motor_right_pins: Tuple[int, int, int] = (5, 26, 13)   # IN3, IN4, EN2
    pwm_frequency: int = 30                               # Hz

    # --------------------------------------------------------------------- #
    # Physical robot parameters
    # --------------------------------------------------------------------- #
    wheel_base: float = 0.14          # meters (distance between wheels)
    wheel_radius: float = 0.03        # meters
    max_speed: float = 0.62           # m/s  (linear)
    max_angular_speed: float = math.pi / 3   # rad/s (used in DWA window)

    # --------------------------------------------------------------------- #
    # Acceleration limits (used by the dynamic window)
    # --------------------------------------------------------------------- #
    max_accel: float = 1.0            # m/s²   – tune for your motors
    max_angular_accel: float = 2.0    # rad/s² – tune for your motors

    # --------------------------------------------------------------------- #
    # Navigation loop
    # --------------------------------------------------------------------- #
    goal_tolerance: float = 0.30      # meters
    dt: float = 0.10                  # control-loop time step (seconds)
    max_steps: int = 5000

    # --------------------------------------------------------------------- #
    # DWA sampling / prediction
    # --------------------------------------------------------------------- #
    v_samples: int = 5                # number of linear-velocity samples
    w_samples: int = 7                # number of angular-velocity samples
    prediction_horizon: float = 1.0   # seconds to look ahead
    traj_samples: int = 20            # points per simulated trajectory

    # --------------------------------------------------------------------- #
    # LIDAR / map
    # --------------------------------------------------------------------- #
    max_lidar_range: float = 3.5      # metres (LD19 max)
    map_resolution: float = 0.10      # metres per cell
    map_size: Tuple[int, int] = (200, 200)   # cells → 20 m × 20 m world

    # --------------------------------------------------------------------- #
    # Safety / stuck detection
    # --------------------------------------------------------------------- #
    obstacle_threshold: float = 0.5   # metres – stop if closer
    stuck_threshold: float = 0.10     # metres moved in last 10 steps

    # --------------------------------------------------------------------- #
    # Fuzzy logic
    # --------------------------------------------------------------------- #
    fuzzy_enabled: bool = True
