# ============================================================================
# FUZZY LOGIC CONTROLLER
# ============================================================================
import math
from .configuration import RobotConfig
class FuzzyController:
    """Fuzzy inference system for adaptive DWA parameters"""

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg

    def infer(self, min_obstacle_dist: float, goal_angle: float,
             velocity: float):
        """
        Fuzzy inference based on obstacle proximity and goal alignment.

        Returns:
            dict with 'speed_factor' (0-1) and 'turn_factor' (0-1)
        """
        # Normalize inputs
        goal_angle_norm = abs(goal_angle) / math.pi  # 0-1

        # Fuzzy rules
        if min_obstacle_dist < 0.4:
            # Very close obstacle: slow down, turn sharply
            speed_factor = 0.3
            turn_factor = 1.0
        elif min_obstacle_dist < 0.8:
            # Close obstacle: moderate speed, prefer turning
            speed_factor = 0.6
            turn_factor = 0.8
        elif goal_angle_norm < 0.2 and min_obstacle_dist > 1.5:
            # Goal aligned and clear path: go fast
            speed_factor = 1.0
            turn_factor = 0.3
        elif goal_angle_norm > 0.6:
            # Goal not aligned: turn more
            speed_factor = 0.7
            turn_factor = 0.9
        else:
            # Default moderate behavior
            speed_factor = 0.8
            turn_factor = 0.5

        return {
            'speed_factor': speed_factor,
            'turn_factor': turn_factor,
            'goal_angle': goal_angle
        }
