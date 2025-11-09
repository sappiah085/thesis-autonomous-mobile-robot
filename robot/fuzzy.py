from .config import RobotConfig
import numpy as np
# ============================================================================
# FUZZY LOGIC CONTROLLER
# ============================================================================
class FuzzyController:
    """Fuzzy logic for adaptive DWA weight adjustment"""
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg

    def membership_near(self, dist: float) -> float:
        """Membership function for 'near' obstacle"""
        if dist < 0.3:
            return 1.0
        elif dist < 0.6:
            return (0.6 - dist) / 0.3
        return 0.0

    def membership_medium(self, dist: float) -> float:
        """Membership function for 'medium' distance obstacle"""
        if dist < 0.3:
            return 0.0
        elif dist < 0.6:
            return (dist - 0.3) / 0.3
        elif dist < 1.0:
            return (1.0 - dist) / 0.4
        return 0.0

    def membership_far(self, dist: float) -> float:
        """Membership function for 'far' obstacle"""
        if dist < 0.6:
            return 0.0
        elif dist < 1.0:
            return (dist - 0.6) / 0.4
        return 1.0

    def membership_aligned(self, angle_diff: float) -> float:
        """Membership for aligned with goal"""
        angle_diff = abs(angle_diff)
        if angle_diff < 0.2:
            return 1.0
        elif angle_diff < 0.8:
            return (0.8 - angle_diff) / 0.6
        return 0.0

    def infer(self, min_obstacle_dist: float, goal_angle: float, velocity: float) -> dict:
        """
        Fuzzy inference for speed and turn factors

        Returns: {'speed_factor': float, 'turn_factor': float}
        """
        # Membership degrees
        near = self.membership_near(min_obstacle_dist)
        medium = self.membership_medium(min_obstacle_dist)
        far = self.membership_far(min_obstacle_dist)

        aligned = self.membership_aligned(goal_angle)
        misaligned = 1.0 - aligned

        # Fuzzy rules for speed factor
        speed_1 = near * 0.3
        speed_2 = medium * 0.6
        speed_3 = far * 1.0

        speed_factor = max(speed_1, speed_2, speed_3)

        # Fuzzy rules for turn factor
        turn_1 = misaligned * 0.8
        turn_2 = aligned * 0.2
        turn_3 = near * 0.9

        turn_factor = max(turn_1, turn_2, turn_3)

        return {
            'speed_factor': speed_factor,
            'turn_factor': turn_factor
        }
