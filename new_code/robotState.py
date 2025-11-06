
# ============================================================================
# ROBOT STATE & ODOMETRY
# ============================================================================

import math

class RobotState:
    """Track robot pose and velocity"""
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.w = 0.0
        self.path = [(x, y)]

    def update_odometry(self, v: float, w: float, dt: float):
        """Update pose using odometry (dead reckoning)"""
        if abs(w) < 1e-6:
            self.x += v * dt * math.cos(self.theta)
            self.y += v * dt * math.sin(self.theta)
        else:
            self.x += (v/w) * (math.sin(self.theta + w*dt) - math.sin(self.theta))
            self.y += (v/w) * (math.cos(self.theta) - math.cos(self.theta + w*dt))
            self.theta += w * dt

        self.v = v
        self.w = w
        self.path.append((self.x, self.y))
