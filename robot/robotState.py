# ============================================================================
# ROBOT STATE
# ============================================================================
class RobotState:
    """Robot pose and velocity state"""
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0  # linear velocity
        self.w = 0.0  # angular velocity

        # Store trajectory history
        self.trajectory_x = [x]
        self.trajectory_y = [y]
        self.max_history = 500

    def update(self, x: float, y: float, theta: float):
        """Update pose and store in trajectory"""
        self.x = x
        self.y = y
        self.theta = theta

        self.trajectory_x.append(x)
        self.trajectory_y.append(y)

        # Limit history size
        if len(self.trajectory_x) > self.max_history:
            self.trajectory_x.pop(0)
            self.trajectory_y.pop(0)

