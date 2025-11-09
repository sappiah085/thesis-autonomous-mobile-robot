"""
Complete Autonomous Navigation System with ICP Scan Matching
Includes: Configuration, State, Grid, Fuzzy Logic, DWA Planner, ICP Odometry, Visualization
Compatible with LD19 LIDAR (uses scan matching for odometry)
"""
from robot.lidar import LD19Lidar
from robot.autonomousRobot import AutonomousRobot
from robot.config import RobotConfig
# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Configuration
    config = RobotConfig()

    # Initialize LD19 LIDAR
    lidar = LD19Lidar(port='/dev/ttyAMA0', baudrate=230400, min_confidence=50)

    # Create robot with goal at (3, 3)
    robot = AutonomousRobot(config, lidar, goal_x=1, goal_y=1)

    # Start navigation
    robot.start()

    # Keep plot open
    plt.ioff()
    plt.show()
