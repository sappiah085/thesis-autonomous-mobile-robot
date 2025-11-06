# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
from new_code.configuration import RobotConfig
from new_code.lidar import LD19Lidar
from new_code.autonomous_navigation import AutonomousRobot

def main():
    """Main entry point for hardware robot"""
    print("=" * 60)
    print("AUTONOMOUS ROBOT NAVIGATION - Hardware Implementation")
    print("=" * 60)

    # Configuration
    cfg = RobotConfig()

    # LIDAR selection
    # Option 1: Placeholder (for testing without LIDAR)
    lidar = LD19Lidar()

    # Option 2: RPLidar A1/A2 (uncomment when you have RPLidar hardware)
    # Install: pip install rplidar
    # lidar = RPLidarA1(port='/dev/ttyUSB0')

    # Option 3: YDLidar X2/X4 or M1C1 module (uncomment for your LIDAR)
    # Install: pip install pyserial
    # lidar = YDLidarX2(port='/dev/ttyUSB0', baudrate=115200)
    # Note: Try baudrate=128000 if 115200 doesn't work

    goal_x = 3.0
    goal_y = 3.0
    # Create and start robot
    robot = AutonomousRobot(cfg, lidar, goal_x, goal_y)
    robot.start()
    print("\n✅ Navigation completed!")
    print(f"Final position: ({robot.state.x:.2f}, {robot.state.y:.2f})")
    print(f"Path length: {len(robot.state.path)} points")

if __name__ == "__main__":
    main()
