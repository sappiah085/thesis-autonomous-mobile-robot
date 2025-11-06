# config.py
# Robot and simulation parameters
WHEEL_RADIUS = 0.05  # meters
WHEEL_BASE = 0.2     # meters
MAX_SPEED = 0.5      # m/s
MAX_ANGULAR_SPEED = 2.0  # rad/s
DT = 0.1             # time step (s)

# LiDAR simulation (RPLIDAR A1 specs)
LIDAR_MAX_RANGE = 12.0    # max range (m)
LIDAR_MIN_RANGE = 0.15    # min range (m)
LIDAR_ANGULAR_RES = 1.0   # degrees per point
LIDAR_NOISE_STD = 0.02    # noise in meters

# SLAM parameters
MAP_SIZE = 20.0      # map size (m)
MAP_RESOLUTION = 0.1 # meters per cell
PARTICLE_COUNT = 10  # reduced for performance

# DWA parameters
MAX_ACCEL = 0.5      # m/s^2
MAX_ANGULAR_ACCEL = 1.5  # rad/s^2
GOAL_TOLERANCE = 0.5 # meters
OBSTACLE_THRESHOLD = 0.5  # meters
GOAL_WEIGHT = 1.0    # prioritize goal
OBSTACLE_WEIGHT = 0.5 # reduce avoidance

# Fuzzy logic parameters
FUZZY_OBSTACLE_RANGE = [0, 2.0]  # min/max distance
FUZZY_SPEED_RANGE = [0, MAX_SPEED]
FUZZY_TURN_RANGE = [-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED]
