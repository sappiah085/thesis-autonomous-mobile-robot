import numpy as np
from .config import RobotConfig
from typing import Tuple
from scipy.spatial import KDTree

# ============================================================================
# ICP SCAN MATCHING FOR ODOMETRY - FIXED COORDINATE SYSTEM
# ============================================================================
class ICPScanMatcher:
    """
    Iterative Closest Point scan matching for odometry estimation

    CRITICAL COORDINATE CONVENTION FIX:
    - LIDAR/Robot frame: angle=0 points in +Y direction (forward), increases CCW
    - When transforming from robot frame to world frame, we must use:
      * x (rightward in robot) → rotated by theta in world
      * y (forward in robot) → rotated by theta in world
    - Standard 2D rotation with theta=0 pointing in +Y (not +X!)
    """

    def __init__(self, cfg: RobotConfig, initial_heading: float = 0.0):
        self.cfg = cfg
        self.previous_scan = None
        self.previous_pose = np.array([0.0, 0.0, initial_heading])  # x, y, theta
        print(f"✓ ICP Scan Matcher initialized (heading: {np.degrees(initial_heading):.1f}°)")
        print(f"  Coordinate system: theta=0 → +Y (forward), increases CCW")

    def scan_to_cartesian(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Convert polar LIDAR scan to Cartesian points

        LIDAR convention: angle=0 is forward (+Y), angle increases CCW
        x = r * sin(angle)  [rightward]
        y = r * cos(angle)  [forward]
        """
        valid_mask = (ranges > 0.1) & (ranges < self.cfg.max_lidar_range)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # LIDAR's "North-up" convention
        x = valid_ranges * np.sin(valid_angles)
        y = valid_ranges * np.cos(valid_angles)

        return np.column_stack((x, y))

    def transform_points(self, points: np.ndarray, dx: float, dy: float, dtheta: float) -> np.ndarray:
        """
        Apply 2D rigid transformation to points
        Rotation around origin, then translation
        """
        cos_theta = np.cos(dtheta)
        sin_theta = np.sin(dtheta)

        # Rotation matrix (standard 2D rotation)
        rotation = np.array([[cos_theta, -sin_theta],
                             [sin_theta, cos_theta]])

        # Rotate then translate
        transformed = points @ rotation.T
        transformed[:, 0] += dx
        transformed[:, 1] += dy

        return transformed

    def icp(self, source: np.ndarray, target: np.ndarray,
            initial_guess: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
        """
        Perform ICP alignment between source and target point clouds

        Args:
            source: Current scan points (Nx2)
            target: Previous scan points (Nx2)
            initial_guess: Initial transformation (dx, dy, dtheta)

        Returns:
            (dx, dy, dtheta) transformation that aligns source to target
        """
        if len(source) < self.cfg.icp_min_points or len(target) < self.cfg.icp_min_points:
            return initial_guess

        dx, dy, dtheta = initial_guess

        # Build KD-tree for fast nearest neighbor search
        tree = KDTree(target)

        prev_error = float('inf')

        for iteration in range(self.cfg.icp_max_iterations):
            # Transform source points with current estimate
            transformed_source = self.transform_points(source, dx, dy, dtheta)

            # Find nearest neighbors
            distances, indices = tree.query(transformed_source)

            # Filter correspondences by distance threshold
            valid_mask = distances < self.cfg.icp_max_correspondence_dist

            if np.sum(valid_mask) < self.cfg.icp_min_points:
                # Not enough valid correspondences
                break

            src_matched = transformed_source[valid_mask]
            tgt_matched = target[indices[valid_mask]]

            # Compute centroids
            src_centroid = np.mean(src_matched, axis=0)
            tgt_centroid = np.mean(tgt_matched, axis=0)

            # Center the points
            src_centered = src_matched - src_centroid
            tgt_centered = tgt_matched - tgt_centroid

            # Compute cross-covariance matrix
            H = src_centered.T @ tgt_centered

            # SVD for rotation estimation
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Extract rotation angle
            rotation_angle = np.arctan2(R[1, 0], R[0, 0])

            # Compute translation
            translation = tgt_centroid - (R @ src_centroid)

            # Update transformation incrementally
            dtheta += rotation_angle

            # Apply rotation to translation offset
            cos_t = np.cos(dtheta)
            sin_t = np.sin(dtheta)
            dx = cos_t * translation[0] - sin_t * translation[1] + dx
            dy = sin_t * translation[0] + cos_t * translation[1] + dy

            # Normalize angle to -pi to pi
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

            # Check convergence
            error = np.mean(distances[valid_mask])

            if abs(prev_error - error) < self.cfg.icp_tolerance:
                break

            prev_error = error

        # Reliability check after convergence
        transformed_source = self.transform_points(source, dx, dy, dtheta)
        final_distances, _ = tree.query(transformed_source)
        final_valid_mask = final_distances < self.cfg.icp_max_correspondence_dist
        num_inliers = np.sum(final_valid_mask)
        mean_error = np.mean(final_distances[final_valid_mask]) if num_inliers > 0 else float('inf')

        # If unreliable (few inliers or high error), fallback to initial guess
        if num_inliers < self.cfg.icp_min_points * 2 or mean_error > 0.1:
            # Don't print every time to reduce spam
            return initial_guess

        return dx, dy, dtheta

    def estimate_odometry(self, ranges: np.ndarray, angles: np.ndarray,
                          commanded_v: float, commanded_w: float) -> Tuple[float, float, float]:
        """
        Estimate odometry from scan matching

        Args:
            ranges, angles: Current LIDAR scan
            commanded_v, commanded_w: Commanded velocities (used as initial guess)

        Returns:
            (dx, dy, dtheta) estimated motion in robot frame
        """
        # Convert current scan to Cartesian
        current_points = self.scan_to_cartesian(ranges, angles)

        if self.previous_scan is None:
            # First scan - no motion yet
            self.previous_scan = current_points
            return 0.0, 0.0, 0.0

        # Check if we have enough points
        if len(current_points) < self.cfg.icp_min_points or len(self.previous_scan) < self.cfg.icp_min_points:
            # Fallback to commanded odometry
            dt = self.cfg.dt
            # In robot frame: forward is +Y
            dx = 0.0
            dy = commanded_v * dt
            dtheta = commanded_w * dt
            self.previous_scan = current_points
            return dx, dy, dtheta

        # Use commanded velocities as initial guess for ICP
        dt = self.cfg.dt
        # Robot frame motion: forward motion is +Y
        initial_dx = 0.0  # No lateral motion typically
        initial_dy = commanded_v * dt  # Forward motion
        initial_dtheta = commanded_w * dt

        # Perform ICP scan matching
        dx, dy, dtheta = self.icp(
            current_points,
            self.previous_scan,
            initial_guess=(initial_dx, initial_dy, initial_dtheta)
        )

        # Update previous scan
        self.previous_scan = current_points

        return dx, dy, dtheta

    def update_pose(self, dx: float, dy: float, dtheta: float) -> Tuple[float, float, float]:
        """
        Update global pose from odometry increment

        CRITICAL FIX: Correct transformation from robot frame to world frame

        Args:
            dx, dy, dtheta: Motion in robot frame
                           (dx=right, dy=forward, dtheta=CCW rotation)

        Returns:
            (x, y, theta) new global pose
        """
        x, y, theta = self.previous_pose

        # FIXED: Transform motion from robot frame to world frame
        # For theta=0 pointing in +Y direction (not +X):
        # Standard rotation matrix rotates vectors, but our theta measures
        # angle from +Y axis (North), not +X axis (East)

        # When theta=0 (pointing North/+Y):
        #   - Robot's +Y (forward) should map to World's +Y
        #   - Robot's +X (right) should map to World's +X

        # When theta=90° (pointing East/+X):
        #   - Robot's +Y (forward) should map to World's +X
        #   - Robot's +X (right) should map to World's -Y

        # This is achieved by:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rotate robot-frame motion into world frame
        # Using "North-up" convention where theta=0 is +Y
        world_dx = sin_theta * dy + cos_theta * dx   # dx component in world
        world_dy = cos_theta * dy - sin_theta * dx   # dy component in world

        # Update pose
        new_x = x + world_dx
        new_y = y + world_dy
        new_theta = theta + dtheta

        # Normalize angle to -pi to pi using atan2
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        self.previous_pose = np.array([new_x, new_y, new_theta])

        return new_x, new_y, new_theta

    def set_pose(self, x: float, y: float, theta: float):
        """Manually set the current pose (useful for initialization or corrections)"""
        self.previous_pose = np.array([x, y, theta])
        print(f"Pose set to: ({x:.2f}, {y:.2f}, {np.degrees(theta):.1f}°)")
