#!/usr/bin/env python3
"""
LIDAR Calibration Tool
Finds the exact angle offset needed to align LIDAR frame with robot frame
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import time

class LidarCalibrator:
    """Interactive calibration tool to find LIDAR angle offset"""

    def __init__(self, lidar):
        self.lidar = lidar
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))

    def calibrate(self):
        """
        Interactive calibration procedure
        Returns the angle offset needed to align LIDAR with robot
        """
        print("\n" + "="*70)
        print("LIDAR ANGLE OFFSET CALIBRATION")
        print("="*70)
        print("\nThis tool will find the exact angle offset to align your LIDAR")
        print("with your robot's forward direction.")
        print("\nSetup Instructions:")
        print("  1. Place robot in an open area")
        print("  2. Place a DISTINCTIVE MARKER (box, pole, person) directly")
        print("     in front of the robot, about 1-2 meters away")
        print("  3. Make sure the marker is clearly in front where the")
        print("     robot's 'nose' is pointing")
        print("\n" + "="*70)

        input("\nPress ENTER when marker is placed in front of robot...")

        # Start LIDAR if not running
        was_running = hasattr(self.lidar, '_running') and self.lidar._running
        if not was_running:
            self.lidar.start()
            time.sleep(1.5)

        # Get scan
        ranges, angles = self.lidar.get_scan()

        print(f"\n✓ Scan captured: {len(ranges)} points")

        # Find the marker (closest object)
        min_idx = np.argmin(ranges)
        marker_range = ranges[min_idx]
        marker_angle_rad = angles[min_idx]
        marker_angle_deg = np.degrees(marker_angle_rad)

        print(f"\n📍 Marker detected:")
        print(f"   Distance: {marker_range:.2f} m")
        print(f"   Current angle: {marker_angle_deg:.2f}°")

        # Visualize current state
        self._plot_calibration(ranges, angles, min_idx, marker_angle_deg, offset=0)

        print(f"\n" + "-"*70)
        print(f"CALIBRATION RESULT:")
        print(f"-"*70)

        # The offset needed is the negative of the detected angle
        # Because we want to rotate the angles so the marker ends up at 0°
        required_offset_deg = -marker_angle_deg
        required_offset_rad = -marker_angle_rad

        print(f"\n✓ Required angle offset: {required_offset_deg:.2f}°")
        print(f"  (or {required_offset_rad:.4f} radians)")

        # Show what it will look like with correction
        corrected_angles = angles + required_offset_rad
        corrected_angles = np.arctan2(np.sin(corrected_angles), np.cos(corrected_angles))

        corrected_marker_angle = np.degrees(corrected_angles[min_idx])

        print(f"\n  After applying this offset:")
        print(f"    Marker will be at: {corrected_marker_angle:.2f}° (should be ~0°)")

        # Visualize corrected state
        self._plot_calibration(ranges, corrected_angles, min_idx,
                             corrected_marker_angle, offset=required_offset_deg)

        print(f"\n" + "="*70)
        print(f"UPDATE YOUR CODE:")
        print(f"="*70)
        print(f"\nIn your LD19Lidar class __init__ method, change:")
        print(f"\n  self.angle_offset = np.radians({required_offset_deg:.2f})")
        print(f"\nOr equivalently:")
        print(f"\n  self.angle_offset = {required_offset_rad:.6f}  # radians")
        print(f"\n" + "="*70)

        # Verification test
        print(f"\n🧪 VERIFICATION TEST:")
        print(f"-"*70)
        print(f"\nNow let's verify with objects at known positions...")

        input("\nPlace objects at these positions and press ENTER:")
        print("  - Object A: Directly in FRONT (0°)")
        print("  - Object B: Directly to the RIGHT (90°)")
        print("  - Object C: Directly BEHIND (180° or -180°)")
        print("  - Object D: Directly to the LEFT (-90°)")

        # Get new scan
        time.sleep(0.5)
        ranges_verify, angles_verify = self.lidar.get_scan()
        angles_verify_corrected = angles_verify + required_offset_rad
        angles_verify_corrected = np.arctan2(np.sin(angles_verify_corrected),
                                            np.cos(angles_verify_corrected))

        # Find prominent objects (local minima in range)
        close_threshold = np.percentile(ranges_verify, 20)  # Bottom 20% = close objects
        close_mask = ranges_verify < close_threshold

        close_angles = np.degrees(angles_verify_corrected[close_mask])
        close_ranges = ranges_verify[close_mask]

        # Cluster angles into 4 cardinal directions
        cardinal_directions = {
            'Front (0°)': (close_angles, 0, 30),
            'Right (90°)': (close_angles, 90, 30),
            'Back (180°)': (close_angles, 180, 30),
            'Left (-90°)': (close_angles, -90, 30)
        }

        print(f"\n✓ Objects detected at corrected angles:")
        for name, (angles_array, target, tolerance) in cardinal_directions.items():
            # Find angles near this cardinal direction
            if target == 180:
                # Special case for ±180°
                near_mask = (np.abs(angles_array - 180) < tolerance) | \
                           (np.abs(angles_array + 180) < tolerance)
            else:
                near_mask = np.abs(angles_array - target) < tolerance

            if np.any(near_mask):
                detected_angle = np.mean(angles_array[near_mask])
                error = detected_angle - target
                if target == 180 and detected_angle < 0:
                    error = (detected_angle + 360) - target

                status = "✓" if abs(error) < 15 else "⚠"
                print(f"  {status} {name}: {detected_angle:6.1f}° (error: {error:+.1f}°)")
            else:
                print(f"  ✗ {name}: No object detected")

        # Final visualization
        self._plot_verification(ranges_verify, angles_verify_corrected, required_offset_deg)

        if not was_running:
            self.lidar.stop()

        return required_offset_rad

    def _plot_calibration(self, ranges, angles, marker_idx, marker_angle, offset):
        """Plot current calibration state"""
        self.ax.clear()
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        if offset == 0:
            title = f'Current State (before correction)\nMarker at {marker_angle:.1f}°'
        else:
            title = f'After Correction (offset = {offset:.1f}°)\nMarker at {marker_angle:.1f}°'

        self.ax.set_title(title, fontsize=14, fontweight='bold')

        # Plot all scan points
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)
        self.ax.scatter(x, y, c='lightgray', s=10, alpha=0.5, label='LIDAR scan')

        # Highlight marker
        x_marker = ranges[marker_idx] * np.sin(angles[marker_idx])
        y_marker = ranges[marker_idx] * np.cos(angles[marker_idx])
        self.ax.scatter(x_marker, y_marker, c='red', s=300, marker='*',
                       edgecolors='darkred', linewidths=2,
                       label=f'Marker at {marker_angle:.1f}°', zorder=10)

        # Robot at center
        self.ax.plot(0, 0, 'bo', markersize=20, label='Robot', zorder=5)

        # Expected forward direction (0°)
        arrow_len = 2.5
        self.ax.arrow(0, 0, 0, arrow_len,
                     head_width=0.3, head_length=0.2,
                     fc='green', ec='green', linewidth=3,
                     label='Expected Front (0°)', zorder=5)

        # Draw cardinal direction guides
        for angle_deg, label in [(0, '0°\nFront'), (90, '90°\nRight'),
                                  (180, '180°\nBack'), (-90, '-90°\nLeft')]:
            angle_rad = np.radians(angle_deg)
            dx = 2.8 * np.sin(angle_rad)
            dy = 2.8 * np.cos(angle_rad)
            self.ax.plot([0, dx], [0, dy], 'k--', alpha=0.2, linewidth=1)
            self.ax.text(dx*1.15, dy*1.15, label, ha='center', va='center',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.ax.legend(loc='upper right')
        plt.draw()
        plt.pause(0.1)

    def _plot_verification(self, ranges, angles, offset):
        """Plot verification with multiple objects"""
        self.ax.clear()
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'Verification (offset = {offset:.1f}°)',
                         fontsize=14, fontweight='bold')

        # Plot scan
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)

        # Color by angle for clarity
        angles_deg = np.degrees(angles)
        self.ax.scatter(x, y, c=angles_deg, s=20, alpha=0.6,
                       cmap='hsv', vmin=-180, vmax=180, label='LIDAR scan')

        # Robot
        self.ax.plot(0, 0, 'ko', markersize=20, label='Robot', zorder=5)

        # Cardinal directions
        for angle_deg, label, color in [(0, '0° Front', 'green'),
                                        (90, '90° Right', 'orange'),
                                        (180, '180° Back', 'red'),
                                        (-90, '-90° Left', 'blue')]:
            angle_rad = np.radians(angle_deg)
            arrow_len = 3.0
            dx = arrow_len * np.sin(angle_rad)
            dy = arrow_len * np.cos(angle_rad)
            self.ax.arrow(0, 0, dx*0.9, dy*0.9,
                         head_width=0.2, head_length=0.15,
                         fc=color, ec=color, linewidth=2, alpha=0.6, zorder=4)
            self.ax.text(dx*1.1, dy*1.1, label, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.colorbar(self.ax.collections[0], ax=self.ax, label='Angle (degrees)')
        self.ax.legend(loc='upper left')
        plt.draw()
        plt.pause(0.1)


# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    """
    Run this script to calibrate your LIDAR:

    python lidar_calibration.py

    Or import and use in your main script:

    from lidar_calibration import LidarCalibrator

    lidar = LD19Lidar(port="/dev/ttyUSB0")
    calibrator = LidarCalibrator(lidar)
    offset_radians = calibrator.calibrate()

    # Then update your LD19Lidar class with the offset
    """
    print("Import this module and run LidarCalibrator(lidar).calibrate()")
