"""
Coordinate System Debugging Tool
Run this BEFORE starting navigation to verify your coordinate frames are correct
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import time

class CoordinateDebugger:
    """Debug tool to verify robot coordinate system alignment"""

    def __init__(self, lidar, motors, cfg):
        self.lidar = lidar
        self.motors = motors
        self.cfg = cfg

        # Create debug visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle('Robot Coordinate System Debugger', fontsize=16, fontweight='bold')

    def test_coordinate_system(self):
        """Interactive test to verify coordinate frames"""
        print("\n" + "="*70)
        print("ROBOT COORDINATE SYSTEM DEBUG TEST")
        print("="*70)
        print("\nThis will help identify coordinate frame misalignment issues")
        print("\nTest sequence:")
        print("  1. Check LIDAR angle=0 direction (should point FORWARD)")
        print("  2. Command forward motion and verify direction")
        print("  3. Measure actual heading from LIDAR landmarks")
        print("\n" + "="*70)

        # Start LIDAR
        self.lidar.start()
        time.sleep(1.0)

        # Test 1: LIDAR Frame Check
        print("\n[TEST 1] LIDAR Frame Verification")
        print("-" * 70)
        self._test_lidar_frame()

        input("\nPress ENTER to continue to motion test...")

        # Test 2: Motion Direction Check
        print("\n[TEST 2] Motion Direction Verification")
        print("-" * 70)
        self._test_motion_direction()

        input("\nPress ENTER to continue to heading measurement...")

        # Test 3: Heading Measurement
        print("\n[TEST 3] Absolute Heading Measurement")
        print("-" * 70)
        self._test_heading_measurement()

        # Cleanup
        self.motors.stop()
        self.lidar.stop()
        plt.close()

        print("\n" + "="*70)
        print("DEBUG TEST COMPLETE")
        print("="*70)

    def _test_lidar_frame(self):
        """Test 1: Verify LIDAR angle=0 points forward"""
        print("\nPlace a distinctive object (box, person, etc.) directly in")
        print("front of the robot, about 1-2 meters away.")
        input("Press ENTER when ready...")

        # Get scan
        ranges, angles = self.lidar.get_scan()

        # Find closest object
        min_idx = np.argmin(ranges)
        min_range = ranges[min_idx]
        min_angle = angles[min_idx]

        print(f"\n✓ Closest object detected:")
        print(f"  Distance: {min_range:.2f} m")
        print(f"  Angle: {np.degrees(min_angle):.1f}°")

        # Visualize
        self._plot_lidar_frame(ranges, angles, min_idx)

        # Analysis
        print(f"\n📊 ANALYSIS:")
        if abs(np.degrees(min_angle)) < 15:
            print(f"  ✓ PASS: Object at {np.degrees(min_angle):.1f}° (close to 0°)")
            print(f"  → LIDAR angle=0 correctly points FORWARD")
        else:
            print(f"  ✗ FAIL: Object at {np.degrees(min_angle):.1f}° (expected ~0°)")
            print(f"  → LIDAR frame may need offset correction!")
            print(f"  → Suggested offset: {-np.degrees(min_angle):.1f}°")

    def _test_motion_direction(self):
        """Test 2: Verify forward command moves robot forward"""
        print("\nThis test will move the robot forward for 1 second.")
        print("Watch the robot carefully and note which direction it moves.")
        print("\nExpected: Robot should move in the direction its FRONT is pointing")
        input("Press ENTER to start motion test...")

        # Get initial scan
        ranges_before, angles_before = self.lidar.get_scan()

        # Move forward
        print("\n🚗 Moving forward...")
        speed = 0.7  # 30% speed
        self.motors.set_motor_speed(speed * self.cfg.pwm_frequency)
        time.sleep(1.0)
        self.motors.stop()
        print("✓ Stopped")

        time.sleep(0.5)

        # Get final scan
        ranges_after, angles_after = self.lidar.get_scan()

        # Analyze motion by comparing scans
        self._analyze_motion_direction(ranges_before, angles_before,
                                      ranges_after, angles_after)

    def _test_heading_measurement(self):
        """Test 3: Measure absolute heading using landmarks"""
        print("\nPlace TWO distinctive objects:")
        print("  Object A: Directly in FRONT (angle should be ~0°)")
        print("  Object B: Directly to the RIGHT (angle should be ~90°)")
        input("Press ENTER when objects are placed...")

        ranges, angles = self.lidar.get_scan()

        # Find prominent objects
        threshold = np.min(ranges) + 0.5  # Objects within 50cm of closest
        close_objects = ranges < threshold

        # Cluster detection (simple)
        object_angles = angles[close_objects]
        object_ranges = ranges[close_objects]

        if len(object_angles) < 2:
            print("⚠️  Not enough objects detected. Try again with clearer objects.")
            return

        # Find two most separated objects
        angle_diffs = []
        for i in range(len(object_angles)):
            for j in range(i+1, len(object_angles)):
                diff = abs(object_angles[i] - object_angles[j])
                angle_diffs.append((diff, i, j))

        angle_diffs.sort(reverse=True)
        _, idx1, idx2 = angle_diffs[0]

        obj1_angle = np.degrees(object_angles[idx1])
        obj2_angle = np.degrees(object_angles[idx2])
        obj1_range = object_ranges[idx1]
        obj2_range = object_ranges[idx2]

        print(f"\n✓ Detected objects:")
        print(f"  Object 1: {obj1_angle:6.1f}° at {obj1_range:.2f}m")
        print(f"  Object 2: {obj2_angle:6.1f}° at {obj2_range:.2f}m")
        print(f"  Angular separation: {abs(obj1_angle - obj2_angle):.1f}°")

        self._plot_heading_measurement(ranges, angles, [idx1, idx2],
                                      [obj1_angle, obj2_angle])

    def _plot_lidar_frame(self, ranges, angles, min_idx):
        """Visualize LIDAR frame with detected object"""
        self.ax1.clear()
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('LIDAR Frame: Is angle=0 pointing FORWARD?')

        # Plot all points
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)
        self.ax1.scatter(x, y, c='lightgray', s=10, alpha=0.5)

        # Highlight closest object
        x_min = ranges[min_idx] * np.sin(angles[min_idx])
        y_min = ranges[min_idx] * np.cos(angles[min_idx])
        self.ax1.scatter(x_min, y_min, c='red', s=200, marker='*',
                        label=f'Object at {np.degrees(angles[min_idx]):.1f}°', zorder=10)

        # Robot at center
        self.ax1.plot(0, 0, 'bo', markersize=15, label='Robot', zorder=5)

        # Draw angle=0 direction (should be FORWARD)
        arrow_len = 2.0
        self.ax1.arrow(0, 0, 0, arrow_len,
                      head_width=0.3, head_length=0.2,
                      fc='green', ec='green', linewidth=3,
                      label='angle=0 (expected FORWARD)', zorder=5)

        # Draw angle grid
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.radians(angle_deg)
            dx = 2.5 * np.sin(angle_rad)
            dy = 2.5 * np.cos(angle_rad)
            self.ax1.plot([0, dx], [0, dy], 'k--', alpha=0.3, linewidth=1)
            self.ax1.text(dx*1.1, dy*1.1, f'{angle_deg}°',
                         ha='center', va='center', fontsize=10)

        self.ax1.legend()
        plt.draw()
        plt.pause(0.1)

    def _analyze_motion_direction(self, ranges_before, angles_before,
                                  ranges_after, angles_after):
        """Analyze which direction robot actually moved"""
        print("\n📊 MOTION ANALYSIS:")

        # Find objects that moved (got closer or farther)
        # Match points by angle (assuming static environment)
        range_changes = ranges_after - ranges_before

        # Find regions that got closer (robot moved toward them)
        got_closer = range_changes < -0.1  # 10cm threshold
        got_farther = range_changes > 0.1

        if np.sum(got_closer) > 0:
            # Find angle where most distance decreased
            close_angles = angles_before[got_closer]
            mean_close_angle = np.degrees(np.mean(close_angles))

            print(f"  Objects got CLOSER at angles around {mean_close_angle:.1f}°")
            print(f"  → Robot moved toward {mean_close_angle:.1f}°")

            if abs(mean_close_angle) < 30:
                print(f"  ✓ PASS: Robot moved forward (toward angle=0)")
            elif abs(mean_close_angle - 90) < 30:
                print(f"  ✗ FAIL: Robot moved RIGHT instead of forward!")
                print(f"  → Your X/Y or sin/cos may be swapped")
            elif abs(mean_close_angle + 90) < 30:
                print(f"  ✗ FAIL: Robot moved LEFT instead of forward!")
                print(f"  → Your X/Y or sin/cos may be swapped")
            elif abs(abs(mean_close_angle) - 180) < 30:
                print(f"  ✗ FAIL: Robot moved BACKWARD instead of forward!")
                print(f"  → Your motor directions may be reversed")
            else:
                print(f"  ⚠️  WARNING: Robot moved at unexpected angle")
        else:
            print(f"  ⚠️  Could not detect clear motion direction")
            print(f"  → Try the test again with more obstacles around")

        # Visualize
        self._plot_motion_analysis(ranges_before, angles_before,
                                   ranges_after, angles_after, range_changes)

    def _plot_motion_analysis(self, ranges_before, angles_before,
                             ranges_after, angles_after, range_changes):
        """Visualize motion analysis"""
        self.ax2.clear()
        self.ax2.set_xlim(-3, 3)
        self.ax2.set_ylim(-3, 3)
        self.ax2.set_aspect('equal')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_title('Motion Direction: Which way did robot move?')

        # Plot before/after
        x_before = ranges_before * np.sin(angles_before)
        y_before = ranges_before * np.cos(angles_before)
        x_after = ranges_after * np.sin(angles_after)
        y_after = ranges_after * np.cos(angles_after)

        self.ax2.scatter(x_before, y_before, c='blue', s=10, alpha=0.3, label='Before')
        self.ax2.scatter(x_after, y_after, c='red', s=10, alpha=0.3, label='After')

        # Highlight regions that got closer
        got_closer = range_changes < -0.1
        if np.sum(got_closer) > 0:
            x_close = ranges_after[got_closer] * np.sin(angles_after[got_closer])
            y_close = ranges_after[got_closer] * np.cos(angles_after[got_closer])
            self.ax2.scatter(x_close, y_close, c='green', s=50,
                           marker='o', label='Got Closer', zorder=10)

        # Robot
        self.ax2.plot(0, 0, 'ko', markersize=15, label='Robot', zorder=5)

        # Expected forward direction
        self.ax2.arrow(0, 0, 0, 1.5,
                      head_width=0.3, head_length=0.2,
                      fc='green', ec='green', linewidth=3,
                      label='Expected: Forward', zorder=5)

        self.ax2.legend()
        plt.draw()
        plt.pause(0.1)

    def _plot_heading_measurement(self, ranges, angles, indices, object_angles):
        """Visualize heading measurement"""
        self.ax1.clear()
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-3, 3)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Heading Measurement from Landmarks')

        # Plot scan
        x = ranges * np.sin(angles)
        y = ranges * np.cos(angles)
        self.ax1.scatter(x, y, c='lightgray', s=10, alpha=0.5)

        # Highlight detected objects
        colors = ['red', 'blue']
        for i, idx in enumerate(indices):
            x_obj = ranges[idx] * np.sin(angles[idx])
            y_obj = ranges[idx] * np.cos(angles[idx])
            self.ax1.scatter(x_obj, y_obj, c=colors[i], s=200, marker='*',
                           label=f'Object {i+1}: {object_angles[i]:.1f}°', zorder=10)

        # Robot
        self.ax1.plot(0, 0, 'ko', markersize=15, label='Robot', zorder=5)

        self.ax1.legend()
        plt.draw()
        plt.pause(0.1)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    """
    Add this to your main script BEFORE starting navigation:

    from coordinate_debugger import CoordinateDebugger

    # Initialize hardware
    lidar = LD19(port="/dev/ttyUSB0")
    motors = MotorController(cfg)

    # Run debug test
    debugger = CoordinateDebugger(lidar, motors, cfg)
    debugger.test_coordinate_system()

    # Now start normal navigation...
    """
    print("Import this module and run the test before navigation")
