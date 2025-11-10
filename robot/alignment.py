#!/usr/bin/env python3
"""
Physical Robot Alignment Checker
Helps verify the physical mounting and orientation of the LIDAR
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def check_physical_alignment(lidar):
    """
    Interactive tool to verify LIDAR physical mounting
    """
    print("\n" + "="*70)
    print("PHYSICAL ROBOT ALIGNMENT CHECK")
    print("="*70)
    print("\nThis tool helps verify:")
    print("  1. Which direction is ACTUALLY the robot's front")
    print("  2. If LIDAR is mounted correctly")
    print("  3. If there are any hardware issues")
    print("\n" + "="*70)

    # Start LIDAR
    lidar.start()
    time.sleep(1.5)

    print("\n" + "-"*70)
    print("TEST 1: Cardinal Direction Mapping")
    print("-"*70)
    print("\nWe'll map each cardinal direction one at a time.")

    measurements = {}

    for direction, expected_angle in [("FRONT", 0), ("RIGHT", 90),
                                      ("BACK", 180), ("LEFT", -90)]:
        print(f"\n📍 {direction} side test:")
        print(f"   1. Stand directly at the {direction} of the robot")
        print(f"   2. Stand about 1-2 meters away")
        print(f"   3. Make sure you're clearly visible to LIDAR")

        input(f"\nPress ENTER when you're at the {direction}...")

        # Get scan
        ranges, angles = lidar.get_scan()

        # Find closest object (should be the person)
        min_idx = np.argmin(ranges)
        detected_range = ranges[min_idx]
        detected_angle = np.degrees(angles[min_idx])

        measurements[direction] = {
            'expected': expected_angle,
            'detected': detected_angle,
            'range': detected_range,
            'error': detected_angle - expected_angle
        }

        print(f"   ✓ Detected at: {detected_angle:.1f}°")
        print(f"   ✓ Distance: {detected_range:.2f} m")
        print(f"   ✓ Error: {detected_angle - expected_angle:.1f}°")

    # Analysis
    print("\n" + "="*70)
    print("MEASUREMENT SUMMARY")
    print("="*70)
    print(f"\n{'Direction':<10} {'Expected':<10} {'Detected':<10} {'Error':<10} {'Distance':<10}")
    print("-" * 70)

    for direction in ["FRONT", "RIGHT", "BACK", "LEFT"]:
        m = measurements[direction]
        print(f"{direction:<10} {m['expected']:>8.1f}°  {m['detected']:>8.1f}°  "
              f"{m['error']:>8.1f}°  {m['range']:>8.2f}m")

    # Calculate average error
    errors = [m['error'] for m in measurements.values()]
    avg_error = np.mean(errors)
    std_error = np.std(errors)

    print("-" * 70)
    print(f"Average error: {avg_error:.1f}°")
    print(f"Std deviation: {std_error:.1f}°")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if std_error < 10 and abs(avg_error) < 15:
        # Consistent offset
        print(f"\n✓ GOOD NEWS: Errors are consistent!")
        print(f"\n  All measurements are off by approximately {avg_error:.1f}°")
        print(f"  This is a simple angle offset issue.")
        print(f"\n  FIX: Set angle_offset = np.radians({-avg_error:.2f})")

    elif std_error < 10 and abs(avg_error) > 15:
        # Large consistent offset
        print(f"\n⚠ CONSISTENT OFFSET DETECTED")
        print(f"\n  All measurements are off by {avg_error:.1f}°")
        print(f"  This suggests either:")
        print(f"    - LIDAR is mounted rotated on the robot")
        print(f"    - You're identifying the wrong side as 'front'")
        print(f"\n  FIX: Set angle_offset = np.radians({-avg_error:.2f})")

    elif std_error > 20:
        # Inconsistent errors - major problem
        print(f"\n✗ INCONSISTENT ERRORS DETECTED")
        print(f"\n  Errors vary by {std_error:.1f}° between directions")
        print(f"  This suggests:")
        print(f"    - LIDAR hardware issue (damaged/faulty)")
        print(f"    - Angular distortion in LIDAR data")
        print(f"    - Physical obstruction affecting some angles")
        print(f"\n  RECOMMENDATION: Check LIDAR hardware and mounting")

    # Check if one direction is completely wrong
    print(f"\n" + "-"*70)
    print("DETAILED ANALYSIS BY DIRECTION:")
    print("-"*70)

    for direction in ["FRONT", "RIGHT", "BACK", "LEFT"]:
        m = measurements[direction]
        abs_error = abs(m['error'])

        if abs_error < 15:
            status = "✓ GOOD"
        elif abs_error < 30:
            status = "⚠ MINOR OFFSET"
        elif abs_error < 60:
            status = "⚠ MODERATE OFFSET"
        else:
            status = "✗ MAJOR ISSUE"

        print(f"\n{direction:>6}: {status}")
        print(f"         Expected {direction} at {m['expected']:>4.0f}°")
        print(f"         Actually detected at {m['detected']:>6.1f}°")

        # Suggest what might be wrong
        if abs_error > 60:
            actual_direction = angle_to_direction(m['detected'])
            print(f"         ⚠ This is closer to {actual_direction}!")
            print(f"         → Check if you're standing at the correct side")

    # Visual map
    print(f"\n" + "="*70)
    print("VISUAL ANGLE MAP")
    print("="*70)
    plot_angle_map(measurements)

    lidar.stop()

    return measurements

def angle_to_direction(angle):
    """Convert angle to cardinal direction name"""
    # Normalize to -180 to 180
    angle = ((angle + 180) % 360) - 180

    if -45 <= angle < 45:
        return "FRONT (0°)"
    elif 45 <= angle < 135:
        return "RIGHT (90°)"
    elif angle >= 135 or angle < -135:
        return "BACK (180°)"
    else:
        return "LEFT (-90°)"

def plot_angle_map(measurements):
    """Create visual representation of angle measurements"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Physical Alignment Map\n(Where things should be vs where LIDAR sees them)',
                 fontsize=14, fontweight='bold')

    # Robot at center
    ax.plot(0, 0, 'ko', markersize=30, label='Robot', zorder=10)
    ax.text(0, -0.2, 'ROBOT', ha='center', va='top', fontsize=12, fontweight='bold')

    colors = {'FRONT': 'green', 'RIGHT': 'orange', 'BACK': 'red', 'LEFT': 'blue'}

    for direction, color in colors.items():
        m = measurements[direction]

        # Expected position (green circle with label)
        expected_rad = np.radians(m['expected'])
        exp_x = 1.5 * np.sin(expected_rad)
        exp_y = 1.5 * np.cos(expected_rad)

        ax.scatter(exp_x, exp_y, c='lightgray', s=500, marker='o',
                  edgecolors=color, linewidths=3, alpha=0.5, zorder=5)
        ax.text(exp_x, exp_y, f'Expected\n{direction}\n({m["expected"]:.0f}°)',
               ha='center', va='center', fontsize=9, fontweight='bold')

        # Actual detected position (colored star)
        detected_rad = np.radians(m['detected'])
        det_x = 1.5 * np.sin(detected_rad)
        det_y = 1.5 * np.cos(detected_rad)

        ax.scatter(det_x, det_y, c=color, s=500, marker='*',
                  edgecolors='darkred', linewidths=2, zorder=8)
        ax.text(det_x*1.35, det_y*1.35, f'Detected\n{m["detected"]:.1f}°',
               ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

        # Arrow showing the error
        ax.annotate('', xy=(det_x, det_y), xytext=(exp_x, exp_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.7))

    # Add cardinal direction labels
    for angle_deg, label in [(0, 'N\n(0°)'), (90, 'E\n(90°)'),
                             (180, 'S\n(180°)'), (-90, 'W\n(-90°)')]:
        angle_rad = np.radians(angle_deg)
        dx = 1.8 * np.sin(angle_rad)
        dy = 1.8 * np.cos(angle_rad)
        ax.text(dx, dy, label, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.legend(['Robot', 'Expected Position', 'Actual Detection'],
             loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('lidar_alignment_map.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visual map saved as 'lidar_alignment_map.png'")
    plt.show(block=False)
    plt.pause(10)


# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    """
    Run this to check your robot's physical alignment:

    from physical_alignment_check import check_physical_alignment
    from LD19 import LD19Lidar

    lidar = LD19Lidar(port="/dev/ttyUSB0")
    measurements = check_physical_alignment(lidar)
    """
    print("Import and run: check_physical_alignment(lidar)")
