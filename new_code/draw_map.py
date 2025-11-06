import matplotlib.pyplot as plt
from .lidar import LidarInterface
import numpy as np
def initializeMap():
      # Setup plot
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Raw LIDAR scan
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('LD19 LIDAR Raw Scan')

        # Right plot: Occupancy grid
        ax2.set_xlim(-5, 5)
        ax2.set_ylim(-5, 5)
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.set_title('Occupancy Grid Map')
        return ax1, ax2

def drawMap(ranges, angles, scatter):
    # Convert to cartesian coordinates
    x = ranges * np.sin(angles)
    y = ranges * np.cos(angles)
    # Update plot
    scatter.set_offsets(np.c_[x, y])
    plt.pause(0.05)
