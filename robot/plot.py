import matplotlib.pyplot as plt
# ============================================================================
# VISUALIZATION
# ============================================================================
def initialize_plot():
    """Initialize matplotlib figure with two subplots"""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('LD19 LIDAR Raw Scan')

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_title('Occupancy Grid Map with ICP Odometry')

    return fig, ax1, ax2
