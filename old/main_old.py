import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from lidar_sim import Lidar
from slam import SLAM
from dwa import DWA
from fuzzy_logic import FuzzyController
from motor_control import MotorController
from config import DT, MAP_SIZE, GOAL_TOLERANCE
import os

def main():
    lidar = Lidar()
    slam = SLAM()
    dwa = DWA()
    fuzzy = FuzzyController()
    motors = MotorController()

    robot_pose = [0.0, 0.0, 0.0]
    goal = [5.0, 5.0]
    environment = slam

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-MAP_SIZE / 2, MAP_SIZE / 2)
    ax.set_ylim(-MAP_SIZE / 2, MAP_SIZE / 2)
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_count = 0

    try:
        while True:
            ranges, angles = lidar.get_scan(robot_pose, environment)
            slam.update(robot_pose, ranges, angles)
            obstacle_dist = min(ranges) if ranges.size > 0 else float('inf')
            fuzzy_speed, fuzzy_turn = fuzzy.compute(obstacle_dist)
            dwa_speed, dwa_turn = dwa.plan(robot_pose, goal, ranges, angles)
            v = fuzzy_speed if obstacle_dist < 1.0 else dwa_speed
            w = fuzzy_turn if obstacle_dist < 1.0 else dwa_turn

            goal_dist = np.sqrt((robot_pose[0] - goal[0])**2 + (robot_pose[1] - goal[1])**2)
            print(f"Robot pose: {robot_pose}, Velocity: v={v:.2f}, w={w:.2f}, Goal distance: {goal_dist:.2f}, Obstacle distance: {obstacle_dist:.2f}")
            if v < 0.1 and abs(w) < 0.1:
                print("Robot stuck! Attempting recovery...")

            if goal_dist <= GOAL_TOLERANCE:
                print("Goal reached!")
                motors.stop()
                break

            robot_pose[0] += v * np.cos(robot_pose[2]) * DT
            robot_pose[1] += v * np.sin(robot_pose[2]) * DT
            robot_pose[2] += w * DT
            robot_pose[2] = (robot_pose[2] + np.pi) % (2 * np.pi) - np.pi
            motors.set_velocity(v, w)

            ax.clear()
            ax.imshow(slam.map.T, origin='lower', extent=(-MAP_SIZE / 2, MAP_SIZE / 2, -MAP_SIZE / 2, MAP_SIZE / 2), cmap='gray')
            ax.plot(robot_pose[0], robot_pose[1], 'ro', label='Robot')
            ax.plot(goal[0], goal[1], 'g*', label='Goal')
            ax.legend()
            plt.savefig(f"{output_dir}/plot_{plot_count:04d}.png")
            plot_count += 1
            plt.pause(0.1)

    except KeyboardInterrupt:
        motors.stop()
        plt.savefig(f"{output_dir}/final_plot.png")
        plt.close()

if __name__ == "__main__":
    main()
