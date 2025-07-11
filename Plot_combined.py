"""This module provides functionality to visualize drone racing trajectories with gates and obstacles in a 3D plot.

It processes trajectory data from NPZ files and renders them using matplotlib.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation as R

def plot_trajectory_multiple(directory: str = "flight_logs"):
    """Create plots fore trajectories grouped by trajectory type (initial and gate replans).

    Args:
        directory (str): Directory containing trajectory NPZ files
    """
    # Find trajectory files
    search_pattern = os.path.join(directory, "*.npz")
    trajectory_files = glob.glob(search_pattern)

    if not trajectory_files:
        print(f"No trajectory files found in {directory}")
        return

    # Print available files for debugging
    print("Available trajectory files:")
    for file in trajectory_files:
        print(f"  - {os.path.basename(file)}")

    # Sort files to get consistent ordering
    trajectory_files.sort()

    # Group files by time (the last three parts after splitting by '_')
    trajectory_groups = {}
    for file_path in trajectory_files:
        filename = os.path.basename(file_path)
        # Extract time from filename (e.g., "-1_planned_trajectories_10_10_32.npz" -> "10_10_32")
        if "_" in filename and "planned_trajectories" in filename:
            parts = filename.split("_")
            if len(parts) >= 6:  # Expecting at least gate_planned_trajectories_h_m_s.npz
                time_key = "_".join(parts[-3:]).replace(".npz", "")  # Get last 3 parts (h_m_s)
                if time_key not in trajectory_groups:
                    trajectory_groups[time_key] = []
                trajectory_groups[time_key].append(file_path)

    # Filter groups to only include those with all 5 trajectories
    complete_trajectory_groups = {}
    expected_gate_indices = {-1, 0, 1, 2, 3}

    for time_key, group_files in trajectory_groups.items():
        # Extract gate indices from filenames in this group
        gate_indices = set()
        for file_path in group_files:
            filename = os.path.basename(file_path)
            if "_" in filename:
                try:
                    gate_index = int(filename.split("_")[0])
                    gate_indices.add(gate_index)
                except ValueError:
                    continue
        
        # Only keep groups that have all expected gate indices
        if gate_indices == expected_gate_indices:
            complete_trajectory_groups[time_key] = group_files
            print(f"Complete trajectory set found for time {time_key}: {len(group_files)} files")
        else:
            missing = expected_gate_indices - gate_indices
            print(f"Incomplete trajectory set for time {time_key}: missing gate indices {missing}")

    if not complete_trajectory_groups:
        print("No complete trajectory sets found (need all 5 trajectories: -1, 0, 1, 2, 3)")
        return

    print(f"Found {len(complete_trajectory_groups)} complete trajectory sets")

    # Define original gates for reference
    org_gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Create 5 subplots (one for each trajectory type)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle("Trajectory Comparison by Type", fontsize=16)

    # Gate indices to plot
    gate_indices = [-1, 0, 1, 2, 3]
    plot_titles = ["Initial Trajectory", "Gate 0 Replan", "Gate 1 Replan", "Gate 2 Replan", "Gate 3 Replan"]
    
    # Generate colors for different time groups
    cmap = get_cmap('tab10')
    time_keys = sorted(complete_trajectory_groups.keys(), key=lambda x: tuple(map(int, x.split("_"))))
    time_colors = {time_key: cmap(i % 10) for i, time_key in enumerate(time_keys)}

    # Plot each trajectory type
    for plot_idx, gate_index in enumerate(gate_indices):
        ax = axes[plot_idx]
        ax.set_title(plot_titles[plot_idx], fontsize=12)

        # Plot trajectories from all time groups for this gate index
        for time_key, group_files in complete_trajectory_groups.items():
            # Find the file for this gate index in this time group
            target_file = None
            for file_path in group_files:
                filename = os.path.basename(file_path)
                if "_" in filename:
                    try:
                        file_gate_index = int(filename.split("_")[0])
                        if file_gate_index == gate_index:
                            target_file = file_path
                            break
                    except ValueError:
                        continue

            if target_file:
                try:
                    data = np.load(target_file, allow_pickle=True)
                    num_trajectories = int(data["num_trajectories"])

                    # Plot first trajectory from file
                    for i in range(num_trajectories):
                        traj_key = f"traj_{i}"
                        if traj_key in data:
                            traj_data = data[traj_key].item()

                            # Plot trajectory (no label to avoid cluttering legend)
                            ax.plot(
                                traj_data["x"],
                                traj_data["y"],
                                color=time_colors[time_key],
                                linewidth=2,
                                alpha=0.8,
                            )
                            break  # Only plot first trajectory from file

                except Exception as e:
                    print(f"Error loading {target_file}: {e}")
                    continue

        # Plot original gates
        org_gate_x = [gate["pos"][0] for gate in org_gates]
        org_gate_y = [gate["pos"][1] for gate in org_gates]

        ax.scatter(
            org_gate_x,
            org_gate_y,
            color="green",
            marker="s",
            s=100,
            label="Gates",
            alpha=0.8,
            zorder=5,
        )

        # Plot obstacles
        obstacles = [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0.0, 1.5, 1.4], [-0.5, 0.5, 1.4]]
        obstacle_x = [obs[0] for obs in obstacles]
        obstacle_y = [obs[1] for obs in obstacles]

        ax.scatter(
            obstacle_x,
            obstacle_y,
            color="red",
            marker="o",
            s=100,
            label="Obstacles",
            alpha=0.8,
            zorder=5,
        )

        # Set plot properties
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Set axis limits
        ax.set_ylim(bottom=-1.5, top=1.5)
        ax.set_xlim(-0.7, 1.6)

        # Add legend for gates and obstacles only
        ax.legend(loc='upper right', fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # create multiple trajectory plots
    plot_trajectory_multiple()