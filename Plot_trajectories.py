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
    """Create plots for trajectories grouped by time with original gates only.

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

    # Use the filtered groups
    trajectory_groups = complete_trajectory_groups

    # Sort groups by time (convert to comparable format)
    def time_sort_key(time_str):
        try:
            h, m, s = map(int, time_str.split("_"))
            return h * 3600 + m * 60 + s  # Convert to seconds for sorting
        except:
            return 0

    sorted_groups = sorted(trajectory_groups.items(), key=lambda x: time_sort_key(x[0]))

    print(f"Plotting {len(sorted_groups)} complete trajectory sets")

    # Define original gates for reference
    org_gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Determine subplot layout based on number of complete time groups
    num_groups = len(sorted_groups)
    if num_groups <= 3:
        rows, cols = 1, num_groups
        figsize = (6 * num_groups, 6)
    elif num_groups <= 6:
        rows, cols = 2, 3
        figsize = (18, 12)
    else:
        rows, cols = 3, 4
        figsize = (24, 18)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("Trajectory Comparison - Grouped by Time", fontsize=16)

    # Handle single subplot case
    if num_groups == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

    # Colors for different first indices (0-9)
    colors = [
        "blue",
        "red",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    # Store handles and labels for combined legend
    all_handles = []
    all_labels = []

    # Plot each trajectory group (grouped by time)
    for group_idx, (time_key, group_files) in enumerate(sorted_groups):
        if group_idx >= len(axes):
            break

        ax = axes[group_idx]

        # Sort files within group by first index
        group_files_sorted = []
        for file_path in group_files:
            filename = os.path.basename(file_path)
            if "_" in filename:
                first_index = filename.split("_")[0]
                try:
                    first_index_int = int(first_index)
                    group_files_sorted.append((first_index_int, file_path))
                except ValueError:
                    continue

        # Sort by first index
        group_files_sorted.sort(key=lambda x: x[0])

        # Plot each file in the group
        for first_index_int, file_path in group_files_sorted:
            filename = os.path.basename(file_path)
            color = colors[first_index_int % len(colors)]

            # Create label based on first index
            file_label = f"Trajectory {first_index_int}"

            try:
                data = np.load(file_path, allow_pickle=True)
                num_trajectories = int(data["num_trajectories"])

                # Plot first trajectory from file
                for i in range(num_trajectories):
                    traj_key = f"traj_{i}"
                    if traj_key in data:
                        traj_data = data[traj_key].item()

                        # Plot trajectory
                        line = ax.plot(
                            traj_data["x"],
                            traj_data["y"],
                            color=color,
                            linewidth=2,
                            label=file_label,
                            alpha=0.8,
                        )
                        break  # Only plot first trajectory from file

            except Exception as e:
                # Silently continue on error
                continue

        # Plot original gates only
        org_gate_x = [gate["pos"][0] for gate in org_gates]
        org_gate_y = [gate["pos"][1] for gate in org_gates]

        gates_scatter = ax.scatter(
            org_gate_x,
            org_gate_y,
            color="green",
            marker="s",
            s=100,
            label="Gates" if group_idx == 0 else "",  # Only label once
            alpha=0.8,
            zorder=5,
        )

        # Set plot properties
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_title(f"Time {time_key}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Set axis limits
        ax.set_ylim(bottom=-1.5, top=1.5)
        ax.set_xlim(-0.7, 1.6)

        # Collect legend handles and labels from first subplot only
        if group_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

    # Hide unused subplots
    for i in range(num_groups, len(axes)):
        axes[i].set_visible(False)

    # CREATE SINGLE LEGEND BELOW ALL PLOTS
    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(all_handles, all_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Create legend below all subplots
    fig.legend(
        unique_labels,
        unique_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(unique_labels), 6),  # Max 6 columns
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add space at bottom for legend
    # plt.savefig("flight_logs/figs/trajectory_multiple.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # create multiple trajectory plots
    plot_trajectory_multiple()