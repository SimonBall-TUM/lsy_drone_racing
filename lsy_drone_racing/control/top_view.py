"""This module provides functionality to visualize drone racing trajectories with gates and obstacles in a 3D plot.

It processes trajectory data from NPZ files and renders them using matplotlib.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation as R


def plot_trajectories(directory: str = "flight_logs", show_waypoints: bool = True):
    """Plot all trajectory NPZ files with gates and obstacles in 3D.

    Args:
        directory (str): Directory containing trajectory NPZ files
        show_waypoints (bool): Whether to plot waypoints as markers
    """
    # Find all NPZ files in the directory
    search_pattern = os.path.join(directory, "*_trajectories.npz")
    trajectory_files = glob.glob(search_pattern)

    if not trajectory_files:
        print(f"No trajectory files found in {directory}")
        return

    print(f"Found {len(trajectory_files)} trajectory files")

    # Create figure for 3D plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Use colormap to distinguish between different files
    cmap = get_cmap("viridis")
    file_colors = cmap(np.linspace(0, 1, len(trajectory_files)))

    # Track legend entries
    legend_entries = []

    # Process each trajectory file
    for file_idx, trajectory_file in enumerate(trajectory_files):
        try:
            # Load trajectory data
            data = np.load(trajectory_file, allow_pickle=True)
            num_trajectories = int(data["num_trajectories"])

            file_basename = os.path.basename(trajectory_file)
            print(f"Processing {file_basename} with {num_trajectories} trajectories")

            # Plot each trajectory in this file
            for i in range(num_trajectories):
                traj_key = f"traj_{i}"
                if traj_key in data:
                    # Extract trajectory data
                    traj_data = data[traj_key].item()  # Convert from 0d array to dict

                    # Plot the main trajectory
                    tick = traj_data["tick"]
                    label = f"{file_basename} (tick {tick})"

                    # Use a gradient of the file color for each trajectory in the file
                    traj_color = file_colors[file_idx]
                    if num_trajectories > 1:
                        # Adjust alpha to distinguish trajectories from same file
                        alpha = 0.5 + 0.5 * (i / num_trajectories)
                    else:
                        alpha = 1.0

                    # Plot trajectory line
                    ax.plot(
                        traj_data["x"],
                        traj_data["y"],
                        traj_data["z"],
                        color=traj_color,
                        alpha=alpha,
                        linewidth=2,
                        label=label,
                    )
                    legend_entries.append(label)

                    # Plot waypoints if requested
                    if show_waypoints and "waypoints_x" in traj_data:
                        ax.scatter(
                            traj_data["waypoints_x"],
                            traj_data["waypoints_y"],
                            traj_data["waypoints_z"],
                            color=traj_color,
                            marker="o",
                            s=50,
                            alpha=0.7,
                        )

                        # Connect waypoints with dotted lines
                        ax.plot(
                            traj_data["waypoints_x"],
                            traj_data["waypoints_y"],
                            traj_data["waypoints_z"],
                            color=traj_color,
                            linestyle=":",
                            alpha=0.5,
                        )

        except Exception as e:
            print(f"Error processing {trajectory_file}: {e}")

    # Plot gates and obstacles from the provided data
    # Define gates
    gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Define obstacles
    obstacles = [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0.0, 1.5, 1.4], [-0.5, 0.5, 1.4]]

    # Plot gates as thick points with labels
    gate_x = [gate["pos"][0] for gate in gates]
    gate_y = [gate["pos"][1] for gate in gates]
    gate_z = [gate["pos"][2] for gate in gates]

    ax.scatter(
        gate_x,
        gate_y,
        gate_z,
        color="green",
        marker="s",  # square markers
        s=150,  # large size
        label="Gates",
        alpha=0.8,
    )

    # Add direction vectors for gates
    for i, gate in enumerate(gates):
        pos = gate["pos"]
        # Add gate number label
        ax.text(pos[0], pos[1], pos[2] + 0.1, f"Gate {i + 1}", color="green")

    # Plot obstacles as thick points with labels
    obstacles_x = [obs[0] for obs in obstacles]
    obstacles_y = [obs[1] for obs in obstacles]
    obstacles_z = [obs[2] for obs in obstacles]

    ax.scatter(
        obstacles_x,
        obstacles_y,
        obstacles_z,
        color="red",
        marker="o",  # circle markers
        s=150,  # large size
        label="Obstacles",
        alpha=0.8,
    )

    # Add obstacle labels
    for i, obs in enumerate(obstacles):
        ax.text(obs[0], obs[1], obs[2] + 0.1, f"Obs {i + 1}", color="red")

    # Add gate and obstacle entries to legend
    legend_entries = ["Gates", "Obstacles"] + legend_entries

    # Set plot labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # LIMIT Y-AXIS TO -4m
    ax.set_ylim(bottom=-4.0)

    # Add a smaller legend to avoid overcrowding
    if len(legend_entries) > 12:
        # If too many entries, limit the legend
        legend_entries = legend_entries[:12] + ["..."]
    ax.legend(legend_entries, loc="upper right", fontsize="small")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.5])  # Make z axis half scale for better visualization

    # Add grid
    ax.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_trajectory_comparison_2d(directory: str = "flight_logs"):
    """Create 3 individual 2D top-view plots comparing trajectories by index.

    Args:
        directory (str): Directory containing trajectory NPZ files
    """
    # Find trajectory files
    search_pattern = os.path.join(directory, "*_trajectories.npz")
    trajectory_files = glob.glob(search_pattern)

    if not trajectory_files:
        print(f"No trajectory files found in {directory}")
        return

    # Print available files for debugging
    print("Available trajectory files:")
    for file in trajectory_files:
        print(f"  - {os.path.basename(file)}")

    # Define gates for reference
    org_gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]
    gates = [
        {"pos": [0.57457256, -0.56958073, 0.5334348], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0139579, -0.9989303, 1.16429], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.1489047, 1.0194494, 0.49401802], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.4379613, -0.03196271, 1.0649462], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Define obstacles
    obstacles = [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0.0, 1.5, 1.4], [-0.5, 0.5, 1.4]]

    # Create figure with 3 subplots - add extra space at bottom for legend
    fig, axes = plt.subplots(1, 4, figsize=(18, 7))

    # Compare indices 0, 1, and 2
    comparison_indices = [-1, 0, 1, 2]

    # Store handles and labels for combined legend
    all_handles = []
    all_labels = []

    for plot_idx, traj_idx in enumerate(comparison_indices):
        ax = axes[plot_idx]

        # Find matching trajectory files - be more flexible with naming
        base_file = None
        racing_file = None

        for file_path in trajectory_files:
            filename = os.path.basename(file_path)
            # Look for base trajectory files
            if filename == f"{traj_idx}_trajectories.npz":
                base_file = file_path
            # Look for racing line files with various possible naming patterns
            elif filename in [
                f"Racing_Line_{traj_idx}_trajectories.npz",
                f"racing_line_{traj_idx}_trajectories.npz",
                f"RacingLine_{traj_idx}_trajectories.npz",
            ]:
                racing_file = file_path

        trajectories_found = []

        # Plot base trajectory
        if base_file:
            try:
                data = np.load(base_file, allow_pickle=True)
                num_trajectories = int(data["num_trajectories"])

                for i in range(num_trajectories):
                    traj_key = f"traj_{i}"
                    if traj_key in data:
                        traj_data = data[traj_key].item()

                        # Plot trajectory
                        line = ax.plot(
                            traj_data["x"],
                            traj_data["y"],
                            color="blue",
                            linewidth=2,
                            label=f"Base Trajectory",
                            alpha=0.8,
                        )

                        # Plot waypoints if available
                        if "waypoints_x" in traj_data:
                            points = ax.scatter(
                                traj_data["waypoints_x"],
                                traj_data["waypoints_y"],
                                color="blue",
                                marker="o",
                                s=30,
                                alpha=0.6,
                                label=f"Base Waypoints",
                            )

                        trajectories_found.append(f"Base_{traj_idx}")
                        break  # Only plot first trajectory from file

            except Exception as e:
                print(f"Error loading base trajectory {traj_idx}: {e}")

        # Plot racing line trajectory (if it exists)
        if racing_file:
            try:
                data = np.load(racing_file, allow_pickle=True)
                num_trajectories = int(data["num_trajectories"])

                for i in range(num_trajectories):
                    traj_key = f"traj_{i}"
                    if traj_key in data:
                        traj_data = data[traj_key].item()

                        # Plot trajectory
                        line = ax.plot(
                            traj_data["x"],
                            traj_data["y"],
                            color="red",
                            linewidth=2,
                            label=f"Racing Line",
                            alpha=0.8,
                            linestyle="--",
                        )

                        # Plot waypoints if available
                        if "waypoints_x" in traj_data:
                            points = ax.scatter(
                                traj_data["waypoints_x"],
                                traj_data["waypoints_y"],
                                color="red",
                                marker="s",
                                s=30,
                                alpha=0.6,
                                label=f"Racing Waypoints",
                            )

                        trajectories_found.append(f"Racing_{traj_idx}")
                        break  # Only plot first trajectory from file

            except Exception as e:
                print(f"Error loading racing line trajectory {traj_idx}: {e}")

        # Plot gates
        gate_x = [gate["pos"][0] for gate in gates]
        gate_y = [gate["pos"][1] for gate in gates]

        gates_scatter = ax.scatter(
            gate_x, gate_y, color="green", marker="s", s=100, label="Gates", alpha=0.8, zorder=5
        )

        # Plot original gates with dashed markers and no fill
        org_gate_x = [gate["pos"][0] for gate in org_gates]
        org_gate_y = [gate["pos"][1] for gate in org_gates]

        org_gates_scatter = ax.scatter(
            org_gate_x,
            org_gate_y,
            color="green",
            marker="s",
            s=100,
            label="Original Gates",
            alpha=0.8,
            zorder=5,
            facecolors="none",  # No fill
            edgecolors="green",  # Green border
            linestyles="dashed",  # Dashed border
            linewidths=2,  # Make border thicker (default is 1)
        )

        # Set plot properties
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if traj_idx == -1:
            ax.set_title("Base Trajectory")
        else:
            ax.set_title(f"Racing Line Trajectory {traj_idx}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Limit Y-axis to -2m
        ax.set_ylim(bottom=-1.5, top=1.5)
        ax.set_xlim(-0.7, 1.6)

        # NO INDIVIDUAL LEGENDS - we'll create one combined legend below
        # ax.legend(loc="upper right", fontsize="small")  # REMOVED

        # Collect legend handles and labels from first subplot only (to avoid duplicates)
        if plot_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

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
        unique_handles,
        unique_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(unique_labels),  # All items in one row
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add space at bottom for legend
    plt.savefig("flight_logs/figs/trajectory_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_trajectory_multiple(directory: str = "flight_logs"):
    """Create plots for trajectories grouped by second index with original gates only.

    Args:
        directory (str): Directory containing trajectory NPZ files
    """
    # Find trajectory files
    search_pattern = os.path.join(directory, "*_trajectories.npz")
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

    # Group files by second index (the one after the first underscore)
    trajectory_groups = {}
    for file_path in trajectory_files:
        filename = os.path.basename(file_path)
        # Extract second index from filename (e.g., "0_-1_trajectories.npz" -> "-1")
        if "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                second_index = parts[1]  # Take the second part
                if second_index not in trajectory_groups:
                    trajectory_groups[second_index] = []
                trajectory_groups[second_index].append(file_path)

    # Sort groups by index (handle negative numbers properly)
    sorted_groups = sorted(trajectory_groups.items(), key=lambda x: int(x[0]))

    # Define original gates for reference
    org_gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Determine subplot layout based on number of second indices
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
    fig.suptitle("Trajectory Comparison - Grouped by Second Index", fontsize=16)

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

    # Plot each trajectory group (grouped by second index)
    for group_idx, (second_index, group_files) in enumerate(sorted_groups):
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
        ax.set_title(f"Second Index {second_index}", fontsize=12)
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
    plt.savefig("flight_logs/figs/trajectory_multiple.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # You can specify a different directory if needed
    # plot_trajectories()

    # Also create the 2D comparison plots
    # plot_trajectory_comparison_2d()

    # create multiple trajectory plots
    plot_trajectory_multiple()
