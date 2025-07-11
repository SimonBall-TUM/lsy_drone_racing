"""This module provides functionality to visualize drone racing trajectories with gates and obstacles in a 3D plot.

It processes trajectory data from NPZ files and renders them using matplotlib.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation as R

def plot_trajectory_multiple(directory: str = "overlay_data"):
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

    # Find actual trajectory files
    actual_trajectory_files = glob.glob(os.path.join(directory, "actual_trajectory_*.npz"))
    print(f"Found {len(actual_trajectory_files)} actual trajectory files")

    # Define original gates for reference
    org_gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Define original obstacles
    org_obstacles = [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0.0, 1.5, 1.4], [-0.5, 0.5, 1.4]]

    # Load actual gate and obstacle positions from observations.npz
    actual_gate_pos = None
    actual_obs_pos = None
    observations_file = os.path.join(directory, "observations.npz")

    if os.path.exists(observations_file):
        try:
            obs_data = np.load(observations_file, allow_pickle=True)
            print(f"Loading observations from {observations_file}")
            print(f"Keys in observations file: {list(obs_data.keys())}")
            
            # Extract the observation dictionary
            obs_dict = obs_data["obs"].item()
            
            if 'gates_pos' in obs_dict:
                actual_gate_pos = obs_dict['gates_pos']
                print(f"Actual gate positions shape: {actual_gate_pos.shape}")
                print(f"Actual gate positions:\n{actual_gate_pos}")
            
            if 'obstacles_pos' in obs_dict:
                actual_obs_pos = obs_dict['obstacles_pos']
                print(f"Actual obstacle positions shape: {actual_obs_pos.shape}")
                print(f"Actual obstacle positions:\n{actual_obs_pos}")
                
        except Exception as e:
            print(f"Error loading observations.npz: {e}")
    else:
        print(f"Observations file not found: {observations_file}")

    # Create single plot showing all trajectories
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle("All Trajectories Overlay", fontsize=16)

    # Define colors for different trajectory types
    trajectory_colors = {
        -1: 'blue',    # Initial trajectory
        0: 'orange',   # Gate 0 replan
        1: 'green',    # Gate 1 replan
        2: 'red',      # Gate 2 replan
        3: 'purple'    # Gate 3 replan
    }

    trajectory_labels = {
        -1: "Initial Trajectory",
        0: "Gate 0 Replan",
        1: "Gate 1 Replan", 
        2: "Gate 2 Replan",
        3: "Gate 3 Replan"
    }

    # Plot planned trajectories from all time groups
    for time_key, group_files in complete_trajectory_groups.items():
        for file_path in group_files:
            filename = os.path.basename(file_path)
            if "_" in filename:
                try:
                    gate_index = int(filename.split("_")[0])
                    
                    # Load and plot trajectory
                    data = np.load(file_path, allow_pickle=True)
                    num_trajectories = int(data["num_trajectories"])
                    
                    for i in range(num_trajectories):
                        traj_key = f"traj_{i}"
                        if traj_key in data:
                            traj_data = data[traj_key].item()
                            
                            # Calculate trajectory length
                            x_coords = traj_data["x"]
                            y_coords = traj_data["y"]
                            trajectory_length = sum(np.sqrt((x_coords[j+1] - x_coords[j])**2 + 
                                                           (y_coords[j+1] - y_coords[j])**2) 
                                                    for j in range(len(x_coords)-1))
                            
                            print(f"Planned trajectory {gate_index} (time {time_key}): length = {trajectory_length:.3f}m")
                            
                            # Plot trajectory
                            ax.plot(
                                x_coords,
                                y_coords,
                                color=trajectory_colors[gate_index],
                                linewidth=2,
                                alpha=0.6,
                                label=trajectory_labels[gate_index] if time_key == list(complete_trajectory_groups.keys())[0] else ""
                            )
                            break  # Only plot first trajectory from file
                            
                except (ValueError, Exception) as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    # Plot actual trajectories
    for actual_file in actual_trajectory_files:
        try:
            data = np.load(actual_file, allow_pickle=True)
            print(f"Loading actual trajectory from {actual_file}")
            print(f"Keys in file: {list(data.keys())}")
            
            # Extract actual trajectory data
            if 'x' in data and 'y' in data:
                x_coords = data['x']
                y_coords = data['y']
            elif 'trajectory' in data:
                traj_data = data['trajectory']
                x_coords = traj_data[:, 0]  # Assuming first column is x
                y_coords = traj_data[:, 1]  # Assuming second column is y
            elif 'arr_0' in data:
                # Handle arr_0 format
                traj_data = data['arr_0']
                print(f"Trajectory shape: {traj_data.shape}")
                x_coords = traj_data[:, 0]  # First column is x
                y_coords = traj_data[:, 1]  # Second column is y
            else:
                print(f"Unknown format in {actual_file}, available keys: {list(data.keys())}")
                continue
                
            # Calculate trajectory length
            trajectory_length = sum(np.sqrt((x_coords[j+1] - x_coords[j])**2 + 
                                        (y_coords[j+1] - y_coords[j])**2) 
                                    for j in range(len(x_coords)-1))
            
            filename = os.path.basename(actual_file)
            print(f"Actual trajectory {filename}: length = {trajectory_length:.3f}m")
            
            # Plot actual trajectory
            ax.plot(
                x_coords,
                y_coords,
                color='black',
                linewidth=3,
                alpha=0.8,
                linestyle='--',
                label='Actual Trajectory'
            )
            
        except Exception as e:
            print(f"Error loading actual trajectory {actual_file}: {e}")
            continue

    # Plot original gates
    org_gate_x = [gate["pos"][0] for gate in org_gates]
    org_gate_y = [gate["pos"][1] for gate in org_gates]

    ax.scatter(
        org_gate_x,
        org_gate_y,
        color="lightgreen",
        marker="s",
        s=120,
        label="Original Gates",
        alpha=0.8,
        zorder=4,
        edgecolors='green',
        linewidth=2
    )

    # Plot actual gates if available
    if actual_gate_pos is not None:
        ax.scatter(
            actual_gate_pos[:, 0],  # X coordinates
            actual_gate_pos[:, 1],  # Y coordinates
            color="green",
            marker="s",
            s=150,
            label="Actual Gates",
            alpha=0.8,
            zorder=5
        )

    # Plot original obstacles
    obstacle_x = [obs[0] for obs in org_obstacles]
    obstacle_y = [obs[1] for obs in org_obstacles]

    ax.scatter(
        obstacle_x,
        obstacle_y,
        color="lightcoral",
        marker="o",
        s=120,
        label="Original Obstacles",
        zorder=4,
        edgecolors='red',
        linewidth=2
    )

    # Plot actual obstacles if available
    if actual_obs_pos is not None:
        ax.scatter(
            actual_obs_pos[:, 0],  # X coordinates
            actual_obs_pos[:, 1],  # Y coordinates
            color="red",
            marker="o",
            s=150,
            label="Actual Obstacles",
            alpha=0.8,
            zorder=5
        )

    # Set plot properties
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Set axis limits
    ax.set_ylim(bottom=-1.5, top=1.5)
    ax.set_xlim(-0.7, 1.6)

    # Add legend
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # create multiple trajectory plots
    plot_trajectory_multiple()