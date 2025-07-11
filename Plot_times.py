import numpy as np
import matplotlib.pyplot as plt

def load_trajectory_times():
    """
    Load trajectory times from NPZ file.
    
    Args:
        filename: Path to NPZ file containing trajectory time data
    
    Returns:
        Tuple of (real_times, simu_times, static_times, obsfree_times)
        Each is a list of tuples (success: bool, time: float)
    """
    simu_times = [(True, time) for time in range(4,10)]
    static_times = [(True, time) for time in range(4,10)]
    obsfree_times = [(True, time) for time in range(4,10)]

    for i, arg in enumerate(["simu_times100", "static_times100", "obstacle_free"]):
        data = np.load(f"flight_log/{arg}.npz", allow_pickle=True)
        result = list(zip(data['success'], data['times']))

        if i == 0:
            simu_times = result
        elif i == 1:
            static_times = result
        elif i == 2:
            obsfree_times = result
    
    return simu_times, static_times, obsfree_times

def plot_trajectory_times(real_times, simu_times, static_times, obsfree_times):
    """
    Plot trajectory completion times with success/failure coloring.
    
    Args:
        real_times: List of tuples (success: bool, time: float)
        simu_times: List of tuples (success: bool, time: float)
        static_times: List of tuples (success: bool, time: float)
        obsfree_times: List of tuples (success: bool, time: float)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Categories and their data
    categories = ['Real Flights', 'Simulation', 'Static Planning', 'Obstacle-Free']
    data_sets = [real_times, simu_times, static_times, obsfree_times]
    
    # Plot each category
    for i, (category, data) in enumerate(zip(categories, data_sets)):
        # Separate successful and failed attempts
        successful_times = [time for success, time in data if success]
        failed_times = [time for success, time in data if not success]
        
        # Create x positions with some jitter for better visibility
        x_pos = i + 1
        successful_x = [x_pos + np.random.normal(0, 0.05) for _ in successful_times]
        failed_x = [x_pos + np.random.normal(0, 0.05) for _ in failed_times]
        
        # Plot successful attempts as green dots
        if successful_times:
            ax.scatter(successful_x, successful_times, color='green', alpha=0.7, s=60)
        
        # Plot failed attempts as red dots
        if failed_times:
            ax.scatter(failed_x, failed_times, color='red', alpha=0.7, s=60)
    
    # Add legend entries for both classes
    ax.scatter([], [], color='green', alpha=0.7, s=60, label='Successful')
    ax.scatter([], [], color='red', alpha=0.7, s=60, label='Failed')
    
    # Customize the plot
    ax.set_xlabel('Flight Method', fontsize=12)
    ax.set_ylabel('Flight Time (seconds)', fontsize=12)
    ax.set_title('Flight Time Comparison Across Different Methods', fontsize=14)
    ax.set_xticks(range(1, len(categories) + 1))
    
    # Calculate success rates and create labels with success rate
    category_labels = []
    for category, data in zip(categories, data_sets):
        successful_count = sum(1 for success, time in data if success)
        total_count = len(data)
        success_rate = (successful_count / total_count * 100) if total_count > 0 else 0
        category_labels.append(f"{category}\n({success_rate:.1f}% success)")
    
    ax.set_xticklabels(category_labels)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Convert real_times to list of tuples (all successful)
real_times = [(True, time) for time in [7.012, 7.002, 7.16, 7.359, 7.345, 7.031, 7.117, 7.013, 6.899, 6.817, 6.926, 6.97, 6.941, 7.072, 7.067, 6.928]]

# Example of how to use load function:
simu_times, static_times, obsfree_times = load_trajectory_times()
print("obsfree: ", obsfree_times)
plot_trajectory_times(real_times, simu_times, static_times, obsfree_times)