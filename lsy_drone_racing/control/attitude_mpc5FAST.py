"""This module implements an MPC using attitude control for a quadrotor for drone racing.

The controller architecture is organized into several key components:
1. Gate handling - detecting and navigating through racing gates
2. Obstacle avoidance - using direct MPC constraints
3. Trajectory planning - computing optimal paths through gates
4. Control execution - implementing the MPC controller
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # define state and input vector
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define position vector for constraints
    pos = vertcat(px, py, pz)

    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs

    # Add position to model for constraint functions
    model.pos = pos

    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    # Initialize constraint expressions for obstacle avoidance
    # Will be populated dynamically during control computation
    ocp.model.con_h_expr = MX.sym("con_h_expr", 0, 1)

    # Setting up nonlinear inequality constraints
    # These will be populated during runtime with obstacle avoidance constraints
    ocp.constraints.lh = np.array([])  # Lower bounds
    ocp.constraints.uh = np.array([])  # Upper bounds

    # Setup to allow for dynamic updating of constraints
    ocp.constraints.constr_type = "BGH"
    ocp.constraints.constr_type_e = "BGH"

    ## Set Cost
    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Define weight matrices for the cost function
    pos = 59.0
    vel = 0.01  # 0.01
    rpy = 0.1
    f_col = 0.01
    rpy_cmd = 0.01

    Q = np.diag(
        [
            pos,
            pos,
            pos,  # Position
            vel,
            vel,
            vel,  # Velocity
            rpy,
            rpy,
            rpy,  # rpy - roll, pitch, yaw
            f_col,
            f_col,  # f_collective, f_collective_cmd
            rpy_cmd,
            rpy_cmd,
            rpy_cmd,  # rpy_cmd
        ]
    )

    # rate of change
    r_param = 0.6  # 0.88
    R = np.diag([r_param, r_param, r_param, r_param])  # Input cost

    Q_e = Q.copy()

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)  # Only select position states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states
    ocp.cost.Vx_e = Vx_e

    # Set initial references
    ocp.cost.yref = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp


class MPController(Controller):
    """MPC using the collective thrust and attitude interface for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller."""
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self.start_pos = obs["pos"]
        self.config = config

        # Configuration parameters
        self.replanning_frequency = 2
        self.last_replanning_tick = 0

        # Receding horizon parameters
        self.horizon_distance = 1.5  # Only replan within 3 meters ahead
        self.horizon_time = 2.5  # Or 2.5 seconds ahead
        self.min_horizon_points = 3  # Minimum points to maintain in horizon

        # Approach parameters
        # self.approach_dist = [0.2, 0.3, 0.5, 0.2]
        self.approach_dist = [0.2, 0.6, 0.3, 0.01]
        self.exit_dist = [1.0, 0.3, 0.05, 1]
        self.default_approach_dist = 0.2
        self.default_exit_dist = 0.5

        # Initialize target gate tracking
        self.current_target_gate_idx = 0

        # Add height offset parameters
        self.approach_height_offset = [
            0.0,
            0.1,
            -0.1,
            0.0,
        ]  # Height offset for each gate's approach
        self.exit_height_offset = [0.0, 0.1, 0.05, 0.0]  # Height offset for each gate's exit
        self.default_approach_height_offset = 0.0
        self.default_exit_height_offset = 0.0

        # Create MPC
        self.N = 60
        self.T_HORIZON = 1.5
        self.dt = self.T_HORIZON / self.N

        # Create the actual solver instance
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        # Controller state
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.finished = False

        # Track which gates have already been updated with observations
        self.updated_gates = set()

        # Add trajectory storage for plotting
        self.saved_trajectories = []
        self.trajectory_metadata = []

        # Store only the most recent trajectory
        self.current_trajectory = None  # Will store the most recent trajectory

        # Initialize trajectory
        self.waypoints = self._generate_waypoints(obs)
        self._generate_trajectory_from_waypoints(self.waypoints, 0)

    ###########################################
    # 1. TRAJECTORY GENERATION AND MANAGEMENT #
    ###########################################

    def _generate_trajectory_from_waypoints(
        self, waypoints: NDArray[np.floating], target_gate_idx: int
    ):
        """Generate a smooth trajectory from waypoints using cubic splines."""
        if len(waypoints) < 2:
            print("Warning: Not enough waypoints to generate trajectory")
            return

        # Convert waypoints to numpy array if it's not already one
        if not isinstance(waypoints, np.ndarray):
            waypoints = np.array(waypoints)

        # Generate cubic spline interpolation
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0], bc_type="natural")
        cs_y = CubicSpline(ts, waypoints[:, 1], bc_type="natural")
        cs_z = CubicSpline(ts, waypoints[:, 2], bc_type="natural")

        # Calculate trajectory duration based on path length
        total_distance = np.sum(np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1))
        avg_speed = 0.8  # m/s
        duration = max(8, total_distance / avg_speed)  # At least 8 seconds

        # Generate trajectory points with sufficient density
        min_points = max(self.freq * duration, self._tick + 3 * self.N)  # Ensure enough points
        ts = np.linspace(0, 1, int(min_points))
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        # Add terminal points for stability
        extra_points = 3 * self.N + 1
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * extra_points))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * extra_points))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * extra_points))

        # Save trajectory for later plotting
        trajectory = {
            "tick": self._tick,
            "waypoints": waypoints.copy(),
            "x": self.x_des.copy(),
            "y": self.y_des.copy(),
            "z": self.z_des.copy(),
            "timestamp": time.time() if hasattr(time, "time") else 0,
        }

        # Store only the most recent trajectory
        self.current_trajectory = trajectory

        # Save trajectory to file
        self.save_trajectories_to_file(f"flight_logs/{target_gate_idx}_trajectories.npz")

    def save_trajectories_to_file(self, filename: str = "trajectories.npz") -> bool:
        """Save the current trajectory to a file for later analysis."""
        try:
            # Check if we have a trajectory to save
            if self.current_trajectory is None:
                print("No trajectory to save")
                return False

            # Create data structure with only one trajectory
            waypoints = self.current_trajectory["waypoints"]

            # Handle different waypoint formats
            if waypoints is not None:
                # Check if waypoints is a dictionary (from horizon trajectory)
                if isinstance(waypoints, dict):
                    # Extract coordinate arrays from dictionary
                    if "x" in waypoints and "y" in waypoints and "z" in waypoints:
                        # Convert dictionary format to numpy array
                        x_coords = np.array(waypoints["x"])
                        y_coords = np.array(waypoints["y"])
                        z_coords = np.array(waypoints["z"])

                        # Take a subset for waypoints (every 10th point for visualization)
                        step = max(1, len(x_coords) // 10)
                        waypoints_x = x_coords[::step]
                        waypoints_y = y_coords[::step]
                        waypoints_z = z_coords[::step]
                    else:
                        # Empty arrays if dictionary doesn't have expected keys
                        waypoints_x = np.array([])
                        waypoints_y = np.array([])
                        waypoints_z = np.array([])
                elif isinstance(waypoints, np.ndarray):
                    # Check if it's a proper 2D array
                    if waypoints.ndim == 2 and waypoints.shape[1] >= 3:
                        waypoints_x = waypoints[:, 0]
                        waypoints_y = waypoints[:, 1]
                        waypoints_z = waypoints[:, 2]
                    elif waypoints.ndim == 1 and len(waypoints) >= 3:
                        # Single waypoint case
                        waypoints_x = np.array([waypoints[0]])
                        waypoints_y = np.array([waypoints[1]])
                        waypoints_z = np.array([waypoints[2]])
                    else:
                        # Invalid array format
                        print(f"Warning: Invalid waypoints array shape: {waypoints.shape}")
                        waypoints_x = np.array([])
                        waypoints_y = np.array([])
                        waypoints_z = np.array([])
                else:
                    # Try to convert to numpy array
                    try:
                        waypoints_array = np.array(waypoints)
                        if waypoints_array.ndim == 2 and waypoints_array.shape[1] >= 3:
                            waypoints_x = waypoints_array[:, 0]
                            waypoints_y = waypoints_array[:, 1]
                            waypoints_z = waypoints_array[:, 2]
                        else:
                            waypoints_x = np.array([])
                            waypoints_y = np.array([])
                            waypoints_z = np.array([])
                    except Exception:
                        waypoints_x = np.array([])
                        waypoints_y = np.array([])
                        waypoints_z = np.array([])
            else:
                # Empty arrays if no waypoints
                waypoints_x = np.array([])
                waypoints_y = np.array([])
                waypoints_z = np.array([])

            data = {
                "traj_0": {
                    "tick": self.current_trajectory["tick"],
                    "x": self.current_trajectory["x"],
                    "y": self.current_trajectory["y"],
                    "z": self.current_trajectory["z"],
                    "waypoints_x": waypoints_x,
                    "waypoints_y": waypoints_y,
                    "waypoints_z": waypoints_z,
                }
            }

            # Add metadata about number of trajectories
            data["num_trajectories"] = 1  # Always one trajectory

            # Ensure the directory exists
            import os

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            np.savez(filename, **data)
            return True
        except Exception as e:
            print(f"Error saving trajectory: {e}")
            print(f"Waypoints type: {type(self.current_trajectory['waypoints'])}")
            print(f"Waypoints content: {self.current_trajectory['waypoints']}")
            return False

    def _update_trajectory(
        self,
        new_waypoints: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]] | None = None,
    ):
        """Update trajectory using receding horizon approach."""
        if len(new_waypoints) < 2:
            print("Warning: Not enough waypoints to update trajectory")
            return

        current_pos = obs["pos"] if obs else np.array([0, 0, 0])

        # Apply receding horizon planning
        if hasattr(self, "x_des") and len(self.x_des) > 0:
            # Only replan within the horizon
            horizon_waypoints = self._get_horizon_waypoints(current_pos, new_waypoints)
            self._replan_horizon_trajectory(horizon_waypoints, obs)
        else:
            # First trajectory generation - plan full path
            self._generate_trajectory_from_waypoints(
                new_waypoints, obs["target_gate"] if obs and "target_gate" in obs else 0
            )

    def _get_horizon_waypoints(
        self, current_pos: NDArray[np.floating], new_waypoints: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Extract waypoints within the receding horizon, filtering based on drone-gate proximity."""
        horizon_waypoints = [current_pos]  # Start with current position

        for waypoint in new_waypoints:
            distance_to_waypoint = np.linalg.norm(waypoint - current_pos)

            # Check if waypoint is within horizon distance
            if distance_to_waypoint <= self.horizon_distance:
                horizon_waypoints.append(waypoint)
            elif len(horizon_waypoints) >= self.min_horizon_points:
                # Stop adding waypoints once we have enough in horizon
                break

        # Ensure we have at least minimum points by adding the next few waypoints
        remaining_waypoints = new_waypoints[len(horizon_waypoints) - 1 :]
        needed_points = max(0, self.min_horizon_points - len(horizon_waypoints))
        if needed_points > 0 and len(remaining_waypoints) > 0:
            horizon_waypoints.extend(remaining_waypoints[:needed_points])

        return np.array(horizon_waypoints)

    def _should_add_waypoint_based_on_gate_proximity(
        self, current_pos: NDArray[np.floating], gate_info: dict
    ) -> bool:
        """Determine if a waypoint should be added based on drone proximity to gate.

        Args:
            current_pos: Current drone position
            waypoint: Candidate waypoint to add
            gate_info: Information about the current target gate

        Returns:
            bool: Whether to add this waypoint
        """
        gate_center = gate_info["center"]
        gate_normal = gate_info["normal"]

        # Calculate approach point for this gate - use gate-specific distance
        approach_dist = self.default_approach_dist

        # Get the correct gate index - this is the key fix
        # if hasattr(self, 'current_target_gate_idx') and self.current_target_gate_idx is not None:
        #     gate_idx = self.current_target_gate_idx
        #     if gate_idx < len(self.approach_dist):
        #         approach_dist = self.approach_dist[gate_idx]

        approach_point = gate_center + gate_normal * approach_dist

        # Calculate distances
        drone_to_gate = np.linalg.norm(current_pos - gate_center)
        approach_to_gate = np.linalg.norm(approach_point - gate_center)

        # Check if drone is closer to gate than the approach point
        if drone_to_gate < approach_to_gate:
            print(
                f"Gate {getattr(self, 'current_target_gate_idx', 'unknown')}: Drone closer to gate ({drone_to_gate:.2f}m) than approach point ({approach_to_gate:.2f}m, dist={approach_dist})"
            )
            return False
        else:
            # Drone is not yet close to gate, use normal horizon filtering
            return True

    def _replan_horizon_trajectory(
        self, horizon_waypoints: NDArray[np.floating], obs: dict[str, NDArray[np.floating]]
    ):
        """Replan only the trajectory within the receding horizon."""
        if len(horizon_waypoints) < 2:
            return

        # Store current target gate for proximity checks
        if "target_gate" in obs:
            self.current_target_gate_idx = int(obs["target_gate"])

        # Find the current position in the existing trajectory
        current_trajectory_idx = min(self._tick, len(self.x_des) - 1)

        # Generate new trajectory segment for the horizon
        horizon_trajectory = self._generate_horizon_trajectory(horizon_waypoints)

        # Splice the new horizon trajectory into the existing trajectory
        self._splice_horizon_trajectory(horizon_trajectory, current_trajectory_idx)

    def _generate_horizon_trajectory(
        self,
        horizon_waypoints: NDArray[np.floating],
        current_velocity: NDArray[np.floating] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Generate trajectory that maintains velocity continuity."""
        if len(horizon_waypoints) < 2:
            return None

        # Generate splines for horizon waypoints
        ts = np.linspace(0, 1, len(horizon_waypoints))

        # Use clamped boundary conditions to maintain velocity continuity
        if current_velocity is not None:
            # Set initial velocity to match current drone velocity
            bc_type = ((1, current_velocity[0]), (1, 0))  # (derivative_order, value)
            cs_x = CubicSpline(ts, horizon_waypoints[:, 0], bc_type=bc_type)
            cs_y = CubicSpline(
                ts, horizon_waypoints[:, 1], bc_type=((1, current_velocity[1]), (1, 0))
            )
            cs_z = CubicSpline(
                ts, horizon_waypoints[:, 2], bc_type=((1, current_velocity[2]), (1, 0))
            )
        else:
            cs_x = CubicSpline(ts, horizon_waypoints[:, 0], bc_type="natural")
            cs_y = CubicSpline(ts, horizon_waypoints[:, 1], bc_type="natural")
            cs_z = CubicSpline(ts, horizon_waypoints[:, 2], bc_type="natural")

        # Calculate trajectory points for horizon
        horizon_distance = np.sum(
            np.linalg.norm(horizon_waypoints[1:] - horizon_waypoints[:-1], axis=1)
        )
        avg_speed = 0.8
        horizon_duration = max(self.horizon_time, horizon_distance / avg_speed)

        num_points = int(horizon_duration * self.freq)
        ts_traj = np.linspace(0, 1, num_points)

        horizon_x = cs_x(ts_traj)
        horizon_y = cs_y(ts_traj)
        horizon_z = cs_z(ts_traj)

        return {"x": horizon_x, "y": horizon_y, "z": horizon_z}

    def _splice_horizon_trajectory(
        self, horizon_trajectory: dict[str, np.ndarray], current_idx: int
    ):
        """Add smooth transition to prevent discontinuities."""
        if horizon_trajectory is None:
            return

        # Create smooth transition over several points
        transition_length = 10  # Number of points for smooth transition
        splice_start = current_idx + transition_length

        # Generate smooth transition from current trajectory to new one
        if splice_start <= len(self.x_des):
            current_point = np.array(
                [self.x_des[current_idx], self.y_des[current_idx], self.z_des[current_idx]]
            )
            new_start_point = np.array(
                [horizon_trajectory["x"][0], horizon_trajectory["y"][0], horizon_trajectory["z"][0]]
            )

            # Linear transition
            for i in range(transition_length):
                alpha = i / transition_length
                transition_point = current_point * (1 - alpha) + new_start_point * alpha

                if current_idx + i < len(self.x_des):
                    self.x_des[current_idx + i] = transition_point[0]
                    self.y_des[current_idx + i] = transition_point[1]
                    self.z_des[current_idx + i] = transition_point[2]

    ##############################
    # 2. WAYPOINT GENERATION     #
    ##############################

    def loop_path_gen(self, gates: list[dict], obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Generate a loop path through all gates with corridor constraints and height offsets."""
        waypoints = []
        current_pos = obs["pos"] if obs else np.array([0, 0, 0])

        # Create waypoints for each gate
        for gate_idx, gate in enumerate(gates):
            gate_info = self._get_gate_info(gate)
            original_gate_idx = gate_idx + (4 - len(gates))

            # Set current target gate index for proximity checking
            self.current_target_gate_idx = original_gate_idx

            # Use individual distances if available, otherwise fall back to defaults
            if original_gate_idx < len(self.approach_dist):
                approach_dist = self.approach_dist[original_gate_idx]
            else:
                approach_dist = self.default_approach_dist

            if original_gate_idx < len(self.exit_dist):
                exit_dist = self.exit_dist[original_gate_idx]
            else:
                exit_dist = self.default_exit_dist

            # Get height offsets for this gate
            if original_gate_idx < len(self.approach_height_offset):
                approach_z_offset = self.approach_height_offset[original_gate_idx]
            else:
                approach_z_offset = self.default_approach_height_offset

            if original_gate_idx < len(self.exit_height_offset):
                exit_z_offset = self.exit_height_offset[original_gate_idx]
            else:
                exit_z_offset = self.default_exit_height_offset

            print(f"######### Replanning with gate {gate_info['center']} #########")
            # Create approach, center, and exit points with individual distances and height offsets
            approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
            approach_point[2] += approach_z_offset  # Add height offset

            exit_point = gate_info["center"] - gate_info["normal"] * exit_dist
            exit_point[2] += exit_z_offset  # Add height offset

            # Use proximity function to check if approach point should be added
            # If drone is already close to gate, skip approach point
            if self._should_add_waypoint_based_on_gate_proximity(current_pos, gate_info):
                waypoints.append(approach_point)

            # Always add gate center point
            waypoints.append(gate_info["center"])

            # Add exit point
            waypoints.append(exit_point)

            # Add corridor waypoints to next gate if available
            if gate_idx < len(gates) - 1:
                next_gate = gates[gate_idx + 1]
                next_gate_info = self._get_gate_info(next_gate)

                # Get the approach point of the next gate with height offset
                next_approach_dist = (
                    self.approach_dist[original_gate_idx + 1]
                    if original_gate_idx + 1 < len(self.approach_dist)
                    else self.default_approach_dist
                )
                next_approach_z_offset = (
                    self.approach_height_offset[original_gate_idx + 1]
                    if original_gate_idx + 1 < len(self.approach_height_offset)
                    else self.default_approach_height_offset
                )

                next_approach = (
                    next_gate_info["center"] + next_gate_info["normal"] * next_approach_dist
                )
                next_approach[2] += next_approach_z_offset  # Add height offset

                # Add corridor points between current exit and next approach (prevents shortcuts)
                num_corridor_points = 2  # Number of points in the corridor
                for i in range(1, num_corridor_points):
                    t = i / num_corridor_points
                    corridor_point = exit_point * (1 - t) + next_approach * t

                    # Optional: Add small random offset perpendicular to corridor to smooth path
                    if i > 1 and i < num_corridor_points - 1:
                        # Calculate corridor direction
                        corridor_dir = next_approach - exit_point
                        corridor_dir = corridor_dir / np.linalg.norm(corridor_dir)

                        # Create perpendicular vector for slight variation
                        perp = np.array([-corridor_dir[1], corridor_dir[0], 0]) * 0.05

                        # Add small random offset perpendicular to path
                        offset = perp * np.sin(i * np.pi / num_corridor_points) * 0.1
                        corridor_point += offset

                    waypoints.append(corridor_point)

        print(f"Generated {len(waypoints)} waypoints with proximity-based filtering")
        return np.array(waypoints)

    def _generate_waypoints(
        self, obs: dict[str, NDArray[np.floating]], start_gate_idx: int = 0
    ) -> np.ndarray:
        """Generate waypoints that navigate through all gates."""
        # Start with the current position as a single-element array
        start_point = np.array(obs["pos"]).reshape(1, 3)

        # Generate gate waypoints
        gates = self._get_gates_with_observations(obs, start_gate_idx)
        gate_waypoints = self.loop_path_gen(gates, obs)

        # Concatenate the arrays properly
        return np.concatenate((start_point, gate_waypoints))

    ##############################
    # 3. GATE HANDLING UTILITIES #
    ##############################

    def _get_gate_info(self, gate: dict) -> dict:
        """Extract gate information including position, orientation, and dimensions."""
        try:
            # Ensure we're working with a valid dictionary
            if not isinstance(gate, dict):
                print(f"Warning: Gate is not a dictionary: {gate}")
                gate = {"pos": [0.0, 0.0, 1.0]}

            # Extract position
            if "pos" not in gate:
                print(f"Warning: Gate missing position key: {gate}")
                pos = np.array([0.0, 0.0, 1.0])
            elif gate["pos"] is None:
                print(f"Warning: Gate position is None: {gate}")
                pos = np.array([0.0, 0.0, 1.0])
            else:
                # Handle different position formats
                raw_pos = gate["pos"]
                if isinstance(raw_pos, dict) and "pos" in raw_pos:
                    # Handle nested dictionary case
                    print(f"Converting nested gate dictionary: {raw_pos}")
                    raw_pos = raw_pos["pos"]

                if isinstance(raw_pos, (list, tuple, np.ndarray)) and len(raw_pos) == 3:
                    pos = np.array(raw_pos)
                elif np.isscalar(raw_pos):
                    print(f"Warning: Gate position is scalar {raw_pos}, using as z-coordinate")
                    pos = np.array([0.0, 0.0, float(raw_pos)])
                else:
                    print(f"Warning: Invalid position format: {type(raw_pos)}, value: {raw_pos}")
                    pos = np.array([0.0, 0.0, 1.0])

            # Extract gate dimensions
            gate_height = gate.get("height", 0.5)
            gate_width = gate.get("width", 0.5)

            # Determine target height based on gate dimensions
            is_short_gate = gate_height <= 0.6
            target_height = pos[2] if is_short_gate else pos[2] + 0.2

            # Get orientation
            if "rpy" in gate and gate["rpy"] is not None:
                rpy = np.array(gate["rpy"])
                rotation = R.from_euler("xyz", rpy)
            else:
                rotation = R.from_euler("xyz", [0, 0, 0])
                rpy = np.zeros(3)

            # Calculate normal vector and adjusted gate center
            temp_normal = rotation.apply([1.0, 0.0, 0.0])
            normal = np.array([temp_normal[1], -temp_normal[0], temp_normal[2]])
            y_dir = rotation.apply([0.0, 1.0, 0.0])
            z_dir = rotation.apply([0.0, 0.0, 1.0])

            # Adjust center point height
            gate_center = pos.copy()
            gate_center[2] = target_height

            return {
                "pos": pos,
                "center": gate_center,
                "normal": normal,
                "y_dir": y_dir,
                "z_dir": z_dir,
                "rotation": rotation,
                "height": gate_height,
                "width": gate_width,
                "is_short": is_short_gate,
                "rpy": rpy,
            }
        except Exception as e:
            print(f"Error processing gate: {e}")
            print(f"Gate data: {gate}")
            # Return safe default values
            default_pos = np.array([0.0, 0.0, 1.0])
            return {
                "pos": default_pos,
                "center": default_pos,
                "normal": np.array([1.0, 0.0, 0.0]),
                "y_dir": np.array([0.0, 1.0, 0.0]),
                "z_dir": np.array([0.0, 0.0, 1.0]),
                "rotation": R.from_euler("xyz", [0, 0, 0]),
                "height": 0.5,
                "width": 0.5,
                "is_short": True,
                "rpy": np.zeros(3),
            }

    def _get_gates_with_observations(
        self, obs: dict[str, NDArray[np.floating]], start_idx: int
    ) -> list[dict]:
        """Get gate info using observed positions when available."""
        gates = []
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations:
            for i in range(start_idx, len(self.config.env.track["gates"])):
                gates.append(self._get_gate_with_observation(obs, i))

        return gates

    def _get_gate_with_observation(
        self, obs: dict[str, NDArray[np.floating]], gate_idx: int
    ) -> dict:
        """Get a single gate with observed position if available."""
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations and gate_idx < len(obs["gates_pos"]):
            # Use observed position but keep original orientation
            gate_info = self.config.env.track["gates"][gate_idx].copy()
            gate_info["pos"] = obs["gates_pos"][gate_idx]
            return gate_info
        else:
            # Fall back to static configuration
            return self.config.env.track["gates"][gate_idx]

    ################################
    # 4. CONTROLLER INTERFACE      #
    ################################

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Execute MPC control with receding horizon replanning."""
        # Check for replanning based on gate observations
        if self._tick - self.last_replanning_tick >= self.replanning_frequency:
            self.last_replanning_tick = self._tick

            # Check if we have new gate observations to update trajectory
            if "gates_pos" in obs and obs["gates_pos"] is not None and "target_gate" in obs:
                target_gate_idx = int(obs["target_gate"])

                should_replan = False

                # Check if the target gate hasn't been updated yet
                if (
                    target_gate_idx < len(self.config.env.track["gates"])
                    and target_gate_idx < len(obs["gates_pos"])
                    and target_gate_idx not in self.updated_gates
                ):
                    config_pos = np.array(self.config.env.track["gates"][target_gate_idx]["pos"])
                    observed_pos = np.array(obs["gates_pos"][target_gate_idx])

                    if np.linalg.norm(config_pos - observed_pos) > 0.05:
                        print(f"Gate {target_gate_idx} position updated, horizon replanning")
                        should_replan = True
                        self.updated_gates.add(target_gate_idx)

                if should_replan:
                    # Generate waypoints for the updated gate
                    new_waypoints = self._generate_waypoints(obs, target_gate_idx)
                    self._update_trajectory(new_waypoints, obs)

        # Execute MPC control
        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        # Get current state and setup
        q = obs["quat"]
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False)
        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )

        # Set current state for MPC
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # Get trajectory size
        traj_len = len(self.x_des)
        i = min(i, traj_len - 1)

        # Set reference trajectory
        for j in range(self.N):
            # Get reference point from trajectory
            if i + j < traj_len:
                x_ref = self.x_des[i + j]
                y_ref = self.y_des[i + j]
                z_ref = self.z_des[i + j]
            else:
                x_ref = self.x_des[-1]
                y_ref = self.y_des[-1]
                z_ref = self.z_des[-1]

            # Set reference for this step
            yref = np.array(
                [
                    x_ref,
                    y_ref,
                    z_ref,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.35,
                    0.35,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", yref)

        # Set terminal reference
        terminal_idx = min(i + self.N, traj_len - 1)
        yref_N = np.array(
            [
                self.x_des[terminal_idx],
                self.y_des[terminal_idx],
                self.z_des[terminal_idx],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.acados_ocp_solver.set(self.N, "yref", yref_N)

        # Solve MPC and get control
        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")

        # Smooth control updates
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        # Return thrust and attitude commands
        cmd = x1[10:14]
        return cmd

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        return self.finished

    def episode_callback(self):
        """Reset controller state for new episode."""
        self._tick = 0
        self.finished = False
        self.updated_gates = set()

    def get_predicted_trajectory(self) -> np.ndarray:
        """Return the MPC's predicted trajectory over the horizon."""
        pred_traj = []
        for i in range(self.N):
            x = self.acados_ocp_solver.get(i, "x")
            pred_traj.append(x[:3])
        return np.array(pred_traj)
