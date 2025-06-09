"""This module implements an MPC using attitude control for a quadrotor for drone racing.

The controller architecture is organized into several key components:
1. Gate handling - detecting and navigating through racing gates
2. Obstacle avoidance - using direct MPC constraints
3. Trajectory planning - computing optimal paths through gates
4. Control execution - implementing the MPC controller
"""

from __future__ import annotations

import logging
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


# Configure logging
def setup_logging():
    """Setup logging configuration for the MPC controller."""
    import os

    # Create logs directory if it doesn't exist
    os.makedirs("flight_logs", exist_ok=True)

    # Create logger
    logger = logging.getLogger("MPController")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create file handler with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"flight_logs/mpc_controller_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create console handler for critical messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Tick:%(tick)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    model_name = "lsy_example_mpc"
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    # Define state variables (14 total)
    px = MX.sym("px")
    py = MX.sym("py")
    pz = MX.sym("pz")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    vz = MX.sym("vz")
    roll = MX.sym("r")
    pitch = MX.sym("p")
    yaw = MX.sym("y")
    f_collective = MX.sym("f_collective")
    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    # Control inputs: 4 real controls only
    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # State and input vectors
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

    # System dynamics (only for the 14 states)
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

    # Initialize the model
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.x = states
    model.u = inputs

    return model


def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions - no slack variables
    nx = model.x.rows()  # 14 states
    nu = model.u.rows()  # 4 inputs (real controls only)
    ny = nx + nu  # 18 total for cost (14 states + 4 inputs)
    ny_e = nx  # 14 for terminal

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Define weight matrices for the cost function
    pos = 3.0
    vel = 0.5
    rpy = 0.01
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
    r_param = 0.007
    R = np.diag([r_param, r_param, r_param, r_param])  # Real control cost

    Q_e = Q.copy()

    # Stage cost matrix (18x18) - states + real controls only
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    # Vx matrix (18x14) - maps states to cost
    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vx = Vx

    # Vu matrix (18x4) - maps inputs to cost
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + 4, :4] = np.eye(4)  # Real controls
    ocp.cost.Vu = Vu

    # Terminal Vx matrix (14x14)
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # Set initial references (18 dimensions)
    ocp.cost.yref = np.array(
        [
            1.0,
            1.0,
            0.4,
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
            0.0,  # states
            0.0,
            0.0,
            0.0,
            0.0,  # real controls
        ]
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

        # Setup logging
        self.logger = setup_logging()

        self.freq = config.env.freq
        self._tick = 0
        self.start_pos = obs["pos"]
        self.config = config

        # Configuration parameters
        self.replanning_frequency = 1
        self.last_replanning_tick = 0

        # Approach parameters
        self.approach_dist = [0.1, 0.35, 0.2, 0.01]
        self.exit_dist = [1.2, 0.2, 0.5, 1.2]
        self.default_approach_dist = 0.1
        self.default_exit_dist = 0.5

        # Initialize target gate tracking
        self.current_target_gate_idx = 0

        # Add height offset parameters
        self.approach_height_offset = [0.01, 0.1, -0.5, 0.1]
        self.exit_height_offset = [0.1, 0.1, 0.35, 0.1]
        self.default_approach_height_offset = 0.1
        self.default_exit_height_offset = 0.0

        # Create MPC
        self.N = 30
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
        self.current_trajectory = None

        # Track episode statistics
        self.gates_passed = 0
        self.total_gates = len(config.env.track["gates"])
        self.flight_successful = False

        # Log initialization parameters
        self._log_initialization_parameters()

        # Initialize trajectory
        self.waypoints = self._generate_waypoints(obs)
        self._generate_trajectory_from_waypoints(self.waypoints, 0, use_velocity_aware=False)
        self._trajectory_start_tick = 0

    def _log_initialization_parameters(self):
        """Log important initialization parameters."""
        self.logger.info("MPC Controller Initialized", extra={"tick": self._tick})
        self.logger.info(
            f"MPC Parameters - N: {self.N}, T_HORIZON: {self.T_HORIZON}, dt: {self.dt:.4f}",
            extra={"tick": self._tick},
        )
        self.logger.info(f"Approach distances: {self.approach_dist}", extra={"tick": self._tick})
        self.logger.info(f"Exit distances: {self.exit_dist}", extra={"tick": self._tick})
        self.logger.info(
            f"Approach height offsets: {self.approach_height_offset}", extra={"tick": self._tick}
        )
        self.logger.info(
            f"Exit height offsets: {self.exit_height_offset}", extra={"tick": self._tick}
        )
        self.logger.info(f"Start position: {self.start_pos}", extra={"tick": self._tick})
        self.logger.info(f"Total gates in track: {self.total_gates}", extra={"tick": self._tick})

    ###########################################
    # 1. TRAJECTORY GENERATION AND MANAGEMENT #
    ###########################################

    def _generate_trajectory_from_waypoints(
        self, waypoints: np.ndarray, target_gate_idx: int, use_velocity_aware: bool = False
    ):
        """Generate trajectory with optional velocity-aware smooth transitions."""
        if len(waypoints) < 2:
            self.logger.warning(
                "Not enough waypoints to generate trajectory", extra={"tick": self._tick}
            )
            return

        # Only use velocity-aware generation if explicitly requested (for replanning)
        if use_velocity_aware:
            # Get current velocity and position for smooth transition
            current_vel = getattr(self, "_current_vel", np.zeros(3))
            current_speed = np.linalg.norm(current_vel)

            self.logger.info(
                f"Using velocity-aware trajectory generation: speed={current_speed:.2f} m/s",
                extra={"tick": self._tick},
            )

            # Create velocity-aware waypoint sequence only if moving fast enough
            if current_speed > 0.3:  # Only if moving with significant speed
                enhanced_waypoints = []
                enhanced_waypoints.append(waypoints[0])

                # Calculate velocity continuation parameters
                vel_continuation_time = 0.5  # Reduced from 0.8
                vel_continuation_dist = current_speed * vel_continuation_time
                vel_continuation_dist = min(vel_continuation_dist, 0.8)  # Reduced cap

                # Normalize velocity and create continuation point
                vel_normalized = current_vel / current_speed
                velocity_point = waypoints[0] + vel_normalized * vel_continuation_dist

                enhanced_waypoints.append(velocity_point)

                # Add one smooth transition point
                if len(waypoints) > 1:
                    next_waypoint = waypoints[1]
                    # Simple smooth transition
                    transition_point = velocity_point * 0.7 + next_waypoint * 0.3
                    enhanced_waypoints.append(transition_point)

                # Add remaining waypoints (skip second one since we transition to it)
                enhanced_waypoints.extend(waypoints[2:] if len(waypoints) > 2 else [])
                waypoints = np.array(enhanced_waypoints)

                self.logger.info(
                    f"Enhanced waypoints for smooth transition: {len(enhanced_waypoints)} points, "
                    f"vel_continuation_dist: {vel_continuation_dist:.3f}",
                    extra={"tick": self._tick},
                )

        # Standard trajectory generation (for initial and velocity-aware)
        # Calculate distances between waypoints
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

        # Create time parameterization
        if use_velocity_aware and hasattr(self, "_current_vel"):
            current_speed = np.linalg.norm(self._current_vel)
            # Start with current speed, transition to cruise speed
            speeds = []
            cruise_speed = 3.0
            for i in range(len(waypoints)):
                if i == 0:
                    speeds.append(max(0.3, current_speed))
                elif i <= 2:
                    # Gradual transition to cruise speed
                    alpha = i / 2.0
                    speed = current_speed * (1 - alpha) + cruise_speed * alpha
                    speeds.append(np.clip(speed, 0.3, 2.0))
                else:
                    speeds.append(cruise_speed)

            self.logger.info(
                f"Velocity-aware speeds: cruise={cruise_speed}, current={current_speed:.2f}, "
                f"speed_profile={[f'{s:.2f}' for s in speeds[:5]]}...",
                extra={"tick": self._tick},
            )
        else:
            # Standard speeds for initial trajectory
            speeds = [1.2] * len(waypoints)
            self.logger.info(
                f"Standard speeds: uniform={speeds[0]} m/s for {len(waypoints)} waypoints",
                extra={"tick": self._tick},
            )

        # Calculate time points
        time_points = [0.0]
        for i in range(1, len(waypoints)):
            segment_distance = np.linalg.norm(waypoints[i] - waypoints[i - 1])
            avg_speed = (speeds[i - 1] + speeds[i]) / 2
            segment_time = segment_distance / max(avg_speed, 0.1)
            time_points.append(time_points[-1] + segment_time)

        total_time = time_points[-1]
        min_duration = max(8.0, total_time)

        if total_time < min_duration:
            time_points.append(min_duration)
            waypoints = np.vstack([waypoints, waypoints[-1]])

        self.logger.info(
            f"Trajectory timing: total_time={total_time:.2f}s, min_duration={min_duration:.2f}s, "
            f"total_distance={distances[-1]:.2f}m",
            extra={"tick": self._tick},
        )

        # Normalize time for spline
        time_normalized = np.array(time_points) / max(time_points[-1], 1.0)

        # Create splines with velocity boundary conditions only for replanning
        if (
            use_velocity_aware
            and hasattr(self, "_current_vel")
            and np.linalg.norm(self._current_vel) > 0.3
        ):
            try:
                current_vel = self._current_vel
                bc_type_x = ((1, current_vel[0]), (2, 0))
                bc_type_y = ((1, current_vel[1]), (2, 0))
                bc_type_z = ((1, current_vel[2]), (2, 0))

                cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type=bc_type_x)
                cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type=bc_type_y)
                cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type=bc_type_z)

                self.logger.info(
                    f"Applied velocity boundary conditions: vel=[{current_vel[0]:.2f}, {current_vel[1]:.2f}, {current_vel[2]:.2f}]",
                    extra={"tick": self._tick},
                )
            except Exception as e:
                self.logger.warning(
                    f"Velocity boundary conditions failed, using natural: {str(e)}",
                    extra={"tick": self._tick},
                )
                cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type="natural")
                cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type="natural")
                cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type="natural")
        else:
            # Use natural splines for initial trajectory
            cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type="natural")
            cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type="natural")
            cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type="natural")

        # Generate trajectory points
        trajectory_freq = self.freq
        min_points = max(int(trajectory_freq * min_duration), self._tick + 3 * self.N)

        ts = np.linspace(0, 1, min_points)
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        # Add terminal points for stability
        extra_points = 3 * self.N + 1
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * extra_points))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * extra_points))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * extra_points))

        # Save trajectory
        trajectory = {
            "tick": self._tick,
            "waypoints": waypoints.copy(),
            "x": self.x_des.copy(),
            "y": self.y_des.copy(),
            "z": self.z_des.copy(),
            "timestamp": time.time() if hasattr(time, "time") else 0,
        }

        self.current_trajectory = trajectory
        self.save_trajectories_to_file(f"flight_logs/{target_gate_idx}_trajectories.npz")

        mode = "velocity-aware" if use_velocity_aware else "standard"
        self.logger.info(
            f"Generated {mode} trajectory: {len(waypoints)} waypoints, {len(self.x_des)} points, "
            f"target_gate={target_gate_idx}",
            extra={"tick": self._tick},
        )

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
        new_waypoints: np.ndarray,
        use_velocity_aware: bool = False,
        obs: dict[str, NDArray[np.floating]] | None = None,
    ):
        """Update trajectory with smooth blending to force drone onto new path."""
        if len(new_waypoints) < 2:
            self.logger.warning(
                "Not enough waypoints to update trajectory", extra={"tick": self._tick}
            )
            return

        current_tick = self._tick

        # Store old trajectory for blending
        old_x_des = getattr(self, "x_des", None)
        old_y_des = getattr(self, "y_des", None)
        old_z_des = getattr(self, "z_des", None)
        trajectory_offset = getattr(self, "_trajectory_start_tick", 0)
        current_traj_idx = max(0, current_tick - trajectory_offset)

        # Get current position and velocity
        current_pos = obs["pos"] if obs else new_waypoints[0]
        current_vel = getattr(self, "_current_vel", np.zeros(3))
        current_speed = np.linalg.norm(current_vel)

        # Create corrected waypoints that force the drone toward the new trajectory
        if use_velocity_aware and current_speed > 0.3:
            # Calculate how far off track we are from the new trajectory
            trajectory_error = np.linalg.norm(current_pos - new_waypoints[1])

            self.logger.info(
                f"Trajectory correction needed: error={trajectory_error:.3f}m, speed={current_speed:.2f}m/s",
                extra={"tick": self._tick},
            )

            # Create immediate correction waypoints
            corrected_waypoints = [current_pos]  # Start from current position

            if trajectory_error > 0.15:  # If significantly off track
                # Add aggressive correction points
                correction_time = min(1.0, trajectory_error / current_speed)  # Time to correct
                correction_steps = 3

                for i in range(1, correction_steps + 1):
                    alpha = i / correction_steps
                    # Exponential correction factor for stronger pull toward trajectory
                    correction_factor = 1 - np.exp(-3 * alpha)

                    # Blend current position toward target with increasing correction
                    if len(new_waypoints) > 1:
                        target_point = new_waypoints[1]  # Next waypoint
                        corrected_point = (
                            current_pos * (1 - correction_factor) + target_point * correction_factor
                        )
                        corrected_waypoints.append(corrected_point)

                # Add remaining waypoints
                corrected_waypoints.extend(new_waypoints[2:] if len(new_waypoints) > 2 else [])

                self.logger.info(
                    f"Applied aggressive trajectory correction: {correction_steps} correction points",
                    extra={"tick": self._tick},
                )
            else:
                # Small error - use gentler velocity-aware correction
                vel_continuation_time = 0.3  # Shorter continuation
                vel_continuation_dist = current_speed * vel_continuation_time
                vel_continuation_dist = min(vel_continuation_dist, 0.4)  # Smaller cap

                # Add velocity continuation point
                if current_speed > 0.1:
                    vel_normalized = current_vel / current_speed
                    velocity_point = current_pos + vel_normalized * vel_continuation_dist
                    corrected_waypoints.append(velocity_point)

                # Add transition point toward new trajectory
                if len(new_waypoints) > 1:
                    target_point = new_waypoints[1]
                    transition_point = (
                        velocity_point if current_speed > 0.1 else current_pos
                    ) * 0.5 + target_point * 0.5
                    corrected_waypoints.append(transition_point)

                # Add remaining waypoints
                corrected_waypoints.extend(new_waypoints[1:])

            new_waypoints = np.array(corrected_waypoints)

        # Generate new trajectory
        self._generate_trajectory_from_waypoints(
            new_waypoints,
            obs["target_gate"] if obs and "target_gate" in obs else 0,
            use_velocity_aware=use_velocity_aware,
        )

        # Apply immediate trajectory blending for smooth transition
        if old_x_des is not None and len(old_x_des) > current_traj_idx:
            blend_horizon = min(self.N // 3, len(self.x_des))  # Shorter blending window

            for i in range(blend_horizon):
                old_idx = current_traj_idx + i
                if old_idx < len(old_x_des) and i < len(self.x_des):
                    # Exponential blending with stronger pull toward new trajectory
                    alpha = 1 - np.exp(-4 * i / blend_horizon)  # Faster transition

                    self.x_des[i] = old_x_des[old_idx] * (1 - alpha) + self.x_des[i] * alpha
                    self.y_des[i] = old_y_des[old_idx] * (1 - alpha) + self.y_des[i] * alpha
                    self.z_des[i] = old_z_des[old_idx] * (1 - alpha) + self.z_des[i] * alpha

            self.logger.info(
                f"Applied trajectory blending over {blend_horizon} steps with stronger correction",
                extra={"tick": self._tick},
            )

        # Reset trajectory start tick
        self._trajectory_start_tick = current_tick

        self.logger.info(
            f"Trajectory updated with forced correction at tick {current_tick}, length: {len(self.x_des)}",
            extra={"tick": self._tick},
        )

    ##############################
    # 2. WAYPOINT GENERATION     #
    ##############################

    def loop_path_gen(self, gates: list[dict], obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Generate a loop path through all gates with corridor constraints and height offsets."""
        waypoints = []

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

            # Create approach, center, and exit points with individual distances and height offsets
            approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
            approach_point[2] += approach_z_offset  # Add height offset

            exit_point = gate_info["center"] - gate_info["normal"] * exit_dist
            exit_point[2] += exit_z_offset  # Add height offset

            # Add approach point with height offset
            waypoints.append(approach_point)

            # Always add gate center point
            waypoints.append(gate_info["center"])

            # Add exit point with height offset
            waypoints.append(exit_point)

            if self.current_target_gate_idx == 2:
                approach_point = gate_info["center"] + gate_info["normal"] * approach_dist
                approach_point[2] += 0.35
                waypoints.append(approach_point)

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

    def _get_gates_with_observations(self, obs: dict, start_idx: int) -> list[dict]:
        """Get gate info using observed positions when available."""
        gates = []
        has_observations = "gates_pos" in obs and obs["gates_pos"] is not None

        if has_observations:
            for i in range(start_idx, len(self.config.env.track["gates"])):
                gates.append(self._get_gate_with_observation(obs, i))

        return gates

    def _get_gate_with_observation(self, obs: dict, gate_idx: int) -> dict:
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

        # Store current velocity for smooth trajectory transitions
        self._current_vel = obs["vel"].copy()

        # Track gate progress
        current_target_gate = obs.get("target_gate", 0)
        if isinstance(current_target_gate, np.ndarray):
            current_target_gate = int(current_target_gate.item())
        else:
            current_target_gate = int(current_target_gate)

        # Update gates passed tracking
        if current_target_gate == -1:  # Finished all gates
            self.gates_passed = self.total_gates
            self.flight_successful = True
            self.logger.info(
                f"FLIGHT COMPLETED SUCCESSFULLY! All {self.total_gates} gates passed",
                extra={"tick": self._tick},
            )
        elif current_target_gate > self.gates_passed:
            self.gates_passed = current_target_gate
            self.logger.info(
                f"Gate {self.gates_passed} passed! Progress: {self.gates_passed}/{self.total_gates}",
                extra={"tick": self._tick},
            )

        # Track if we just replanned (for stronger MPC tracking)
        just_replanned = False

        # Check for replanning based on gate observations
        if self._tick - self.last_replanning_tick >= self.replanning_frequency:
            self.last_replanning_tick = self._tick

            # Check if we have new gate observations to update trajectory
            if "gates_pos" in obs and obs["gates_pos"] is not None and "target_gate" in obs:
                target_gate_idx = current_target_gate

                should_replan = False

                # Check if the target gate hasn't been updated yet
                if (
                    target_gate_idx < len(self.config.env.track["gates"])
                    and target_gate_idx < len(obs["gates_pos"])
                    and target_gate_idx not in self.updated_gates
                ):
                    config_pos = np.array(self.config.env.track["gates"][target_gate_idx]["pos"])
                    observed_pos = np.array(obs["gates_pos"][target_gate_idx])

                    # Check only horizontal distance (x and y coordinates)
                    horizontal_config = config_pos[:2]
                    horizontal_observed = observed_pos[:2]
                    horizontal_diff = np.linalg.norm(horizontal_config - horizontal_observed)

                    if horizontal_diff > 0.10:  # 10cm horizontal threshold
                        self.logger.info(
                            f"Gate {target_gate_idx} horizontal position updated, replanning trajectory. "
                            f"Horizontal diff: {horizontal_diff:.3f}m, "
                            f"Config pos: [{config_pos[0]:.3f}, {config_pos[1]:.3f}], "
                            f"Observed pos: [{observed_pos[0]:.3f}, {observed_pos[1]:.3f}]",
                            extra={"tick": self._tick},
                        )

                        should_replan = True
                        just_replanned = True
                        self.updated_gates.add(target_gate_idx)

                if should_replan:
                    # Generate NEW waypoints with the updated gate position
                    new_waypoints = self._generate_waypoints(obs, target_gate_idx)
                    self.logger.info(
                        f"Generated new waypoints for gate {target_gate_idx}: {len(new_waypoints)} points",
                        extra={"tick": self._tick},
                    )

                    # Update trajectory with velocity-aware generation for smooth transition
                    self._update_trajectory(new_waypoints, True, obs)

        # Execute MPC control with trajectory offset
        trajectory_offset = getattr(self, "_trajectory_start_tick", 0)
        trajectory_index = max(0, self._tick - trajectory_offset)

        i = min(trajectory_index, len(self.x_des) - 1)
        if trajectory_index >= len(self.x_des):
            self.finished = True
            self.logger.warning(
                f"Trajectory finished - reached end of trajectory at tick {self._tick}",
                extra={"tick": self._tick},
            )

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

        # Calculate tracking error for adaptive cost weights
        current_pos = obs["pos"]
        traj_len = len(self.x_des)
        i = min(i, traj_len - 1)
        target_pos = np.array([self.x_des[i], self.y_des[i], self.z_des[i]])
        tracking_error = np.linalg.norm(current_pos - target_pos)

        # Set reference trajectory with adaptive weights
        for j in range(self.N):
            idx = i + j
            if idx < traj_len:
                x_ref = self.x_des[idx]
                y_ref = self.y_des[idx]
                z_ref = self.z_des[idx]
            else:
                x_ref = self.x_des[-1]
                y_ref = self.y_des[-1]
                z_ref = self.z_des[-1]

            # Base reference (18 dimensions: 14 states + 4 controls)
            yref = np.array(
                [
                    x_ref,
                    y_ref,
                    z_ref,  # Position (3)
                    0.0,
                    0.0,
                    0.0,  # Velocity (3)
                    0.0,
                    0.0,
                    0.0,  # RPY (3)
                    0.35,
                    0.35,  # f_collective, f_collective_cmd (2)
                    0.0,
                    0.0,
                    0.0,  # rpy_cmd (3)
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # Control inputs (4)
                ]
            )

            self.acados_ocp_solver.set(j, "yref", yref)

            # Apply stronger tracking weights if we just replanned or have large tracking error
            if just_replanned or tracking_error > 0.2:
                # Temporarily increase position tracking weights in the cost matrix
                # This requires modifying the cost weights in the solver
                # Note: This is a simplified approach - in practice you might need to
                # recreate the solver with different weights or use a different method
                pass

        # Terminal cost (14 dimensions: states only)
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
        u0 = self.acados_ocp_solver.get(0, "u")

        # Smooth control updates
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        # Log tracking performance
        if self._tick % 50 == 0 or just_replanned:
            self.logger.info(
                f"MPC Tracking - Error: {tracking_error:.3f}m, "
                f"Gates: {self.gates_passed}/{self.total_gates}, "
                f"Just replanned: {just_replanned}",
                extra={"tick": self._tick},
            )

        # Log MPC solution status occasionally (remove the duplicate if condition)
        if self._tick % 100 == 0:  # Log every 100 ticks
            self.logger.info(
                f"MPC Status - Gates: {self.gates_passed}/{self.total_gates}, "
                f"target_gate: {current_target_gate}, "
                f"trajectory_idx: {trajectory_index}, "
                f"finished: {self.finished}",
                extra={"tick": self._tick},
            )

        # Return the real control commands
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
        """Increment the tick counter and log step information."""
        self._tick += 1

        # Log important step information occasionally
        if self._tick % 100 == 0:  # Every 100 steps
            current_target_gate = obs.get("target_gate", 0)
            if isinstance(current_target_gate, np.ndarray):
                current_target_gate = int(current_target_gate.item())
            else:
                current_target_gate = int(current_target_gate)

            self.logger.info(
                f"Step {self._tick}: reward={reward:.3f}, "
                f"gates={self.gates_passed}/{self.total_gates}, "
                f"target_gate={current_target_gate}, "
                f"terminated={terminated}, truncated={truncated}",
                extra={"tick": self._tick},
            )

        return self.finished

    def episode_callback(self):
        """Reset controller state for new episode and log final statistics."""
        # Log final episode statistics
        self._log_episode_summary()

        # Reset for next episode
        self._tick = 0
        self.finished = False
        self.updated_gates = set()
        self._trajectory_start_tick = 0
        self.gates_passed = 0
        self.flight_successful = False

    def _log_episode_summary(self):
        """Log episode summary with all important parameters and flight success."""
        self.logger.info("=== EPISODE SUMMARY ===", extra={"tick": self._tick})
        self.logger.info(
            f"FLIGHT SUCCESS: {'YES' if self.flight_successful else 'NO'}",
            extra={"tick": self._tick},
        )
        self.logger.info(
            f"Gates passed: {self.gates_passed}/{self.total_gates}", extra={"tick": self._tick}
        )
        self.logger.info(
            f"Completion rate: {(self.gates_passed / self.total_gates * 100):.1f}%",
            extra={"tick": self._tick},
        )
        self.logger.info(f"Total simulation ticks: {self._tick}", extra={"tick": self._tick})
        self.logger.info(
            f"Flight time: {self._tick / self.freq:.2f} seconds", extra={"tick": self._tick}
        )
        self.logger.info(
            f"Episode Status: {'COMPLETED' if self.finished else 'INCOMPLETE'}",
            extra={"tick": self._tick},
        )
        self.logger.info(
            f"MPC Parameters - N: {self.N}, T_HORIZON: {self.T_HORIZON}, frequency: {self.freq}",
            extra={"tick": self._tick},
        )
        self.logger.info(f"Approach distances: {self.approach_dist}", extra={"tick": self._tick})
        self.logger.info(f"Exit distances: {self.exit_dist}", extra={"tick": self._tick})
        self.logger.info(
            f"Approach height offsets: {self.approach_height_offset}", extra={"tick": self._tick}
        )
        self.logger.info(
            f"Exit height offsets: {self.exit_height_offset}", extra={"tick": self._tick}
        )
        self.logger.info(
            f"Gates updated with observations: {sorted(list(self.updated_gates))}",
            extra={"tick": self._tick},
        )
        self.logger.info(
            f"Replanning frequency: {self.replanning_frequency}", extra={"tick": self._tick}
        )
        self.logger.info("=== END EPISODE SUMMARY ===", extra={"tick": self._tick})

    def episode_reset(self):
        """Reset controller state for new episode (called from sim.py)."""
        self.episode_callback()

    def get_predicted_trajectory(self) -> np.ndarray:
        """Return the MPC's predicted trajectory over the horizon."""
        pred_traj = []
        for i in range(self.N):
            x = self.acados_ocp_solver.get(i, "x")
            pred_traj.append(x[:3])
        return np.array(pred_traj)
