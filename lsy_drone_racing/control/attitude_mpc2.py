"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
import os
import time
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

    # Sys ID Params - reduced gains for less aggressive control
    params_pitch_rate = [-5.5, 5.7]  # Slightly reduced from original
    params_roll_rate = [-3.6, 3.7]   # Slightly reduced from original
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.0, 3.5]  # Slightly reduced from original

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

    return model


def create_ocp_solver(
    Tf: float, N: int, config: dict, verbose: bool = False
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

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights - adjusted for less aggressive control
    Q = np.diag(
        [
            5.0,
            5.0,
            5.0,  # Position (reduced from 10.0)
            0.3,
            0.3,
            0.3,  # Velocity (increased from 0.01)
            0.8,
            0.8,
            0.8,  # rpy (increased from 0.1)
            0.02,
            0.02,  # f_collective, f_collective_cmd (increased from 0.01)
            0.05,
            0.05,
            0.05,
        ]
    )  # rpy_cmd (increased from 0.01)

    R = np.diag([0.05, 0.05, 0.05, 0.05])  # Input cost (increased from 0.01)

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

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints - reduced angle limits for less aggressive control
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.0, -1.0, -1.0])  # Reduced from -1.57
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.0, 1.0, 1.0])   # Reduced from 1.57
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Add obstacle avoidance as nonlinear path constraints
    if hasattr(config, "env") and hasattr(config.env, "track") and "obstacles" in config.env.track:
        safety_radius = 0.3  # 30cm minimum distance
        
        # Create constraint for each obstacle (without the time-step loop)
        all_obs_expr = []
        
        for obstacle in config.env.track["obstacles"]:
            obs_pos = np.array(obstacle["pos"])
            
            # Distance to obstacle (squared for numerical stability)
            dist_squared = (model.x[0] - obs_pos[0])**2 + (model.x[1] - obs_pos[1])**2 + (model.x[2] - obs_pos[2])**2
            all_obs_expr.append(dist_squared)
        
        # Only add constraints if there are obstacles
        if all_obs_expr:
            # Convert list of expressions to a single vertically stacked expression
            ocp.model.con_h_expr = vertcat(*all_obs_expr)  # IMPORTANT: use model.con_h_expr, not constraints.con_h_expr
            
            # Set bounds with matching dimensions
            num_constraints = len(all_obs_expr)
            ocp.constraints.lh = np.array([safety_radius**2] * num_constraints)
            ocp.constraints.uh = np.array([np.inf] * num_constraints)

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
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
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0
        self.start_pos = obs["pos"]
        self.config = config

        # Generate waypoints that account for gate orientation
        waypoints = self._generate_waypoints(obs)
        
        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        # Longer completion time for smoother trajectories
        des_completion_time = 12  # Increased from 8 for less aggressive behavior
        ts = np.linspace(0, 1, int(self.freq * des_completion_time))

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        self.N = 30
        self.T_HORIZON = 1.2  # Increased from original
        self.dt = self.T_HORIZON / self.N
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))

        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N, config)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.finished = False

        # Add visualization and monitoring setup
        self.trajectory_history = []  # Store drone positions
        self.reference_error_history = []  # Track reference tracking error
        self.time_history = []  # Store timestamps
        self.start_time = None  # Flight start time
        self.visualization_initialized = False
        self.fig = None
        self.axes = None
        self.plot_data = {}  # Store plot references
        
        # Set this to True to enable real-time visualization
        self.enable_visualization = True
        
        # Performance metrics
        self.mpc_solve_times = []
        self.gate_times = [None] * len(obs["gates_pos"])
        self.gate_passed = [False] * len(obs["gates_pos"])
        self.closest_obstacle_distances = []

        # Log creation
        log_dir = "flight_logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"flight_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.log_file, 'w') as f:
            f.write("time,x,y,z,vx,vy,vz,roll,pitch,yaw,ref_x,ref_y,ref_z,error,thrust,gate_idx\n")

    def _generate_waypoints(self, obs):
        """Generate waypoints that properly navigate through vertical gates"""
        waypoints = []
        
        # Add starting position
        waypoints.append(self.start_pos)
        
        # For each gate, create waypoints that approach and pass through the gate correctly
        for i, gate in enumerate(self.config.env.track["gates"]):
            pos = np.array(gate["pos"])
            
            # Get gate orientation
            if "rpy" in gate:
                rpy = np.array(gate["rpy"])
                rotation = R.from_euler('xyz', rpy)
            else:
                rotation = R.from_euler('xyz', [0, 0, 0])
            
            # Get normal vector (perpendicular to gate plane)
            normal = rotation.apply([1.0, 0.0, 0.0])
            
            # Create pre-gate and post-gate waypoints along the normal direction
            approach_dist = 1.0  # Distance before gate
            exit_dist = 0.8     # Distance after gate
            
            # Approach waypoint - ensure we approach from the correct direction
            approach_point = pos - normal * approach_dist
            
            # Exit waypoint - continue in same direction after gate
            exit_point = pos + normal * exit_dist
            
            # Add waypoints to approach and exit the gate correctly
            waypoints.append(approach_point)
            waypoints.append(pos)  # Gate center
            waypoints.append(exit_point)
        
        # Convert list to numpy array
        return np.array(waypoints)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        start_solve_time = time.time()
        
        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        for j in range(self.N):
            yref = np.array(
                [
                    self.x_des[i + j],
                    self.y_des[i + j],
                    self.z_des[i + j],
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
        yref_N = np.array(
            [
                self.x_des[i + self.N],
                self.y_des[i + self.N],
                self.z_des[i + self.N],
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

        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")
        
        # Modified smoothing for less aggressive control
        w = 0.7 / self.config.env.freq / self.dt  # Reduced factor for smoother transition
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]
        
        # For visualization & logging
        solve_time = time.time() - start_solve_time
        self.mpc_solve_times.append(solve_time)
        
        # Initialize visualization if not already done
        if self.enable_visualization and not self.visualization_initialized:
            self._initialize_visualization()
        
        # Update flight data
        self._update_flight_data(obs, cmd, time.time())
        
        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter and check for termination."""
        self._tick += 1
        
        # Generate report if flight is finished
        if terminated or truncated:
            self.generate_flight_report()
            self.finished = True
            
        return self.finished

    def episode_callback(self):
        """Reset controller for a new episode."""
        self._tick = 0
        self.finished = False
        
        # Reset visualization data
        self.trajectory_history = []
        self.reference_error_history = [] 
        self.time_history = []
        self.start_time = None
        self.mpc_solve_times = []
        self.gate_times = [None] * len(self.gate_passed)
        self.gate_passed = [False] * len(self.gate_passed)
        self.closest_obstacle_distances = []
        
        # Reset control variables
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3

    def _initialize_visualization(self):
        """Initialize the real-time visualization plots."""
        if not self.enable_visualization:
            return
            
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 9))
        gs = self.fig.add_gridspec(2, 3)
        
        # 3D trajectory plot
        self.axes = {}
        self.axes['3d'] = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.axes['xy'] = self.fig.add_subplot(gs[0, 1])
        self.axes['xz'] = self.fig.add_subplot(gs[0, 2])
        self.axes['metrics'] = self.fig.add_subplot(gs[1, 1:])
        
        # Initialize 3D plot elements
        self.plot_data['traj_3d'] = self.axes['3d'].plot([], [], [], 'b-', linewidth=2, label='Actual')[0]
        self.plot_data['ref_3d'] = self.axes['3d'].plot(self.x_des, self.y_des, self.z_des, 'r--', 
                                                       alpha=0.7, linewidth=1, label='Reference')[0]
        
        # Plot gates and obstacles
        self._plot_environment(self.axes['3d'])
        
        # Initialize 2D plots
        self.plot_data['traj_xy'] = self.axes['xy'].plot([], [], 'b-', linewidth=2)[0]
        self.plot_data['ref_xy'] = self.axes['xy'].plot(self.x_des, self.y_des, 'r--', alpha=0.7, linewidth=1)[0]
        
        self.plot_data['traj_xz'] = self.axes['xz'].plot([], [], 'b-', linewidth=2)[0]
        self.plot_data['ref_xz'] = self.axes['xz'].plot(self.x_des, self.z_des, 'r--', alpha=0.7, linewidth=1)[0]
        
        # Initialize metrics plot
        self.plot_data['error'] = self.axes['metrics'].plot([], [], 'g-', label='Position Error (m)')[0]
        self.plot_data['thrust'] = self.axes['metrics'].plot([], [], 'r-', label='Thrust')[0]
        
        # Set labels
        self.axes['3d'].set_xlabel('X (m)')
        self.axes['3d'].set_ylabel('Y (m)')
        self.axes['3d'].set_zlabel('Z (m)')
        self.axes['3d'].set_title('3D Trajectory')
        self.axes['3d'].legend()
        
        self.axes['xy'].set_xlabel('X (m)')
        self.axes['xy'].set_ylabel('Y (m)')
        self.axes['xy'].set_title('Top View')
        
        self.axes['xz'].set_xlabel('X (m)')
        self.axes['xz'].set_ylabel('Z (m)')
        self.axes['xz'].set_title('Side View')
        
        self.axes['metrics'].set_xlabel('Time (s)')
        self.axes['metrics'].set_title('Flight Metrics')
        self.axes['metrics'].legend()
        self.axes['metrics'].grid(True)
        
        # Set equal aspect ratio
        self.axes['xy'].set_aspect('equal')
        self.axes['xz'].set_aspect('equal')
        
        # Add gate indicators to the metrics plot
        for i, _ in enumerate(self.gate_passed):
            self.axes['metrics'].axvline(x=-1, color='g', linestyle='--', alpha=0.5)
        
        # Set limits based on reference trajectory
        pad = 0.5  # Add padding to the plot limits
        x_min, x_max = min(self.x_des) - pad, max(self.x_des) + pad
        y_min, y_max = min(self.y_des) - pad, max(self.y_des) + pad
        z_min, z_max = max(0, min(self.z_des) - pad), max(self.z_des) + pad
        
        self.axes['3d'].set_xlim([x_min, x_max])
        self.axes['3d'].set_ylim([y_min, y_max])
        self.axes['3d'].set_zlim([z_min, z_max])
        
        self.axes['xy'].set_xlim([x_min, x_max])
        self.axes['xy'].set_ylim([y_min, y_max])
        
        self.axes['xz'].set_xlim([x_min, x_max])
        self.axes['xz'].set_ylim([z_min, z_max])
        
        plt.tight_layout()
        
        # Initialize animation
        plt.ion()  # Turn on interactive mode
        self.anim = animation.FuncAnimation(
            self.fig, self._update_plots, interval=100, 
            fargs=(), cache_frame_data=False, blit=False
        )
        
        plt.show(block=False)
        self.visualization_initialized = True

    def _plot_environment(self, ax):
        """Plot gates and obstacles in the 3D environment."""
        # Plot gates as vertical circles
        for i, gate_pos in enumerate(self.config.env.track["gates"]):
            pos = np.array(gate_pos["pos"])
            
            # Get gate orientation from roll-pitch-yaw
            rpy = np.array(gate_pos["rpy"]) if "rpy" in gate_pos else np.array([0, 0, 0])
            rotation = R.from_euler('xyz', rpy)
                
            # Draw gate as a vertical circle
            theta = np.linspace(0, 2*np.pi, 30)
            radius = 0.3  # Gate radius
            
            # Create circle in YZ plane (vertical)
            circle_points = np.zeros((len(theta), 3))
            circle_points[:, 1] = radius * np.cos(theta)  # Y component
            circle_points[:, 2] = radius * np.sin(theta)  # Z component
            
            # Rotate circle based on gate orientation and translate to gate position
            rotated_points = rotation.apply(circle_points) + pos
            
            # Plot the gate circle
            ax.plot(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], 
                   'g-', linewidth=2, alpha=0.8)
            
            # Add gate marker at center with number
            ax.scatter(pos[0], pos[1], pos[2], color='green', s=100, marker='o', 
                     label=f'Gate {i+1}' if i == 0 else "")
            ax.text(pos[0], pos[1], pos[2] + 0.2, f"G{i+1}", color='green', fontsize=10, ha='center')
            
            # Add arrow showing gate orientation/passage direction
            direction = rotation.apply([0.4, 0, 0])  # Forward vector
            ax.quiver(pos[0], pos[1], pos[2], 
                     direction[0], direction[1], direction[2], 
                     color='green', length=0.2, normalize=True)
        
        # Plot obstacles as vertical cylinders
        for i, obs_pos in enumerate(self.config.env.track["obstacles"]):
            pos = np.array(obs_pos["pos"])
            ax.scatter(pos[0], pos[1], pos[2], color='red', s=80, marker='x', 
                     label='Obstacle' if i == 0 else "")
            
            # Draw vertical cylinder for obstacle
            safety_radius = 0.3  # Safety radius
            height = 1.2  # Cylinder height
            z_bottom = pos[2] - height/2
            z_top = pos[2] + height/2
            
            # Create circles for top and bottom
            theta = np.linspace(0, 2*np.pi, 20)
            x_circle = pos[0] + safety_radius * np.cos(theta)
            y_circle = pos[1] + safety_radius * np.sin(theta)
            
            # Plot bottom and top circles
            ax.plot(x_circle, y_circle, [z_bottom] * len(theta), 'r-', alpha=0.3)
            ax.plot(x_circle, y_circle, [z_top] * len(theta), 'r-', alpha=0.3)
            
            # Connect circles with lines to form cylinder
            for j in range(len(theta)):
                ax.plot([x_circle[j], x_circle[j]], 
                       [y_circle[j], y_circle[j]], 
                       [z_bottom, z_top], 'r-', alpha=0.1)

    def _update_plots(self, frame):
        """Update all plots with current data."""
        if not self.trajectory_history:
            return
        
        # Extract trajectory data
        traj_array = np.array(self.trajectory_history)
        times = np.array(self.time_history)
        errors = np.array(self.reference_error_history)
        
        # Update 3D trajectory
        self.plot_data['traj_3d'].set_data(traj_array[:, 0], traj_array[:, 1])
        self.plot_data['traj_3d'].set_3d_properties(traj_array[:, 2])
        
        # Update 2D projections
        self.plot_data['traj_xy'].set_data(traj_array[:, 0], traj_array[:, 1])
        self.plot_data['traj_xz'].set_data(traj_array[:, 0], traj_array[:, 2])
        
        # Update metrics
        if len(times) > 0:
            self.plot_data['error'].set_data(times, errors)
            self.plot_data['thrust'].set_data(times, [t[9] for t in self.trajectory_history])
            
            # Update gate indicators if any gates were passed
            for i, gate_time in enumerate(self.gate_times):
                if gate_time is not None:
                    self.axes['metrics'].axvline(x=gate_time, color='g', linestyle='--', alpha=0.5)
        
        # Set metrics axis limits
        if len(times) > 1:
            self.axes['metrics'].set_xlim([0, max(times)])
            if len(errors) > 0:
                self.axes['metrics'].set_ylim([0, max(max(errors) * 1.1, 1.0)])
        
        # Return the modified artists
        return [self.plot_data[k] for k in self.plot_data]

    def _update_flight_data(self, obs, cmd, current_time):
        """Update flight data for visualization and logging."""
        if self.start_time is None:
            self.start_time = current_time
            
        # Calculate time since start of flight
        flight_time = current_time - self.start_time
        
        # Get current state
        pos = obs["pos"]
        vel = obs["vel"]
        quat = obs["quat"]
        r = R.from_quat(quat)
        rpy = r.as_euler("xyz", degrees=False)
        
        # Get reference at current tick
        i = min(self._tick, len(self.x_des) - 1)
        ref_pos = np.array([self.x_des[i], self.y_des[i], self.z_des[i]])
        
        # Calculate error
        pos_error = np.linalg.norm(pos - ref_pos)
        
        # Store data (pos, vel, rpy, thrust)
        self.trajectory_history.append(np.concatenate([pos, vel, rpy, [cmd[0]]]))
        self.reference_error_history.append(pos_error)
        self.time_history.append(flight_time)
        
        # Check for gate passage with improved detection
        for i, gate_pos in enumerate(obs["gates_pos"]):
            if not self.gate_passed[i]:
                # Get gate position and orientation
                pos_gate = gate_pos
                
                # Get gate orientation if available
                if "rpy" in self.config.env.track["gates"][i]:
                    rpy = np.array(self.config.env.track["gates"][i]["rpy"])
                    rotation = R.from_euler('xyz', rpy)
                else:
                    rotation = R.from_euler('xyz', [0, 0, 0])
                
                # Gate normal vector (points in direction drone should pass)
                normal = rotation.apply([1.0, 0.0, 0.0])
                
                # Vector from gate to drone
                gate_to_drone = pos - pos_gate
                
                # Distance from drone to gate center
                dist_to_gate = np.linalg.norm(gate_to_drone)
                
                # Check if drone is close to the gate
                if dist_to_gate < 0.5:  # Increased detection radius slightly
                    # Check drone's velocity direction alignment with gate orientation
                    vel_alignment = np.dot(normal, obs["vel"])
                    
                    # Gate is considered passed if close to gate and moving in correct direction
                    if vel_alignment > 0.5:
                        self.gate_passed[i] = True
                        self.gate_times[i] = flight_time
                        print(f"Passed gate {i+1} at time {flight_time:.2f}s!")
        
        # Calculate closest obstacle
        min_obstacle_dist = float('inf')
        for obs_pos in obs["obstacles_pos"]:
            dist = np.linalg.norm(pos - obs_pos)
            min_obstacle_dist = min(min_obstacle_dist, dist)
        
        self.closest_obstacle_distances.append(min_obstacle_dist)
        
        # Log data to file
        with open(self.log_file, 'a') as f:
            gate_idx = sum(self.gate_passed)
            f.write(f"{flight_time:.4f},{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},"
                    f"{vel[0]:.4f},{vel[1]:.4f},{vel[2]:.4f},"
                    f"{rpy[0]:.4f},{rpy[1]:.4f},{rpy[2]:.4f},"
                    f"{ref_pos[0]:.4f},{ref_pos[1]:.4f},{ref_pos[2]:.4f},"
                    f"{pos_error:.4f},{cmd[0]:.4f},{gate_idx}\n")

    def generate_flight_report(self):
        """Generate a report of flight statistics and finalize visualization."""
        if not self.trajectory_history:
            return "No flight data recorded."
            
        # Calculate statistics
        avg_error = np.mean(self.reference_error_history)
        max_error = np.max(self.reference_error_history)
        avg_solve_time = np.mean(self.mpc_solve_times)
        min_obstacle_dist = np.min(self.closest_obstacle_distances)
        gates_passed = sum(self.gate_passed)
        total_time = self.time_history[-1] if self.time_history else 0
        
        report = (
            f"Flight Report:\n"
            f"Total Flight Time: {total_time:.2f} seconds\n"
            f"Average Position Error: {avg_error:.4f} m\n"
            f"Maximum Position Error: {max_error:.4f} m\n"
            f"Average MPC Solve Time: {avg_solve_time:.4f} seconds\n"
            f"Minimum Distance to Obstacles: {min_obstacle_dist:.4f} m\n"
            f"Gates Passed: {gates_passed}/{len(self.gate_passed)}\n"
        )
        print(report)

        # Finalize visualization
        # if self.enable_visualization and self.visualization_initialized:
        #     plt.ioff()  # Turn off interactive mode
        #     plt.show()
        return report
        # Close the log file
        with open(self.log_file, 'a') as f:
            f.write("Flight report generated.\n")
        with open(self.log_file, 'a') as f:
            f.write(report)
