"""This module implements an example MPC using attitude control for a quadrotor.

smoothed control commands with obstacle avoidance

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

import os
import time

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface with CasADi."""

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

        # Define Gravitational Acceleration
        self.GRAVITY = 9.806

        # Sys ID Params - original
        self.params_pitch_rate = [-6.003842038081178, 6.213752925707588]  # pitch dynamics
        self.params_roll_rate = [-3.960889336015948, 4.078293254657104]  # roll dynamics
        self.params_yaw_rate = [-0.005347588299390372, 0.0]  # yaw dynamics
        self.params_acc = [20.907574256269616, 3.653687545690674]  # acceleration dynamics

        # Generate waypoints that account for gate orientation
        waypoints = self._generate_waypoints(obs)
        # waypoints = np.array([
        #     [0.5, 0.5, 1.5],
        #     [1.5, 0.5, 1.5],
        #     [1.5, 1.5, 1.5],
        #     [0.5, 1.5, 1.5]
        # ])
        # Same waypoints as in the trajectory controller. Determined by trial and error.
        # waypoints = np.array([
        #     self.start_pos,
        #     obs["obstacles_pos"][0] + [0.2, 0.5, -0.7],
        #     obs["obstacles_pos"][0] + [0.2, -0.3, -0.7],
        #     obs["gates_pos"][0] + 0.5 * (obs["obstacles_pos"][0] - [0, 0, 0.6] - obs["gates_pos"][0]),
        #     obs["gates_pos"][0] + [-0.1, 0.1, 0],
        #     obs["gates_pos"][0] + [-0.6, -0.2, 0],
        #     obs["obstacles_pos"][1] + [-0.3, -0.3, -0.7],
        #     obs["gates_pos"][1] + [-0.1, -0.2, 0],
        #     obs["gates_pos"][1],
        #     obs["gates_pos"][1] + [0.2, 0.5, 0],
        #     obs["obstacles_pos"][0] + [-0.3, 0, -0.7],
        #     obs["gates_pos"][2] + [0.2, -0.5, 0],
        #     obs["gates_pos"][2] + [0.1, 0, 0],
        #     obs["gates_pos"][2] + [0.1, 0.15, 0],
        #     obs["gates_pos"][2] + [0.1, 0.15, 1],
        #     obs["obstacles_pos"][3] + [0.4, 0.3, -0.2],
        #     obs["obstacles_pos"][3] + [0.4, 0, -0.2],
        #     obs["gates_pos"][3],
        #     obs["gates_pos"][3] + [0, -0.5, 0],
        # ])
        # waypoints = np.array(
        #     [
        #         [1.0, 1.5, 0.05],
        #         [0.8, 1.0, 0.2],
        #         [0.55, -0.3, 0.5],
        #         [0.2, -1.3, 0.65],
        #         [1.1, -0.85, 1.1],
        #         [0.2, 0.5, 0.65],
        #         [0.0, 1.2, 0.525],
        #         [0.0, 1.2, 1.1],
        #         [-0.5, 0.0, 1.1],
        #         [-0.5, -0.5, 1.1],
        #     ]
        # )
        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        des_completion_time = 12  # Increase from 8 to 12 seconds
        ts = np.linspace(0, 1, int(self.freq * des_completion_time))

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        self.N = 30  # Horizon length
        self.T_HORIZON = 1.2 # Time horizon
        self.dt = self.T_HORIZON / self.N  # Time step

        # Extend trajectory for prediction horizon
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))

        # Create CasADi MPC solver
        self._setup_casadi_solver()

        self.last_f_collective = 0.3 # last collective thrust
        self.last_rpy_cmd = np.zeros(3) # last roll/pitch/yaw command
        self.last_f_cmd = 0.3 # last thrust command
        self.config = config
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
        # self.closest_obstacle_distances = [1 * np.ones(len(obs["obstacles_pos"])) for _ in range(len(obs["gates_pos"]))] ## changed
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

    def _setup_casadi_solver(self):
        """Set up the CasADi solver for MPC."""
        # State dimensions
        nx = 14
        # Input dimensions
        nu = 4

        # Define CasADi symbolic variables
        self.x_sym = ca.SX.sym('x', nx)
        self.u_sym = ca.SX.sym('u', nu)

        # Unpack state variables
        px, py, pz = self.x_sym[0], self.x_sym[1], self.x_sym[2]
        vx, vy, vz = self.x_sym[3], self.x_sym[4], self.x_sym[5]
        roll, pitch, yaw = self.x_sym[6], self.x_sym[7], self.x_sym[8]
        f_collective = self.x_sym[9]
        f_collective_cmd = self.x_sym[10]
        r_cmd, p_cmd, y_cmd = self.x_sym[11], self.x_sym[12], self.x_sym[13]
        

        # Unpack control inputs
        df_cmd, dr_cmd, dp_cmd, dy_cmd = self.u_sym[0], self.u_sym[1], self.u_sym[2], self.u_sym[3]

        # Define system dynamics
        dynamics = ca.vertcat(
            vx,
            vy,
            vz,
            (self.params_acc[0] * f_collective + self.params_acc[1])
            * (ca.cos(roll) * ca.sin(pitch) * ca.cos(yaw) + ca.sin(roll) * ca.sin(yaw)),
            (self.params_acc[0] * f_collective + self.params_acc[1])
            * (ca.cos(roll) * ca.sin(pitch) * ca.sin(yaw) - ca.sin(roll) * ca.cos(yaw)),
            (self.params_acc[0] * f_collective + self.params_acc[1]) * ca.cos(roll) * ca.cos(pitch) - self.GRAVITY,
            self.params_roll_rate[0] * roll + self.params_roll_rate[1] * r_cmd,
            self.params_pitch_rate[0] * pitch + self.params_pitch_rate[1] * p_cmd,
            self.params_yaw_rate[0] * yaw + self.params_yaw_rate[1] * y_cmd,
            10.0 * (f_collective_cmd - f_collective),
            df_cmd,
            dr_cmd,
            dp_cmd,
            dy_cmd,
        )

        # Create a continuous time dynamics function
        self.f_cont = ca.Function('f_cont', [self.x_sym, self.u_sym], [dynamics])

        # Create the discrete time dynamics using RK4 integration
        self._create_discrete_dynamics()

        # Create the optimization problem
        self._create_optimization_problem()

    def _create_discrete_dynamics(self):
        """Create discrete time dynamics using RK4 integration."""
        # Implement RK4 integration for the system dynamics
        x_next = self.x_sym
        k1 = self.f_cont(self.x_sym, self.u_sym)
        k2 = self.f_cont(self.x_sym + self.dt/2 * k1, self.u_sym)
        k3 = self.f_cont(self.x_sym + self.dt/2 * k2, self.u_sym)
        k4 = self.f_cont(self.x_sym + self.dt * k3, self.u_sym)
        x_next = self.x_sym + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Create a discrete time dynamics function
        self.f_disc = ca.Function('f_disc', [self.x_sym, self.u_sym], [x_next])

    def _create_optimization_problem(self):
        """Create the MPC optimization problem."""
        nx = self.x_sym.shape[0]
        nu = self.u_sym.shape[0]

        # Define optimization variables
        self.X = ca.SX.sym('X', nx, self.N+1)  # States over the horizon
        self.U = ca.SX.sym('U', nu, self.N)    # Controls over the horizon

        # Define weight matrices for the cost function (same as in the original code)
        # Q = np.diag([
        #     10.0, 10.0, 10.0,   # Position
        #     0.01, 0.01, 0.01,   # Velocity
        #     0.1, 0.1, 0.1,      # rpy - roll, pitch, yaw
        #     0.01, 0.01,         # f_collective, f_collective_cmd
        #     0.01, 0.01, 0.01,   # rpy_cmd
        # ])
        # R = np.diag([0.01, 0.01, 0.01, 0.01])  # Input cost

        # Define weight matrices for the cost function
        Q = np.diag([
            5.0, 5.0, 5.0,       # Position weights
            0.3, 0.3, 0.3,       # Velocity weights
            0.8, 0.8, 0.8,       # rpy weights
            0.02, 0.02,          # f_collective, f_collective_cmd
            0.05, 0.05, 0.05,    # rpy_cmd
        ])
        R = np.diag([0.05, 0.05, 0.05, 0.05]) # Input cost
        
        Q_e = Q.copy()  # Terminal cost

        # Define the objective function
        objective = 0
        g = []  # Constraints
        lbg = []  # Lower bounds of constraints
        ubg = []  # Upper bounds of constraints

        # Initial constraints
        g.append(self.X[:, 0] - self.x_sym)
        lbg.extend([0] * nx)
        ubg.extend([0] * nx)

        # Define reference trajectory placeholder
        self.y_ref_sym = ca.SX.sym('y_ref', nx, self.N)  # Reference trajectory for each step
        self.y_ref_N_sym = ca.SX.sym('y_ref_N', nx)      # Terminal reference

        # Formulate the MPC problem
        for k in range(self.N):
            # Cost for states and inputs
            state_error = self.X[:, k] - self.y_ref_sym[:, k]
            objective += state_error.T @ Q @ state_error + self.U[:, k].T @ R @ self.U[:, k]

            # Dynamic constraints
            x_next = self.f_disc(self.X[:, k], self.U[:, k])
            g.append(self.X[:, k+1] - x_next)
            lbg.extend([0] * nx)
            ubg.extend([0] * nx)

        # Terminal cost
        state_error = self.X[:, self.N] - self.y_ref_N_sym
        objective += state_error.T @ Q_e @ state_error

        # State constraints (collective thrust and attitude commands)
        for k in range(1, self.N+1):
            g.append(self.X[9:14, k])  # Constraints on f_collective, f_cmd, rpy_cmd
            lbg.extend([0.1, 0.1, -1.57, -1.57, -1.57])  # Lower bounds
            ubg.extend([0.55, 0.55, 1.57, 1.57, 1.57])   # Upper bounds

        # Limit rate of change of control inputs
        if self.N > 1:
            for k in range(1, self.N):
                # Constrain change in control inputs
                delta_u = self.U[:, k] - self.U[:, k-1]
                g.append(delta_u)
                max_rate = 0.1  # Maximum allowed rate of change
                lbg.extend([-max_rate, -max_rate, -max_rate, -max_rate])
                ubg.extend([max_rate, max_rate, max_rate, max_rate])

        # Add soft constraints for obstacle avoidance
        safety_radius = 0.3  # 30cm minimum distance
        obstacle_weight = 100.0  # Weight for obstacle avoidance in cost function
            
        # For each obstacle, add soft constraint to cost function
        for obs_pos in self.config.env.track["obstacles"]:
            obs_pos_array = np.array(obs_pos["pos"])
            
            for k in range(1, self.N+1):
                # For each prediction step, compute distance to obstacle
                pos_k = self.X[:3, k]
                dist_to_obs = ca.norm_2(pos_k - obs_pos_array)
                
                # Add soft constraint to cost function
                # This increases cost when drone gets too close to obstacles
                penalty = ca.fmax(0, safety_radius - dist_to_obs)
                objective += obstacle_weight * penalty * penalty

        # Define parameters for the solver
        self.opt_vars = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        self.opt_params = ca.vertcat(
            self.x_sym,                        # Current state
            ca.reshape(self.y_ref_sym, -1, 1), # Reference trajectory
            self.y_ref_N_sym                   # Terminal reference
        )

        # Store constraints and their bounds
        self.constraints = ca.vertcat(*g)
        self.lbg = np.array(lbg)  # Convert directly to numpy array
        self.ubg = np.array(ubg)  # Convert directly to numpy array
        
        # Create the NLP problem
        nlp = {
            'x': self.opt_vars,
            'f': objective,
            'g': self.constraints,
            'p': self.opt_params
        }

        # Set solver options
        opts = {
            'ipopt': {
                'max_iter': 50,
                'print_level': 0,
                'acceptable_tol': 1e-5,
                'acceptable_obj_change_tol': 1e-5
            },
            'print_time': 0
        }

        # Create the solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Prepare variable bounds
        self.lbx = []
        self.ubx = []

        # State constraints (only for bounds checking, dynamics enforced through equality constraints)
        for _ in range(self.N+1):
            self.lbx.extend([-np.inf] * nx)  # Lower bounds
            self.ubx.extend([np.inf] * nx)   # Upper bounds

        # Input constraints
        for _ in range(self.N):
            self.lbx.extend([-10.0] * nu)  # Lower bounds
            self.ubx.extend([10.0] * nu)   # Upper bounds
            
        # Convert bounds to numpy arrays
        self.lbx = np.array(self.lbx)
        self.ubx = np.array(self.ubx)

        # Initialize solution for warm start
        self.x_init = np.zeros((nx * (self.N + 1) + nu * self.N, 1))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control with visualization of gate passages"""
        start_solve_time = time.time()
        
        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True

        # Get current state
        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        # Combine state variables
        x_current = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )

        # Create reference trajectory for the horizon
        y_ref = np.zeros((14, self.N))
        for j in range(self.N):
            ref_state = np.array([
                self.x_des[i + j],
                self.y_des[i + j],
                self.z_des[i + j],
                0.0, 0.0, 0.0,  # Velocity
                0.0, 0.0, 0.0,  # rpy
                0.35, 0.35,     # f_collective, f_cmd
                0.0, 0.0, 0.0,  # rpy_cmd
            ])
            y_ref[:, j] = ref_state
        
        # Terminal reference
        y_ref_N = np.array([
            self.x_des[i + self.N],
            self.y_des[i + self.N],
            self.z_des[i + self.N],
            0.0, 0.0, 0.0,  # Velocity
            0.0, 0.0, 0.0,  # rpy
            0.35, 0.35,     # f_collective, f_cmd
            0.0, 0.0, 0.0,  # rpy_cmd
        ])

        # Prepare parameters for the solver
        p = np.concatenate([x_current, y_ref.flatten(), y_ref_N])

        # Solve the optimization problem
        sol = self.solver(
            x0=self.x_init,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p
        )

        # Extract the solution
        solution = np.array(sol['x']).flatten()
        
        # Update the initial solution for warm start
        self.x_init = solution

        # Extract the optimal control inputs and states
        x_opt = solution[:14 * (self.N + 1)].reshape(self.N + 1, 14)
        u_opt = solution[14 * (self.N + 1):].reshape(self.N, 4)

        # Store the optimal trajectory as a class attribute for visualization
        self.X_opt = x_opt.T  # Transpose to get shape [state_dim, horizon_length]

        # Extract the first control to apply (MPC strategy)
        optimal_control = u_opt[0]
        
        # Extract the next state for smoothing
        x1 = x_opt[1]

        # Smooth the control
        w = 0.7 / self.config.env.freq / self.dt  # Reduce weight factor for smoother transition
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = self.last_f_cmd * (1 - w) + x1[10] * w
        self.last_rpy_cmd = self.last_rpy_cmd * (1 - w) + x1[11:14] * w

        # Extract the commanded values [f_collective_cmd, r_cmd, p_cmd, y_cmd]
        cmd = x1[10:14]

        # Create the full 13-dimensional control vector
        # The state_control interface in crazyflow expects a 13-dimensional control vector:
        # [pos_des (3), vel_des (3), acc_des (3), yaw_des (1), thrust_des (1), rates_des (2)]
        full_cmd = np.zeros(13)
        
        # Set thrust and attitude commands
        # Position at index 10 for thrust (f_cmd)
        full_cmd[10] = cmd[0]  # Thrust command
        # Positions 11-12 for angular rates (roll and pitch rates)
        full_cmd[11] = cmd[1]  # Roll rate command
        full_cmd[12] = cmd[2]  # Pitch rate command
        # Position 9 for yaw angle
        full_cmd[9] = cmd[3]   # Yaw command
        
        # Extract only the attitude control components (thrust, roll, pitch, yaw)
        attitude_cmd = np.zeros(4)
        attitude_cmd[0] = cmd[0]  # Thrust command
        attitude_cmd[1] = cmd[1]  # Roll rate command
        attitude_cmd[2] = cmd[2]  # Pitch rate command
        attitude_cmd[3] = cmd[3]  # Yaw rate command
        
        # print("---------------------------")
        # print(f"Original roll/pitch: {attitude_cmd[1]:.3f}, {attitude_cmd[2]:.3f}")
        
        # # Modify commands
        # attitude_cmd[1] += 0.2  # Add roll bias
        # attitude_cmd[2] += 0.1  # Add pitch bias 
        
        # print(f"Modified roll/pitch: {attitude_cmd[1]:.3f}, {attitude_cmd[2]:.3f}")
        
        # For visualization & logging, keep using full_cmd
        solve_time = time.time() - start_solve_time
        self.mpc_solve_times.append(solve_time)
        
        # Initialize visualization if not already done
        if self.enable_visualization and not self.visualization_initialized:
            self._initialize_visualization()
        
        # Update flight data using the full command for consistent visualization
        self._update_flight_data(obs, full_cmd, time.time())
        
        # Return the 4D attitude command with the correct shape
        return attitude_cmd.reshape(1, 1, 4)

    def step_callback(
        self, action, obs, reward, terminated, truncated, info
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        
        # Generate report if flight is finished
        if terminated or truncated:
            self.generate_flight_report()
            self.finished = True
            
        return self.finished

    def episode_callback(self):
        """Reset the controller for a new episode."""
        self._tick = 0
        self.finished = False

    def get_predicted_trajectory(self):
        """Return the MPC's predicted trajectory over the horizon."""
        # For CasADi implementation
        if hasattr(self, 'X_opt'):
            # Extract position states (first 3 states are x,y,z)
            return self.X_opt[:3, :].T  # Return with shape [N, 3]
        return None
    
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
            f"\n===== Flight Report =====\n"
            f"Total flight time: {total_time:.2f} seconds\n"
            f"Gates passed: {gates_passed}/{len(self.gate_passed)}\n"
            f"Average tracking error: {avg_error:.4f} m\n"
            f"Maximum tracking error: {max_error:.4f} m\n"
            f"Average MPC solve time: {avg_solve_time*1000:.2f} ms\n"
            f"Minimum obstacle distance: {min_obstacle_dist:.4f} m\n"
            f"Flight log saved to: {self.log_file}\n"
            f"=========================="
        )
        
        # If all gates were passed, add gate timing information
        if all(self.gate_passed):
            report += "\n\nGate Timing Information:\n"
            for i, time in enumerate(self.gate_times):
                report += f"  Gate {i+1}: {time:.2f} s"
                if i > 0:
                    report += f" (Interval: {time - self.gate_times[i-1]:.2f} s)"
                report += "\n"
        
        print(report)
        
        # Stop the animation if it exists
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
        
        # Create a static figure with the final trajectory
        self._create_final_visualization()
        
        return report

    def _create_final_visualization(self):
        """Create a static visualization of the complete flight trajectory."""
        if not self.trajectory_history:
            return
            
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle("Complete Flight Trajectory", fontsize=16)
        
        # Create gridspec layout
        gs = fig.add_gridspec(2, 3)
        
        # 3D trajectory plot
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[0, 2])
        ax_metrics = fig.add_subplot(gs[1, 1:])
        
        # Extract trajectory data
        traj_array = np.array(self.trajectory_history)
        times = np.array(self.time_history)
        errors = np.array(self.reference_error_history)
        
        # Plot trajectories
        ax_3d.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'b-', linewidth=2, label='Actual Trajectory')
        ax_3d.plot(self.x_des, self.y_des, self.z_des, 'r--', linewidth=1, alpha=0.7, label='Planned Trajectory')
        
        # Plot environment
        self._plot_environment(ax_3d)
        
        # Plot gate passage points
        for i, gate_time in enumerate(self.gate_times):
            if gate_time is not None:
                idx = min(int(gate_time * self.freq), len(traj_array)-1)
                if idx >= 0:
                    ax_3d.scatter(traj_array[idx, 0], traj_array[idx, 1], traj_array[idx, 2], 
                                 color='green', s=100, marker='*', 
                                 label=f'Gate {i+1} Pass' if i == 0 else "")
        
        # Plot 2D projections
        ax_xy.plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=2, label='Actual')
        ax_xy.plot(self.x_des, self.y_des, 'r--', linewidth=1, alpha=0.7, label='Planned')
        
        ax_xz.plot(traj_array[:, 0], traj_array[:, 2], 'b-', linewidth=2, label='Actual')
        ax_xz.plot(self.x_des, self.z_des, 'r--', linewidth=1, alpha=0.7, label='Planned')
        
        # Plot metrics
        ax_metrics.plot(times, errors, 'g-', linewidth=1.5, label='Position Error (m)')
        ax_metrics.plot(times, [t[9] for t in self.trajectory_history], 'r-', linewidth=1.5, label='Thrust')
        
        # Plot gate passage times
        for i, gate_time in enumerate(self.gate_times):
            if gate_time is not None:
                ax_metrics.axvline(x=gate_time, color='g', linestyle='--', alpha=0.7,
                                 label=f'Gate {i+1} Pass' if i == 0 else "")
    
        # Add labels
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Flight Trajectory')
        
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('Top View (XY Plane)')
        ax_xy.grid(True)
        
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        ax_xz.set_title('Side View (XZ Plane)')
        ax_xz.grid(True)
        
        ax_metrics.set_xlabel('Time (s)')
        ax_metrics.set_title('Flight Metrics')
        ax_metrics.grid(True)
        
        # Set equal aspect ratio
        ax_xy.set_aspect('equal')
        ax_xz.set_aspect('equal')
        
        # Set limits based on data
        pad = 0.5  # Add padding to the plot limits
        x_min, x_max = min(min(self.x_des), min(traj_array[:, 0])) - pad, max(max(self.x_des), max(traj_array[:, 0])) + pad
        y_min, y_max = min(min(self.y_des), min(traj_array[:, 1])) - pad, max(max(self.y_des), max(traj_array[:, 1])) + pad
        z_min, z_max = max(0, min(min(self.z_des), min(traj_array[:, 2])) - pad), max(max(self.z_des), max(traj_array[:, 2])) + pad
        
        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])
        
        ax_xy.set_xlim([x_min, x_max])
        ax_xy.set_ylim([y_min, y_max])
        
        ax_xz.set_xlim([x_min, x_max])
        ax_xz.set_ylim([z_min, z_max])
        
        if len(times) > 1:
            ax_metrics.set_xlim([0, max(times)])
            if len(errors) > 0:
                ax_metrics.set_ylim([0, max(max(errors) * 1.1, 1.0)])
        
        # Add legends
        ax_3d.legend(loc='upper right')
        ax_xy.legend(loc='upper right')
        ax_xz.legend(loc='upper right')
        ax_metrics.legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show(block=True)  # This will block until the window is closed

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
        
        # Store data
        self.trajectory_history.append(np.concatenate([pos, vel, rpy, [cmd[10]]])) # pos, vel, rpy, thrust
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
                    # Check if drone is passing through the gate plane in the correct direction
                    # by calculating the dot product between the gate normal and gate-to-drone vector
                    dot_product = np.dot(normal, gate_to_drone)
                    
                    # Check drone's velocity direction aligns with gate orientation
                    vel_alignment = np.dot(normal, obs["vel"])
                    
                    # Gate is considered passed if:
                    # 1. Close to gate center
                    # 2. Velocity is pointing in roughly the right direction
                    if vel_alignment > 0.5:  # Moving in approximately the correct direction
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
                    f"{pos_error:.4f},{cmd[10]:.4f},{gate_idx}\n")