"""MPC Controller for drone racing using attitude control interface.

This module contains the main MPC controller class that coordinates trajectory planning,
gate detection, and control execution for autonomous drone racing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.acados_model import create_ocp_solver
from lsy_drone_racing.control.logging_setup import FlightLogger
from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPController(Controller):
    """MPC using the collective thrust and attitude interface for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller."""
        super().__init__(obs, info, config)

        # Setup enhanced logging with no console output
        self.flight_logger = FlightLogger("MPController")

        # Basic configuration
        self.freq = config.env.freq
        self._tick = 0
        self.start_pos = obs["pos"]
        self.config = config

        # MPC parameters
        self.N = 30
        self.T_HORIZON = 1.5
        self.dt = self.T_HORIZON / self.N

        # Replanning configuration
        self.replanning_frequency = 1
        self.last_replanning_tick = 0

        # Approach parameters for different gates
        self.approach_dist = [0.2, 0.35, 0.25, 0.01]
        self.exit_dist = [1.2, 0.35, 0.5, 1.2]
        self.default_approach_dist = 0.1
        self.default_exit_dist = 0.5

        # Height offset parameters
        self.approach_height_offset = [0.01, 0.1, -0.2, 0.1]
        self.exit_height_offset = [0.1, 0.1, 0.4, 0.1]
        self.default_approach_height_offset = 0.1
        self.default_exit_height_offset = 0.0

        # Initialize target gate tracking
        self.current_target_gate_idx = 0

        # Create the MPC solver
        self.mpc_weights = {
            "Q_pos": 2.9,  # Position tracking weight -2.3
            "Q_vel": 0.1,  # Velocity tracking weight
            "Q_rpy": 0.1,  # Attitude (roll/pitch/yaw) weight
            "Q_thrust": 0.01,  # Collective thrust weight
            "Q_cmd": 0.01,  # Command tracking weight
            "R": 0.007,  # Control input regularization weight
        }
        self.acados_ocp_solver, self.ocp = create_ocp_solver(
            self.T_HORIZON, self.N, self.mpc_weights
        )

        # Controller state variables
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.finished = False

        # Trajectory tracking
        self.updated_gates = set()
        self.saved_trajectories = []
        self.trajectory_metadata = []
        self.current_trajectory = None

        # Episode statistics
        self.gates_passed = 0
        self.total_gates = len(config.env.track["gates"])
        self.flight_successful = False

        # Initialize trajectory planner with correct argument order
        self.trajectory_planner = TrajectoryPlanner(self.config, self.flight_logger)

        # Store parameters directly in the trajectory planner if needed
        self.trajectory_planner.freq = self.freq
        self.trajectory_planner.approach_dist = self.approach_dist
        self.trajectory_planner.exit_dist = self.exit_dist
        self.trajectory_planner.approach_height_offset = self.approach_height_offset
        self.trajectory_planner.exit_height_offset = self.exit_height_offset
        self.trajectory_planner.default_approach_dist = self.default_approach_dist
        self.trajectory_planner.default_exit_dist = self.default_exit_dist
        self.trajectory_planner.default_approach_height_offset = self.default_approach_height_offset
        self.trajectory_planner.default_exit_height_offset = self.default_exit_height_offset

        # Log initialization parameters
        controller_params = {
            "MPC_N": self.N,
            "MPC_T_HORIZON": self.T_HORIZON,
            "MPC_dt": self.dt,
            "MPC_weights": self.mpc_weights,
            "approach_distances": self.approach_dist,
            "exit_distances": self.exit_dist,
            "approach_height_offsets": self.approach_height_offset,
            "exit_height_offsets": self.exit_height_offset,
            "start_position": self.start_pos.tolist(),
            "total_gates": self.total_gates,
            "freq": self.freq,
            "replanning_frequency": self.replanning_frequency,
            "default_approach_dist": self.default_approach_dist,
            "default_exit_dist": self.default_exit_dist,
            "default_approach_height_offset": self.default_approach_height_offset,
            "default_exit_height_offset": self.default_exit_height_offset,
        }
        self.flight_logger.log_initialization(controller_params, self._tick)

        # Initialize trajectory
        self.waypoints = self.trajectory_planner.generate_waypoints(obs, 0)
        x_des, y_des, z_des = self.trajectory_planner.generate_trajectory_from_waypoints(
            self.waypoints, 0, use_velocity_aware=False
        )
        self._trajectory_start_tick = 0

        # Store trajectory references
        self.x_des = x_des
        self.y_des = y_des
        self.z_des = z_des

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Execute MPC control with replanning capabilities."""
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
            self.flight_logger.log_gate_progress(
                self.gates_passed, self.total_gates, current_target_gate, self._tick
            )
        elif current_target_gate > self.gates_passed:
            self.gates_passed = current_target_gate
            self.flight_logger.log_gate_progress(
                self.gates_passed, self.total_gates, current_target_gate, self._tick
            )

        # Check for replanning
        just_replanned = self._check_and_execute_replanning(obs, current_target_gate)

        # Execute MPC control
        return self._execute_mpc_control(obs, current_target_gate, just_replanned)

    def _check_and_execute_replanning(self, obs: dict, current_target_gate: int) -> bool:
        """Check if replanning is needed and execute if necessary."""
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
                        replan_info = {
                            "gate_idx": target_gate_idx,
                            "horizontal_diff": horizontal_diff,
                            "config_pos": f"[{config_pos[0]:.3f}, {config_pos[1]:.3f}]",
                            "observed_pos": f"[{observed_pos[0]:.3f}, {observed_pos[1]:.3f}]",
                        }
                        self.flight_logger.log_replanning_event(replan_info, self._tick)

                        should_replan = True
                        just_replanned = True
                        self.updated_gates.add(target_gate_idx)

                if should_replan:
                    # Generate NEW waypoints with the updated gate position
                    new_waypoints = self.trajectory_planner.generate_waypoints(
                        obs,
                        target_gate_idx,  # Fixed: removed config parameter
                    )

                    trajectory_info = {
                        "mode": "velocity-aware",
                        "waypoints": len(new_waypoints),
                        "target_gate": target_gate_idx,
                    }
                    self.flight_logger.log_trajectory_update(trajectory_info, self._tick)

                    # Generate new trajectory with velocity-aware correction
                    new_x_des, new_y_des, new_z_des = (
                        self.trajectory_planner.generate_trajectory_from_waypoints(
                            new_waypoints,
                            target_gate_idx,
                            use_velocity_aware=True,
                            current_vel=self._current_vel,
                            tick=self._tick,
                        )
                    )

                    # Update local trajectory references
                    self.x_des = new_x_des
                    self.y_des = new_y_des
                    self.z_des = new_z_des
                    self._trajectory_start_tick = self._tick

        return just_replanned

    def _execute_mpc_control(
        self, obs: dict, current_target_gate: int, just_replanned: bool
    ) -> NDArray[np.floating]:
        """Execute the MPC optimization and return control commands."""
        # Execute MPC control with trajectory offset
        trajectory_offset = getattr(self, "_trajectory_start_tick", 0)
        trajectory_index = max(0, self._tick - trajectory_offset)

        i = min(trajectory_index, len(self.x_des) - 1)
        if trajectory_index >= len(self.x_des):
            self.finished = True
            self.flight_logger.log_warning(
                f"Trajectory finished - reached end of trajectory at tick {self._tick}", self._tick
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

        # Set MPC references
        self._set_mpc_references(i, traj_len, just_replanned, tracking_error)

        # Solve MPC and get control
        status = self.acados_ocp_solver.solve()

        if status != 0:
            self.flight_logger.log_warning(
                f"MPC solver failed with status {status} at tick {self._tick}", self._tick
            )

        x1 = self.acados_ocp_solver.get(1, "x")
        _ = self.acados_ocp_solver.get(0, "u")

        # Smooth control updates
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        # Log tracking performance occasionally
        if self._tick % 100 == 0 or just_replanned:
            tracking_info = {
                "error": tracking_error,
                "gates_passed": self.gates_passed,
                "total_gates": self.total_gates,
                "just_replanned": just_replanned,
                "solver_status": status,
            }
            self.flight_logger.log_tracking_performance(tracking_info, self._tick)

        # Return the real control commands
        cmd = x1[10:14]  # [f_collective_cmd, roll_cmd, pitch_cmd, yaw_cmd]
        return cmd

    def _set_mpc_references(
        self, i: int, traj_len: int, just_replanned: bool, tracking_error: float
    ):
        """Set MPC reference trajectory with conservative tracking."""
        # CONSERVATIVE reference tracking
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

            # GENTLE reference with conservative position tracking
            if just_replanned or tracking_error > 0.25:  # Higher threshold
                # Gentle velocity references toward trajectory
                if j < self.N - 1:
                    next_idx = min(idx + 1, traj_len - 1)
                    desired_vel = (
                        np.array(
                            [
                                self.x_des[next_idx] - x_ref,
                                self.y_des[next_idx] - y_ref,
                                self.z_des[next_idx] - z_ref,
                            ]
                        )
                        * self.freq
                    )  # Convert to velocity

                    # Conservative scaling for gentle tracking
                    vel_scale = 1.2 if just_replanned else 1.0
                    desired_vel *= vel_scale

                    # Conservative velocity limit
                    max_vel = 2.0
                    vel_norm = np.linalg.norm(desired_vel)
                    if vel_norm > max_vel:
                        desired_vel = desired_vel / vel_norm * max_vel
                else:
                    desired_vel = np.zeros(3)
            else:
                # Very light velocity bias toward trajectory for normal tracking
                if j < self.N - 1:
                    next_idx = min(idx + 1, traj_len - 1)
                    desired_vel = (
                        np.array(
                            [
                                self.x_des[next_idx] - x_ref,
                                self.y_des[next_idx] - y_ref,
                                self.z_des[next_idx] - z_ref,
                            ]
                        )
                        * self.freq
                        * 0.3  # Very light influence
                    )
                else:
                    desired_vel = np.zeros(3)

            # Base reference with gentle velocity tracking
            yref = np.array(
                [
                    x_ref,
                    y_ref,
                    z_ref,  # Position (3)
                    desired_vel[0],
                    desired_vel[1],
                    desired_vel[2],  # Gentle velocity references (3)
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

            step_info = {
                "reward": reward,
                "gates_passed": self.gates_passed,
                "total_gates": self.total_gates,
                "target_gate": current_target_gate,
                "terminated": terminated,
                "truncated": truncated,
            }
            self.flight_logger.log_step_info(step_info, self._tick)

        return self.finished

    def episode_callback(self):
        """Reset controller state for new episode and log final statistics."""
        # Log final episode statistics
        summary = {
            "flight_successful": self.flight_successful,
            "gates_passed": self.gates_passed,
            "total_gates": self.total_gates,
            "finished": self.finished,
            "freq": self.freq,
            "controller_params": {
                "MPC_N": self.N,
                "MPC_T_HORIZON": self.T_HORIZON,
                "MPC_weights": self.mpc_weights,
                "approach_distances": self.approach_dist,
                "exit_distances": self.exit_dist,
                "replanning_frequency": self.replanning_frequency,
            },
        }
        self.flight_logger.log_episode_summary(summary, self._tick)

        # Reset for next episode
        self._tick = 0
        self.finished = False
        self.updated_gates = set()
        self._trajectory_start_tick = 0
        self.gates_passed = 0
        self.flight_successful = False

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
