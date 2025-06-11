"""Trajectory planning module for drone racing.

This module handles waypoint generation, trajectory planning through gates,
and velocity-aware trajectory updates for smooth flight paths.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TrajectoryPlanner:
    """Handles trajectory planning and waypoint generation for drone racing."""

    def __init__(self, config: dict, logger: any):
        """Initialize the trajectory planner."""
        self.config = config
        self.logger = logger  # This is now the FlightLogger instance

        # Current target gate tracking
        self.current_target_gate_idx = 0

        # Trajectory storage
        self.current_trajectory = None
        self.freq = config.env.freq

    def generate_trajectory_from_waypoints(
        self,
        waypoints: np.ndarray,
        target_gate_idx: int,
        use_velocity_aware: bool = False,
        current_vel: np.ndarray = None,
        tick: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate trajectory with optional velocity-aware smooth transitions."""
        if len(waypoints) < 2:
            self.logger.log_warning("Not enough waypoints to generate trajectory", tick)
            return np.array([]), np.array([]), np.array([])

        # Only use velocity-aware generation if explicitly requested (for replanning)
        if use_velocity_aware and current_vel is not None:
            current_speed = np.linalg.norm(current_vel)

            trajectory_info = {
                "type": "velocity_aware_generation",
                "current_speed": current_speed,
                "target_gate": target_gate_idx,
            }
            self.logger.log_trajectory_update(trajectory_info, tick)

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

        # Standard trajectory generation (for initial and velocity-aware)
        # Calculate distances between waypoints
        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])

        # Create time parameterization
        if use_velocity_aware and current_vel is not None:
            current_speed = np.linalg.norm(current_vel)
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

            speed_info = {
                "mode": "velocity_aware",
                "cruise_speed": cruise_speed,
                "current_speed": current_speed,
                "num_waypoints": len(waypoints),
            }
            self.logger.log_trajectory_update(speed_info, tick)
        else:
            # Standard speeds for initial trajectory
            speeds = [1.2] * len(waypoints)
            speed_info = {
                "mode": "standard",
                "uniform_speed": speeds[0],
                "num_waypoints": len(waypoints),
            }
            self.logger.log_trajectory_update(speed_info, tick)

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

        timing_info = {
            "total_time": total_time,
            "min_duration": min_duration,
            "total_distance": distances[-1],
        }
        self.logger.log_trajectory_update(timing_info, tick)

        # Normalize time for spline
        time_normalized = np.array(time_points) / max(time_points[-1], 1.0)

        # Create splines with velocity boundary conditions only for replanning
        if use_velocity_aware and current_vel is not None and np.linalg.norm(current_vel) > 0.3:
            try:
                bc_type_x = ((1, current_vel[0]), (2, 0))
                bc_type_y = ((1, current_vel[1]), (2, 0))
                bc_type_z = ((1, current_vel[2]), (2, 0))

                cs_x = CubicSpline(time_normalized, waypoints[:, 0], bc_type=bc_type_x)
                cs_y = CubicSpline(time_normalized, waypoints[:, 1], bc_type=bc_type_y)
                cs_z = CubicSpline(time_normalized, waypoints[:, 2], bc_type=bc_type_z)

                boundary_info = {
                    "boundary_conditions_applied": True,
                    "velocity": [current_vel[0], current_vel[1], current_vel[2]],
                }
                self.logger.log_trajectory_update(boundary_info, tick)
            except Exception as e:
                self.logger.log_warning(
                    f"Velocity boundary conditions failed, using natural: {str(e)}", tick
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
        min_points = max(int(trajectory_freq * min_duration), tick + 3 * 30)  # Assuming N=30

        ts = np.linspace(0, 1, min_points)
        x_des = cs_x(ts)
        y_des = cs_y(ts)
        z_des = cs_z(ts)

        # Add terminal points for stability
        extra_points = 3 * 30 + 1  # Assuming N=30
        x_des = np.concatenate((x_des, [x_des[-1]] * extra_points))
        y_des = np.concatenate((y_des, [y_des[-1]] * extra_points))
        z_des = np.concatenate((z_des, [z_des[-1]] * extra_points))

        # Save trajectory
        trajectory = {
            "tick": tick,
            "waypoints": waypoints.copy(),
            "x": x_des.copy(),
            "y": y_des.copy(),
            "z": z_des.copy(),
            "timestamp": time.time(),
        }

        self.current_trajectory = trajectory
        self.save_trajectories_to_file(f"flight_logs/{target_gate_idx}_trajectories.npz")

        # Log final trajectory generation summary
        final_info = {
            "mode": "velocity-aware" if use_velocity_aware else "standard",
            "waypoints": len(waypoints),
            "points": len(x_des),
            "target_gate": target_gate_idx,
        }
        self.logger.log_trajectory_update(final_info, tick)

        return x_des, y_des, z_des

    def update_trajectory_with_correction(
        self,
        new_waypoints: np.ndarray,
        old_trajectory: tuple[np.ndarray, np.ndarray, np.ndarray],
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_tick: int,
        trajectory_start_tick: int,
        N: int,
        target_gate_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update trajectory with gentle correction - more conservative approach."""
        if len(new_waypoints) < 2:
            self.logger.log_warning("Not enough waypoints to update trajectory", current_tick)
            return old_trajectory

        # Store old trajectory for blending
        old_x_des, old_y_des, old_z_des = old_trajectory
        current_traj_idx = max(0, current_tick - trajectory_start_tick)

        current_speed = np.linalg.norm(current_vel)

        # GENTLE CORRECTION: Conservative but effective
        if current_speed > 0.4:  # Higher speed threshold
            # Calculate trajectory error
            trajectory_error = np.linalg.norm(current_pos - new_waypoints[1])

            correction_info = {
                "type": "gentle_correction",
                "trajectory_error": trajectory_error,
                "current_speed": current_speed,
                "correction_threshold": 0.20,
            }
            self.logger.log_trajectory_update(correction_info, current_tick)

            # Create waypoints with gentle correction
            corrected_waypoints = [current_pos]

            # Apply correction only for larger errors
            if trajectory_error > 0.20:  # Higher threshold - less sensitive
                # Conservative parameters
                correction_steps = 2  # Fewer correction points

                # Calculate correction direction
                target_point = new_waypoints[1] if len(new_waypoints) > 1 else new_waypoints[0]

                # Create gentle correction sequence
                for i in range(1, correction_steps + 1):
                    alpha = i / correction_steps

                    # Very gentle correction factor - mostly linear
                    correction_factor = alpha  # Linear, no acceleration

                    # STRONG velocity influence - preserve momentum
                    vel_influence = 0.6 * (1 - alpha * 0.3)  # Strong, gradual decrease
                    vel_normalized = current_vel / current_speed
                    velocity_contribution = vel_normalized * (
                        current_speed * 0.4 * vel_influence  # Larger velocity contribution
                    )

                    # Gentle trajectory point with strong velocity preservation
                    corrected_point = (
                        current_pos * (1 - correction_factor)
                        + target_point * correction_factor
                        + velocity_contribution
                    )

                    corrected_waypoints.append(corrected_point)

                # Add remaining waypoints
                corrected_waypoints.extend(new_waypoints[2:] if len(new_waypoints) > 2 else [])

                applied_correction_info = {
                    "correction_steps": correction_steps,
                    "trajectory_error": trajectory_error,
                    "current_speed": current_speed,
                    "correction_applied": True,
                }
                self.logger.log_trajectory_update(applied_correction_info, current_tick)
            else:
                # For smaller errors, use very gentle velocity-aware approach
                vel_continuation_time = 0.6  # Longer continuation time
                vel_continuation_dist = current_speed * vel_continuation_time
                vel_continuation_dist = min(vel_continuation_dist, 0.8)  # Higher cap

                if current_speed > 0.3:
                    vel_normalized = current_vel / current_speed
                    velocity_point = current_pos + vel_normalized * vel_continuation_dist
                    corrected_waypoints.append(velocity_point)

                    # Very smooth transition to new trajectory
                    if len(new_waypoints) > 1:
                        target_point = new_waypoints[1]
                        # Conservative blending - favor velocity continuation
                        transition_point = velocity_point * 0.8 + target_point * 0.2
                        corrected_waypoints.append(transition_point)

                # Add remaining waypoints
                corrected_waypoints.extend(new_waypoints[1:])

                gentle_correction_info = {
                    "vel_continuation_dist": vel_continuation_dist,
                    "gentle_correction_applied": True,
                    "trajectory_error": trajectory_error,
                }
                self.logger.log_trajectory_update(gentle_correction_info, current_tick)

            new_waypoints = np.array(corrected_waypoints)

        # Generate new trajectory
        x_des, y_des, z_des = self.generate_trajectory_from_waypoints(
            new_waypoints,
            target_gate_idx,
            use_velocity_aware=True,
            current_vel=current_vel,
            tick=current_tick,
        )

        # GENTLE trajectory blending
        if old_x_des is not None and len(old_x_des) > current_traj_idx:
            blend_horizon = min(N * 2 // 3, len(x_des))  # Longer blending window

            for i in range(blend_horizon):
                old_idx = current_traj_idx + i
                if old_idx < len(old_x_des) and i < len(x_des):
                    # Gentle blending - slow, smooth transition
                    alpha = 1 - np.exp(-2 * i / blend_horizon)  # Slower transition rate

                    x_des[i] = old_x_des[old_idx] * (1 - alpha) + x_des[i] * alpha
                    y_des[i] = old_y_des[old_idx] * (1 - alpha) + y_des[i] * alpha
                    z_des[i] = old_z_des[old_idx] * (1 - alpha) + z_des[i] * alpha

            blending_info = {"blend_horizon": blend_horizon, "blending_applied": True}
            self.logger.log_trajectory_update(blending_info, current_tick)

        final_update_info = {
            "trajectory_length": len(x_des),
            "update_tick": current_tick,
            "update_successful": True,
        }
        self.logger.log_trajectory_update(final_update_info, current_tick)

        return x_des, y_des, z_des

    def generate_waypoints(
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
                approach_point[2] += 0.4
                waypoints.append(approach_point)

        return np.array(waypoints)

    def _get_gate_info(self, gate: dict) -> dict:
        """Extract gate information including position, orientation, and dimensions."""
        try:
            # Ensure we're working with a valid dictionary
            if not isinstance(gate, dict):
                self.logger.log_warning(f"Gate is not a dictionary: {gate}", 0)
                gate = {"pos": [0.0, 0.0, 1.0]}

            # Extract position
            if "pos" not in gate:
                self.logger.log_warning(f"Gate missing position key: {gate}", 0)
                pos = np.array([0.0, 0.0, 1.0])
            elif gate["pos"] is None:
                self.logger.log_warning(f"Gate position is None: {gate}", 0)
                pos = np.array([0.0, 0.0, 1.0])
            else:
                # Handle different position formats
                raw_pos = gate["pos"]
                if isinstance(raw_pos, dict) and "pos" in raw_pos:
                    # Handle nested dictionary case
                    self.logger.log_warning(f"Converting nested gate dictionary: {raw_pos}", 0)
                    raw_pos = raw_pos["pos"]

                if isinstance(raw_pos, (list, tuple, np.ndarray)) and len(raw_pos) == 3:
                    pos = np.array(raw_pos)
                elif np.isscalar(raw_pos):
                    self.logger.log_warning(
                        f"Gate position is scalar {raw_pos}, using as z-coordinate", 0
                    )
                    pos = np.array([0.0, 0.0, float(raw_pos)])
                else:
                    self.logger.log_warning(
                        f"Invalid position format: {type(raw_pos)}, value: {raw_pos}", 0
                    )
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
            self.logger.log_error(f"Error processing gate: {e}", 0)
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

    def save_trajectories_to_file(self, filename: str = "trajectories.npz") -> bool:
        """Save the current trajectory to a file for later analysis."""
        try:
            # Check if we have a trajectory to save
            if self.current_trajectory is None:
                self.logger.log_warning("No trajectory to save", 0)
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
                        self.logger.log_warning(
                            f"Invalid waypoints array shape: {waypoints.shape}", 0
                        )
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
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            np.savez(filename, **data)
            return True
        except Exception as e:
            self.logger.log_error(f"Error saving trajectory: {e}", 0)
            return False
