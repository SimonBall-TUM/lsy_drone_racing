"""This module handles the collision avoidance by setting up the constraints for the gates and obstacles in the environment."""

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, diag, horzcat, sin, vertcat
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class CollisionAvoidanceHandler:
    """Handles the collision avoidance by setting up the constraints for the gates and obstacles in the environment.

    This class defines ellipsoids representing the gates and infinite cylinders for the obstacles.
    It sets them as constraints in the model / OCP and provides methods to update parameters and draw collision bodies in the environment.
    """

    def __init__(
        self,
        num_gates: int,
        num_obstacles: int,
        gate_length: float,
        ellipsoid_length: float,
        ellipsoid_radius: float,
        obstacle_radius: float,
        ignored_obstacle_indices: list[int] | None = None,
    ):
        """Constructor for the CollisionAvoidanceHandler.

        Args:
            num_gates (int): Number of gates in the environment.
            num_obstacles (int): Number of obstacles in the environment.
            gate_length (float): Length of the gate in meters.
            ellipsoid_length (float): Length of the ellipsoid representing the border of the gates.
            ellipsoid_radius (float): Radius of the ellipsoid representing the border of the gates.
            obstacle_radius (float): Radius of the infinite cylinder representing the obstacles.
            ignored_obstacle_indices (list[int] | None): List of obstacle indices to ignore for collision avoidance. Defaults to None.
        """
        self.num_gates = num_gates
        self.num_obstacles = num_obstacles
        self.gate_length = gate_length
        self.ellipsoid_length = ellipsoid_length
        self.ellipsoid_radius = ellipsoid_radius
        self.obstacle_radius = obstacle_radius
        self.ignored_obstacle_indices = set(ignored_obstacle_indices or [])
        self.obstacle_positions = None
        self.gate_positions = None
        self.gate_rotations = None

        # Calculate active obstacles (those not ignored)
        self.num_active_obstacles = num_obstacles - len(self.ignored_obstacle_indices)
        self.num_constraints = num_gates * 4 + self.num_active_obstacles

        # Midpoints of the four ellipsoids representing the gate
        self.ellipsoid_midpoints = np.array(
            [
                [self.gate_length / 2, 0, 0],
                [0, 0, self.gate_length / 2],
                [-self.gate_length / 2, 0, 0],
                [0, 0, -self.gate_length / 2],
            ]
        )

        # Axes of the four ellipsoids representing the gate
        self.ellipsoid_axes = np.array(
            [
                [self.ellipsoid_radius, self.ellipsoid_radius, self.ellipsoid_length / 2],
                [self.ellipsoid_length / 2, self.ellipsoid_radius, self.ellipsoid_radius],
                [self.ellipsoid_radius, self.ellipsoid_radius, self.ellipsoid_length / 2],
                [self.ellipsoid_length / 2, self.ellipsoid_radius, self.ellipsoid_radius],
            ]
        )

    def setup_model(self, model: AcadosModel):
        """Set up the model with the collision avoidance constraints.

        Args:
            model (AcadosModel): The acados model to which the collision avoidance constraints will be added.
        """
        pos = model.x[:3]

        obs_params, obs_h_expr = self._create_obstacle_expressions(pos)
        gate_params, gate_h_expr = self._create_gate_expressions(pos)

        model.p = vertcat(model.p, *obs_params, *gate_params)
        model.con_h_expr = vertcat(*obs_h_expr, *gate_h_expr)

    def setup_ocp(self, ocp: AcadosOcp):
        """Set up the OCP with the collision avoidance constraints.

        Args:
            ocp (AcadosOCP): The acados OCP to which the collision avoidance constraints will be added.
        """
        ocp.dims.nsh = self.num_constraints

        # Set lower and upper bounds for the constraints
        ocp.constraints.lh = np.array([0] * self.num_constraints)
        ocp.constraints.uh = np.array([1e10] * self.num_constraints)

        # Slack variables for the constraints
        ocp.constraints.idxsh = np.arange(self.num_constraints)
        ocp.constraints.lsh = np.array([0] * self.num_constraints)
        ocp.constraints.ush = np.array([1e10] * self.num_constraints)

        # Set the cost for constraint violations
        ocp.cost.Zl = np.array([10000] * self.num_constraints)
        ocp.cost.Zu = np.array([0] * self.num_constraints)
        ocp.cost.zl = np.array([10000] * self.num_constraints)
        ocp.cost.zu = np.array([0] * self.num_constraints)

        # Set initial parameter values
        ocp.parameter_values = np.array([0.0] * ocp.model.p.rows())

    def update_parameters(
        self, ocp_solver: AcadosOcpSolver, N_horizon: int, obs: dict[str, NDArray[np.floating]]
    ):
        """Update the parameters of the OCP solver with the current obstacle and gate positions.

        Args:
            ocp_solver (AcadosOcpSolver): The acados OCP solver to update the parameters for.
            N_horizon (int): Number of time steps in the horizon.
            obs (dict[str, NDArray[np.floating]]): The current observation of the environment containing obstacle and gate positions.
        """
        self.obstacle_positions = obs["obstacles_pos"]

        # Filter out ignored obstacles
        active_obstacle_positions = np.array(
            [
                pos
                for i, pos in enumerate(self.obstacle_positions)
                if i not in self.ignored_obstacle_indices
            ]
        )

        obstacle_params = (
            active_obstacle_positions[:, :2].flatten()
            if len(active_obstacle_positions) > 0
            else np.array([])
        )

        self.gate_positions = obs["gates_pos"]
        self.gate_rotations = Rotation.from_quat(obs["gates_quat"]).as_euler("xyz", degrees=False)
        gate_params = np.hstack((self.gate_positions, self.gate_rotations[:, 2:])).flatten()
        params = np.concatenate((obstacle_params, gate_params))

        for i in range(N_horizon):
            gate_params = self.gate_positions[:, :4].flatten()
            ocp_solver.set(i, "p", params)

    def get_active_obstacle_indices(self) -> list[int]:
        """Get the indices of obstacles that are not ignored.

        Returns:
            list[int]: List of obstacle indices that are actively considered for collision avoidance.
        """
        return [i for i in range(self.num_obstacles) if i not in self.ignored_obstacle_indices]

    def get_obstacle_cylinders(self) -> dict[str, NDArray[np.floating]]:
        """Get the parameters for the infinite cylinders representing the obstacles.

        Returns:
            dict[str, NDArray[np.floating]]: A dictionary containing the positions and radii of the cylinders.
        """
        if self.obstacle_positions is None:
            return {"pos": np.array([]), "radius": np.array([])}

        # Filter out ignored obstacles
        active_obstacle_positions = np.array(
            [
                pos
                for i, pos in enumerate(self.obstacle_positions)
                if i not in self.ignored_obstacle_indices
            ]
        )

        if len(active_obstacle_positions) == 0:
            return {"pos": np.array([]), "radius": np.array([])}

        cylinder_positions = active_obstacle_positions.copy()
        cylinder_positions[:, 2] = 0  # Set z to 0 for infinite cylinders

        return {
            "pos": np.array(cylinder_positions),
            "radius": np.array([self.obstacle_radius] * len(cylinder_positions)),
        }

    def get_gate_ellipsoids(self) -> dict[str, NDArray[np.floating]]:
        """Get the parameters for the ellipsoids representing the gates.

        Returns:
            dict[str, NDArray[np.floating]]: A dictionary containing the positions, axes, and rotations of the ellipsoids.
        """
        ellipsoid_midpoints = []
        ellipsoid_axes = []
        ellipsoid_rotations = []

        gate_yaws = self.gate_rotations[:, 2]  # Extract yaw angles
        for pos, yaw in zip(self.gate_positions, gate_yaws):
            # Rotation matrix for yaw (around z)
            c = np.cos(yaw)
            s = np.sin(yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            for midpoint, axes in zip(self.ellipsoid_midpoints, self.ellipsoid_axes):
                ellipsoid_midpoints.append(pos + Rz @ midpoint)
                ellipsoid_axes.append(axes)
                ellipsoid_rotations.append(Rz)

        return {
            "pos": np.array(ellipsoid_midpoints),
            "axes": np.array(ellipsoid_axes),
            "rot": np.array(ellipsoid_rotations),
        }

    def _create_gate_expressions(self, drone_pos: MX) -> tuple[list[MX], list[MX]]:
        """Create parameters and expressions for ellipsoidal collision bodies for the gates.

        Args:
            drone_pos (MX): The position of the drone in the form of a CasADi MX variable.

        Returns:
            tuple[list[MX], list[MX]]: A tuple containing two lists:
                - params: List of MX variables representing the gate positions.
                - h_expr: List of MX expressions representing the ellipsoidal constraints for the gates.
        """
        params = []
        h_expr = []
        for i in range(self.num_gates):
            gate_pos = MX.sym(f"p_gate{i}", 4)  # Gate position (x, y, z, yaw)
            params.append(gate_pos)

            center = gate_pos[:3]  # Center of the gate
            yaw = gate_pos[3]  # Yaw angle of the gate

            # Rotation matrix for yaw (around z)
            Rz = vertcat(
                horzcat(cos(yaw), -sin(yaw), 0), horzcat(sin(yaw), cos(yaw), 0), horzcat(0, 0, 1)
            )

            for midpoint, axes in zip(self.ellipsoid_midpoints, self.ellipsoid_axes):
                ellipsoid_center = center + Rz @ MX(midpoint)
                a, b, c = axes  # Axes of the ellipsoid

                dpos = drone_pos - ellipsoid_center

                # Ellipsoid equation
                ellipsoid_exp = (
                    dpos.T @ Rz.T @ diag([1 / a**2, 1 / b**2, 1 / c**2]) @ Rz @ dpos
                ) - 1
                h_expr.append(ellipsoid_exp)

        return params, h_expr

    def _create_obstacle_expressions(self, drone_pos: MX) -> tuple[list[MX], list[MX]]:
        """Create parameters and expressions for infinite cylinder collision bodies for the obstacles.

        Args:
            drone_pos (MX): The position of the drone in the form of a CasADi MX variable.

        Returns:
            tuple[list[MX], list[MX]]: A tuple containing two lists:
                - params: List of MX variables representing the obstacle positions.
                - h_expr: List of MX expressions representing the cylindrical constraints for the obstacles.
        """
        params = []
        h_expr = []

        # Create expressions only for non-ignored obstacles
        active_obstacle_count = 0
        for i in range(self.num_obstacles):
            if i not in self.ignored_obstacle_indices:
                center_obs = MX.sym(f"p_obs{active_obstacle_count}", 2)
                params.append(center_obs)

                # Infinitly high cylinder around the obstacle with radius OBSTACLE_RADIUS
                pos_xy = drone_pos[:2]  # Get the x and y position of the drone
                con_h_expr = (pos_xy - center_obs).T @ (
                    pos_xy - center_obs
                ) - self.obstacle_radius**2
                h_expr.append(con_h_expr)

                active_obstacle_count += 1

        return params, h_expr
