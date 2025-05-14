"""
Model Predictive Controller for autonomous drone racing through dynamic gates and obstacles.
Implements reference tracking, obstacle avoidance, and motor-level control using quadrotor dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import casadi as ca
import numpy as np
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

def rotation_matrix(roll, pitch, yaw):
    """Returns the rotation matrix from body to world frame (Z-Y-X convention)."""
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw), ca.sin(yaw)
    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp, cp*sr, cp*cr)
    )
    return R

class MPCController(Controller):
    """Nonlinear MPC for drone racing with dynamic obstacle avoidance and full quadrotor dynamics."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self.gates = np.array([gate['pos'] for gate in config['env']['track']['gates']])
        self.obstacles = np.array([obs['pos'] for obs in config['env']['track']['obstacles']])
        self._safety_radius = 0.25  # [m] safety margin for obstacles
        self._setup_parameters()
        self._build_mpc()
        self._current_gate_idx = 0

    def _setup_parameters(self):
        self.mass = 0.27  # kg
        self.g = 9.81
        self.arm_length = 0.042  # m
        self.J = np.diag([1.395e-5, 1.436e-5, 2.173e-5])  # kg m^2
        self.Ct = 3.1582e-10  # N/(RPM^2)
        self.Cd = 7.9379e-12  # Nm/(RPM^2)
        self.dt = 0.1  # [s] discretization
        self.N = 10    # MPC horizon
        self.Q = np.diag([20, 20, 20, 2, 2, 2])  # State cost
        self.R = np.diag([0.05, 0.05, 0.05])     # Input cost
        self.z_min = 0.1
        self.v_max = 4.0
        self.a_max = 2.5 * self.g

    def _quadrotor_dynamics(self):
        # State: [x, y, z, vx, vy, vz]
        x = ca.MX.sym('x', 6)
        u = ca.MX.sym('u', 3)  # [ax, ay, az] in world frame

        dxdt = ca.vertcat(
            x[3],
            x[4],
            x[5],
            u[0],
            u[1],
            u[2] - self.g
        )
        return ca.Function('dynamics', [x, u], [dxdt])

    def _build_mpc(self):
        opti = ca.Opti()
        X = opti.variable(6, self.N+1)  # [x, y, z, vx, vy, vz]
        U = opti.variable(3, self.N)    # [ax, ay, az]
        x0 = opti.parameter(6)
        ref = opti.parameter(3)
        obstacles = opti.parameter(self.obstacles.shape[0], 3)

        f = self._quadrotor_dynamics()
        for k in range(self.N):
            x_next = X[:, k] + f(X[:, k], U[:, k]) * self.dt
            opti.subject_to(X[:, k+1] == x_next)

        # Cost: position error, velocity, input effort, terminal cost
        cost = 0
        for k in range(self.N):
            pos_err = X[:3, k] - ref
            vel = X[3:6, k]
            cost += ca.mtimes([pos_err.T, self.Q[:3, :3], pos_err])
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])
        # Terminal cost
        terminal_err = X[:3, self.N] - ref
        cost += ca.mtimes([terminal_err.T, 10*self.Q[:3, :3], terminal_err])
        opti.minimize(cost)

        # Constraints
        opti.subject_to(X[2, :] >= self.z_min)
        opti.subject_to(opti.bounded(-self.v_max, X[3:6, :], self.v_max))
        opti.subject_to(opti.bounded(-self.a_max, U, self.a_max))

        # Obstacle avoidance
        num_obstacles = self.obstacles.shape[0]
        for k in range(self.N+1):
            for j in range(num_obstacles):
                xk = X[:3, k]
                obsj = obstacles[j, :].T  # (3,1)
                dist_sq = ca.sumsqr(xk - obsj)
                opti.subject_to(dist_sq >= self._safety_radius**2)

        # Solver settings
        opts = {
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.linear_solver': 'mumps',
            'ipopt.print_level': 0,
            'print_time': 0
        }
        opti.solver('ipopt', opts)

        # Store for later use
        self.opti = opti
        self.X_var = X
        self.U_var = U
        self.x0_param = x0
        self.ref_param = ref
        self.obstacles_param = obstacles

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: Optional[dict] = None
    ) -> NDArray[np.floating]:
        # Extract current state
        current_pos = obs['pos']
        current_vel = obs['vel']
        x0 = np.concatenate([current_pos, current_vel])

        # Gate progression logic (advance if close to gate)
        target_gate = self.gates[self._current_gate_idx]
        dist_to_gate = np.linalg.norm(current_pos - target_gate)
        direction = (target_gate - current_pos) / (dist_to_gate + 1e-6)
        vel_proj = np.dot(current_vel, direction)
        if dist_to_gate < 0.7 or (dist_to_gate < 1.5 and vel_proj > 0.5):
            if self._current_gate_idx < len(self.gates) - 1:
                self._current_gate_idx += 1
            target_gate = self.gates[self._current_gate_idx]

        # Update obstacles if info is available
        if info and 'obstacles' in info:
            self.obstacles = np.array([obs['pos'] for obs in info['obstacles']])

        # Set MPC parameters
        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.ref_param, target_gate)
        self.opti.set_value(self.obstacles_param, self.obstacles)

        # Solve MPC
        try:
            sol = self.opti.solve()
            a_opt = sol.value(self.U_var)[:, 0]
        except Exception as e:
            print("MPC solve failed:", e)
            a_opt = np.zeros(3)

        # Convert acceleration to quadrotor motor commands
        return self._acceleration_to_motor_commands(a_opt, obs)

    def _acceleration_to_motor_commands(self, a: NDArray, obs: dict) -> NDArray:
        # Compute total force in world frame
        f_des = a + np.array([0, 0, self.g])
        # Desired roll/pitch/yaw from thrust direction (A.19–A.21)
        z_body = f_des / (np.linalg.norm(f_des) + 1e-6)
        yaw_des = 0.0  # Keep yaw fixed for simplicity
        x_world = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        y_body = np.cross(z_body, x_world)
        y_body /= np.linalg.norm(y_body) + 1e-6
        x_body = np.cross(y_body, z_body)
        R_des = np.stack([x_body, y_body, z_body], axis=1)
        # Euler angles from rotation matrix
        pitch_des = np.arcsin(-R_des[2, 0])
        roll_des = np.arctan2(R_des[2, 1], R_des[2, 2])
        # Thrust (projected)
        thrust = self.mass * np.dot(f_des, R_des[:, 2])
        thrust = np.clip(thrust, 0, 4 * self.mass * self.g)
        pwm = np.array([thrust / 4] * 4)
        return np.concatenate([pwm, np.zeros(9)], dtype=np.float32)

    def step_callback(self, *args) -> bool:
        return False