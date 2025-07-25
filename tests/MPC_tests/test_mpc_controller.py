"""Direct tests for MPController class with mocked dependencies."""

from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest

from lsy_drone_racing.control.attitude_mpc_combined import MPController


class TestMPController:
    """Tests for MPController with mocked heavy dependencies."""

    @pytest.fixture
    def config(self) -> Mock:
        """Create test configuration.

        Returns:
            Mock: Mock configuration object with environment settings.
        """
        config = Mock()
        config.env.freq = 50
        config.env.track = {
            "gates": [
                {"pos": [1.0, 0.0, 1.0], "rpy": [0, 0, 0]},
                {"pos": [2.0, 1.0, 1.0], "rpy": [0, 0, 1.57]},
                {"pos": [1.0, 2.0, 1.2], "rpy": [0, 0, 3.14]},
                {"pos": [0.0, 1.0, 1.0], "rpy": [0, 0, -1.57]},
            ]
        }
        return config

    @pytest.fixture
    def obs(self) -> Dict[str, Any]:
        """Create test observation.

        Returns:
            Dict[str, Any]: Dictionary containing observation data with positions, velocities, and orientations.
        """
        return {
            "pos": np.array([0.0, 0.0, 1.0]),
            "vel": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "gates_pos": np.array(
                [[1.0, 0.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.2], [0.0, 1.0, 1.0]]
            ),
            "obstacles_pos": np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 0.8]]),
            "target_gate": 0,
        }

    @pytest.fixture
    def mock_solver(self) -> Mock:
        """Create a mock Acados solver.

        Returns:
            Mock: Mock solver object with predefined state and method responses.
        """
        solver = Mock()
        solver.solve.return_value = 0  # Success

        # Mock state vector (14 elements for your controller)
        state = np.array(
            [
                0.0,
                0.0,
                1.0,  # position
                0.0,
                0.0,
                0.0,  # velocity
                0.0,
                0.0,
                0.0,  # roll, pitch, yaw
                0.3,
                0.3,  # f_collective, f_cmd
                0.0,
                0.0,
                0.0,  # rpy_cmd
            ]
        )
        solver.get.return_value = state
        solver.set = Mock()
        solver.cost_set = Mock()
        solver.cost_get = Mock(return_value=np.eye(18))

        return solver

    @pytest.fixture
    def controller(self, config: Mock, obs: Dict[str, Any], mock_solver: Mock) -> MPController:
        """Create MPController with mocked dependencies.

        Args:
            config: Mock configuration object.
            obs: Observation dictionary.
            mock_solver: Mock Acados solver.

        Returns:
            MPController: Initialized controller instance with mocked dependencies.
        """
        with patch(
            "lsy_drone_racing.control.attitude_mpc_combined.AcadosOcpSolver",
            return_value=mock_solver,
        ):
            with patch("lsy_drone_racing.control.attitude_mpc_combined.export_quadrotor_ode_model"):
                with patch("lsy_drone_racing.control.attitude_mpc_combined.setup_ocp"):
                    with patch("lsy_drone_racing.control.attitude_mpc_combined.FlightLogger"):
                        with patch(
                            "lsy_drone_racing.control.attitude_mpc_combined.CollisionAvoidanceHandler"
                        ):
                            controller = MPController(obs, {}, config)
                            controller.acados_ocp_solver = mock_solver
                            return controller

    def test_controller_initialization(self, controller: MPController) -> None:
        """Test controller initializes with correct parameters.

        Args:
            controller: MPController instance to test.
        """
        assert controller.freq == 50
        assert controller.N == 60
        assert controller.T_HORIZON == 2.0
        assert controller.gates_passed == 0
        assert not controller.finished
        assert controller.total_gates == 4

    def test_weight_adjustment_activation(self, controller: MPController) -> None:
        """Test weight adjustment activation.

        Args:
            controller: MPController instance to test.
        """
        original_q_pos = controller.mpc_weights["Q_pos"]

        controller._activate_replanning_weights_gradual()
        assert controller.weights_adjusted

        controller._update_weights()
        assert controller.mpc_weights["Q_pos"] >= original_q_pos

    def test_weight_adjustment_restoration(self, controller: MPController) -> None:
        """Test weight adjustment restoration after duration.

        Args:
            controller: MPController instance to test.
        """
        controller._activate_replanning_weights_gradual()
        controller._tick = (
            controller.weight_adjustment_start_tick + controller.weight_adjustment_duration + 10
        )

        controller._update_weights()
        assert not controller.weights_adjusted
        assert controller.mpc_weights == controller.original_mpc_weights

    def test_gate_progress_tracking_normal(
        self, controller: MPController, obs: Dict[str, Any]
    ) -> None:
        """Test normal gate progression tracking.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
        """
        # Mock the heavy computation methods
        with patch.object(controller, "_execute_mpc_control", return_value=np.zeros(4)):
            with patch.object(controller, "_check_and_execute_replanning", return_value=False):
                with patch.object(controller.collision_avoidance_handler, "update_parameters"):
                    # Test gate 1
                    obs_gate1 = obs.copy()
                    obs_gate1["target_gate"] = 1
                    controller.compute_control(obs_gate1)
                    assert controller.gates_passed == 1

    def test_gate_progress_tracking_completion(
        self, controller: MPController, obs: Dict[str, Any]
    ) -> None:
        """Test completion tracking.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
        """
        with patch.object(controller, "_execute_mpc_control", return_value=np.zeros(4)):
            with patch.object(controller, "_check_and_execute_replanning", return_value=False):
                with patch.object(controller.collision_avoidance_handler, "update_parameters"):
                    # Test completion
                    obs_complete = obs.copy()
                    obs_complete["target_gate"] = -1
                    controller.compute_control(obs_complete)
                    assert controller.gates_passed == 4
                    assert controller.flight_successful

    def test_replanning_trigger_gate_moved(
        self, controller: MPController, obs: Dict[str, Any]
    ) -> None:
        """Test replanning triggers when gate moves.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
        """
        # Set up controller state for replanning
        controller._tick = 1  # Set tick > 0
        controller.last_replanning_tick = 0  # Allow replanning

        # Create observation with moved gate
        obs_moved = obs.copy()
        obs_moved["gates_pos"] = obs["gates_pos"].copy()
        obs_moved["gates_pos"][0] = [1.1, 0.0, 1.0]  # Move gate 0 by 10cm
        obs_moved["vel"] = np.array([0.5, 0.0, 0.0])  # Moving toward gate

        # Mock the trajectory planner methods
        with patch.object(
            controller.trajectory_planner, "generate_smooth_replanning_waypoints"
        ) as mock_waypoints:
            with patch.object(
                controller.trajectory_planner, "generate_trajectory_from_waypoints"
            ) as mock_traj:
                mock_waypoints.return_value = np.array([[0, 0, 1], [1, 0, 1]])
                mock_traj.return_value = (np.array([0, 1]), np.array([0, 0]), np.array([1, 1]))

                result = controller._check_and_execute_replanning(obs_moved, 0)

        assert result
        assert 0 in controller.updated_gates

    def test_replanning_no_trigger_small_movement(
        self, controller: MPController, obs: Dict[str, Any]
    ) -> None:
        """Test replanning doesn't trigger for small gate movements.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
        """
        obs_small_move = obs.copy()
        obs_small_move["gates_pos"] = obs["gates_pos"].copy()
        obs_small_move["gates_pos"][0] = [1.01, 0.0, 1.0]  # Move gate by 1cm (below threshold)

        result = controller._check_and_execute_replanning(obs_small_move, 0)
        assert not result

    def test_mpc_control_execution_success(
        self, controller: MPController, obs: Dict[str, Any], mock_solver: Mock
    ) -> None:
        """Test MPC control execution with successful solve.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
            mock_solver: Mock solver instance.
        """
        # Set up trajectory data
        controller.x_des = np.array([1.0, 1.1, 1.2])
        controller.y_des = np.array([0.0, 0.0, 0.0])
        controller.z_des = np.array([1.0, 1.0, 1.0])
        controller._trajectory_start_tick = 0

        action = controller._execute_mpc_control(obs, 0, False)

        assert action.shape == (4,)  # [f_collective, roll, pitch, yaw]
        assert mock_solver.solve.called
        assert mock_solver.set.called

    def test_mpc_control_execution_solver_failure(
        self, controller: MPController, obs: Dict[str, Any], mock_solver: Mock
    ) -> None:
        """Test MPC control execution with solver failure.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
            mock_solver: Mock solver instance.
        """
        mock_solver.solve.return_value = 1  # Failure status

        controller.x_des = np.array([1.0])
        controller.y_des = np.array([0.0])
        controller.z_des = np.array([1.0])
        controller._trajectory_start_tick = 0

        # Should not crash despite solver failure
        action = controller._execute_mpc_control(obs, 0, False)
        assert action.shape == (4,)

    def test_step_callback_basic(self, controller: MPController, obs: Dict[str, Any]) -> None:
        """Test step callback increments tick and logs.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
        """
        initial_tick = controller._tick

        controller.step_callback(
            action=np.zeros(4), obs=obs, reward=1.0, terminated=False, truncated=False, info={}
        )

        assert controller._tick == initial_tick + 1

    def test_episode_reset_clears_state(self, controller: MPController) -> None:
        """Test episode reset clears all controller state.

        Args:
            controller: MPController instance to test.
        """
        # Modify controller state
        controller._tick = 100
        controller.gates_passed = 2
        controller.finished = True
        controller.updated_gates.add(1)
        controller.weights_adjusted = True

        # Mock the file saving
        with patch.object(controller, "x_traj", []):
            with patch("numpy.savez"):
                controller.episode_reset()

        # Verify reset
        assert controller._tick == 0
        assert controller.gates_passed == 0
        assert not controller.finished
        assert len(controller.updated_gates) == 0
        assert not controller.weights_adjusted

    def test_is_drone_approaching_gate_true(self, controller: MPController) -> None:
        """Test drone approaching gate detection - true case.

        Args:
            controller: MPController instance to test.
        """
        controller.config.env.track["gates"][0]["pos"] = [1.0, 0.0, 1.0]

        obs_approaching = {
            "pos": np.array([0.0, 0.0, 1.0]),
            "vel": np.array([1.0, 0.0, 0.0]),  # Moving toward gate
            "gates_pos": np.array([[1.0, 0.0, 1.0]]),
        }

        result = controller._is_drone_approaching_gate(obs_approaching, 0)
        assert result

    def test_is_drone_approaching_gate_false(self, controller: MPController) -> None:
        """Test drone approaching gate detection - false case.

        Args:
            controller: MPController instance to test.
        """
        controller.config.env.track["gates"][0]["pos"] = [1.0, 0.0, 1.0]

        obs_moving_away = {
            "pos": np.array([0.5, 0.0, 1.0]),
            "vel": np.array([-1.0, 0.0, 0.0]),  # Moving away
            "gates_pos": np.array([[1.0, 0.0, 1.0]]),
        }

        result = controller._is_drone_approaching_gate(obs_moving_away, 0)
        assert not result

    def test_mpc_reference_setting(
        self, controller: MPController, obs: Dict[str, Any], mock_solver: Mock
    ) -> None:
        """Test MPC reference trajectory setting.

        Args:
            controller: MPController instance to test.
            obs: Observation dictionary.
            mock_solver: Mock solver instance.
        """
        controller.x_des = np.array([1.0, 1.1, 1.2, 1.3])
        controller.y_des = np.array([0.0, 0.0, 0.0, 0.0])
        controller.z_des = np.array([1.0, 1.0, 1.0, 1.0])

        # Should not crash
        controller._set_mpc_references(i=0, traj_len=4, just_replanned=False, tracking_error=0.1)

        # Verify solver.set was called for references
        assert mock_solver.set.call_count > 0

    def test_predicted_trajectory_retrieval(
        self, controller: MPController, mock_solver: Mock
    ) -> None:
        """Test getting predicted trajectory from MPC.

        Args:
            controller: MPController instance to test.
            mock_solver: Mock solver instance.
        """
        # Set up current trajectory
        controller.current_trajectory = {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.0, 0.0, 0.0]),
            "z": np.array([1.0, 1.0, 1.0]),
        }

        pred_traj, full_traj = controller.get_predicted_trajectory()

        assert pred_traj.shape[0] == controller.N  # N prediction points
        assert pred_traj.shape[1] == 3  # 3D coordinates
        assert full_traj.shape[1] == 3  # 3D coordinates

    def test_get_path_method(self, controller: MPController) -> None:
        """Test getting current path.

        Args:
            controller: MPController instance to test.
        """
        controller.x_des = np.array([0.0, 1.0, 2.0] + [2.0] * 100)  # Add extra points
        controller.y_des = np.array([0.0, 0.0, 0.0] + [0.0] * 100)
        controller.z_des = np.array([1.0, 1.0, 1.0] + [1.0] * 100)

        path = controller.get_path()

        assert path.shape[1] == 3  # 3D coordinates
        assert len(path) > 0

    def test_get_predicted_path_method(self, controller: MPController, mock_solver: Mock) -> None:
        """Test getting predicted path from solver.

        Args:
            controller: MPController instance to test.
            mock_solver: Mock solver instance.
        """
        path = controller.get_predicted_path()

        assert path.shape[0] == controller.N + 1  # N+1 prediction points
        assert path.shape[1] == 3  # 3D coordinates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
