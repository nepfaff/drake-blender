# SPDX-License-Identifier: MIT-0

import argparse
import logging
import os
import tempfile
import time
from pydrake.common import configure_logging
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
)
from pydrake.systems.sensors import ImageWriter, PixelType
from pydrake.visualization import VideoWriter
from pydrake.all import (
    StartMeshcat,
    Simulator,
    RigidTransform,
    ConstantValueSource,
    RollPitchYaw,
    Parser,
    DiagramBuilder,
    ConstantVectorSource,
    TrajectorySource,
    DerivativeTrajectory,
    Trajectory,
    PiecewisePolynomial,
)
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# from python import runfiles
import tqdm
from manipulation.station import (
    MakeHardwareStation,
    LoadScenario,
    MultibodyPlant,
    RobotDiagram,
)
from functools import partial
import numpy as np

scenario_str = """
simulation_duration: 1.0

directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa14_polytope_collision.sdf
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
      translation: [0, 0, 0]
      rotation: !Rpy { deg: [0.0, 0.0, 0.0] }
- add_model:
    name: wsg
    file: package://blender_models/wsg_description/wsg50_110_finray_fingers_box_collision.sdf
    default_joint_positions:
        right_finger_sliding_joint: [0.05]
        left_finger_sliding_joint: [-0.05]
- add_frame:
    name: iiwa::wsg_attach
    X_PF:
        base_frame: iiwa::iiwa_link_7
        translation: [0.0, 0.0, 0.08]
        rotation: !Rpy { deg: [90.0, 0.0, 90.0]}
- add_weld:
    parent: iiwa::wsg_attach
    child: wsg::body
- add_model:
    name: toteA
    file: package://blender_models/tote/tote.sdf
- add_weld:
    parent: world
    child: toteA::tote_base
    X_PC:
      translation: [0, -0.45, 0]
      rotation: !Rpy { deg: [0.0, 0.0, -90.0] }
- add_model:
    name: toteB
    file: package://blender_models/tote/tote.sdf
- add_weld:
    parent: world
    child: toteB::tote_base
    X_PC:
      translation: [0, 0.45, 0]
      rotation: !Rpy { deg: [0.0, 0.0, 90.0] }
- add_model:
    name: mustard
    file: package://drake_models/ycb/006_mustard_bottle.sdf
    default_free_body_pose:
        base_link_mustard:
            # Middle of workspace
            translation: [0.5, 0, 0.09515]
            rotation: !Rpy { deg: [-90, 0, -70]}
- add_model:
    name: floor
    file: package://blender_models/floor.sdf
- add_weld:
    parent: world
    child: floor::floor_base

model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}

plant_config:
    time_step: 1.0e-3
    contact_model: "hydroelastic_with_fallback"
    discrete_contact_approximation: "lagged"

cameras:
    blender_camera:
        name: blender_camera
        renderer_name: blender
        renderer_class: !RenderEngineGltfClientParams
            base_url: http://127.0.0.1:8000
        width: 2
        height: 2
        # How many frames per second to record.
        fps: 24
"""

# The following is useful for playing with joint positions:
# python3 -m pydrake.visualization.model_visualizer package://drake_models/iiwa_description/sdf/iiwa14_polytope_collision.sdf
iiwa_positions = {
    "neutral": [0, -0.5, 0, -1.5, 0, 1.6, 0],
    "pick_bin_a": [1.65, 0.2, 0, -2, 0, 1, 0.9],
    "scanning": [1.5, 1.1, 1.8, 1.9, 0.5, -0.83, -1.8],
    "place_bin_b": [-1.55, 0.22, -0.1, -2, 0, 1, 0.9],
    "sys_id": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Not used
}
wsg_positions = {
    "neutral": [0.06],
    "pick_bin_a": [0.03],
    "scanning": [0.06],
    "place_bin_b": [0.05],
    "sys_id": [0.03],
}
system_id_traj_parameter_path = "system_id_traj2"
system_id_traj_time_horizon = 10.0


@dataclass
class FourierSeriesTrajectoryAttributes:
    """A data class to hold the attributes of a finite Fourier series trajectory."""

    a_values: int
    """The `a` parameters of shape (num_joints, num_fourier_terms)."""
    b_values: np.ndarray
    """The `b` parameters of shape (num_joints, num_fourier_terms)."""
    q0_values: np.ndarray
    """The `q0` parameters of shape (num_joints,)."""
    omega: float
    """The frequency of the trajectory in radians."""

    @classmethod
    def from_flattened_data(
        cls,
        a_values: np.ndarray,
        b_values: np.ndarray,
        q0_values: np.ndarray,
        omega: float,
        num_joints: int,
    ) -> "FourierSeriesTrajectoryAttributes":
        """Creates a FourierSeriesTrajectoryAttributes object from flattened data."""
        return cls(
            a_values=a_values.reshape((num_joints, -1), order="F"),
            b_values=b_values.reshape((num_joints, -1), order="F"),
            q0_values=q0_values,
            omega=omega,
        )

    def to_flattened_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Converts the attributes to flattened data.

        Returns: A tuple (a_values, b_values, q0_values, omega) of flattened arrays.
        """
        return (
            self.a_values.flatten(order="F"),
            self.b_values.flatten(order="F"),
            self.q0_values,
            self.omega,
        )

    @classmethod
    def load(
        cls, path: Path, num_joints: int | None = None
    ) -> "FourierSeriesTrajectoryAttributes":
        """Loads the trajectory attributes from disk."""
        a_values = np.load(path / "a_value.npy")
        b_values = np.load(path / "b_value.npy")
        q0_values = np.load(path / "q0_value.npy")
        omega = float(np.load(path / "omega.npy")[0])

        if len(a_values.shape) == 1:
            assert (
                num_joints is not None
            ), "num_joints must be provided when loading flattened data!"
            return cls.from_flattened_data(
                a_values=a_values,
                b_values=b_values,
                q0_values=q0_values,
                omega=omega,
                num_joints=num_joints,
            )

        return cls(
            a_values=a_values,
            b_values=b_values,
            q0_values=q0_values,
            omega=omega,
        )


class FourierSeriesTrajectory(Trajectory):
    """
    Represents the following Fourier series trajectory:
    qᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ sin(ωₙ lt) + bₗⁱ cos(ωₙ lt)) + qᵢ₀
    q̇ᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ ωₙ l cos(ωₙ lt) - bₗⁱ ωₙ l sin(ωₙ lt))
    q̈ᵢ(t) = ∑ₗ₌₁ᴺᵢ (-aₗⁱ ωₙ^2 l^2 sin(ωₙ lt) - bₗⁱ ωₙ^2 l^2 cos(ωₙ lt))
    """

    def __init__(
        self,
        traj_attrs: FourierSeriesTrajectoryAttributes,
        time_horizon: float,
        traj_start_time: float = 0.0,
    ):
        """
        Args:
            traj_attrs: The Fourier series trajectory attributes.
            time_horizon: The time horizon of the trajectory in seconds.
            traj_start_time: The start time of the trajectory in seconds.
        """
        super().__init__()

        self._traj_attrs = traj_attrs
        self._time_horizon = time_horizon
        self._traj_start_time = traj_start_time
        self._a = traj_attrs.a_values
        self._b = traj_attrs.b_values
        self._q0 = traj_attrs.q0_values
        self._omega = traj_attrs.omega
        self._num_positions, self._num_terms = self._a.shape

        # Used for computing the positions, velocities, and accelerations
        self._l_values = np.arange(1, self._num_terms + 1)
        self._omega_l = self._omega * self._l_values

    def _compute_positions(self, time: np.ndarray) -> np.ndarray:
        """qᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ sin(ωₙ lt) + bₗⁱ cos(ωₙ lt)) + qᵢ₀"""
        cos_part = np.cos(self._omega_l * time)
        sin_part = np.sin(self._omega_l * time)
        return (
            np.einsum("ij,j->i", self._a, sin_part)
            + np.einsum("ij,j->i", self._b, cos_part)
            + self._q0
        )

    def _compute_velocities(self, time: np.ndarray) -> np.ndarray:
        """q̇ᵢ(t) = ∑ₗ₌₁ᴺᵢ (aₗⁱ ωₙ l cos(ωₙ lt) - bₗⁱ ωₙ l sin(ωₙ lt))"""
        cos_part = self._omega_l * np.cos(self._omega_l * time)
        sin_part = self._omega_l * np.sin(self._omega_l * time)
        return np.einsum("il,l->i", self._a, cos_part) - np.einsum(
            "il,l->i", self._b, sin_part
        )

    def _compute_accelerations(self, time: np.ndarray) -> np.ndarray:
        """q̈ᵢ(t) = ∑ₗ₌₁ᴺᵢ (-aₗⁱ ωₙ^2 l^2 sin(ωₙ lt) - bₗⁱ ωₙ^2 l^2 cos(ωₙ lt))"""
        sin_part = ((self._omega_l) ** 2) * np.sin(self._omega_l * time)
        cos_part = ((self._omega_l) ** 2) * np.cos(self._omega_l * time)
        return np.einsum("il,l->i", -self._a, sin_part) + np.einsum(
            "il,l->i", -self._b, cos_part
        )

    def rows(self) -> int:
        return self._num_positions

    def cols(self) -> int:
        return 1

    def start_time(self) -> float:
        return self._traj_start_time

    def end_time(self) -> float:
        return self._traj_start_time + self._time_horizon

    def value(self, time: float) -> np.ndarray:
        return self.DoEvalDerivative(time, derivative_order=0)

    def do_has_derivative(self):
        return True

    def DoEvalDerivative(
        self, time: float, derivative_order: int
    ) -> np.ndarray:
        assert derivative_order < 3
        traj_time = time - self._traj_start_time
        if derivative_order == 0:
            return self._compute_positions(traj_time)
        if derivative_order == 1:
            return self._compute_velocities(traj_time)
        if derivative_order == 2:
            return self._compute_accelerations(traj_time)

    def DoMakeDerivative(
        self, derivative_order: int
    ) -> "FourierSeriesTrajectory":
        return DerivativeTrajectory(
            nominal=self, derivative_order=derivative_order
        )

    def Clone(self) -> "FourierSeriesTrajectory":
        return FourierSeriesTrajectory(
            traj_attrs=copy.deepcopy(self._traj_attrs),
            time_horizon=self._time_horizon,
            traj_start_time=self._traj_start_time,
        )


class _ProgressBar:
    def __init__(self, simulation_duration):
        self._tqdm = tqdm.tqdm(total=simulation_duration)
        self._current_time = 0.0

    def __call__(self, context):
        old_time = self._current_time
        self._current_time = context.get_time()
        self._tqdm.update(self._current_time - old_time)


def _run(args):
    """Runs the demo."""
    mode = args.mode
    if mode not in iiwa_positions:
        raise ValueError(f"Invalid mode: {mode}")

    mustard_grasped_mode = mode in ["pick_bin_a"]

    scenario = LoadScenario(data=scenario_str)

    if mode == "place_bin_b":
        # Need small sim duration for stable grasp.
        scenario.simulation_duration = 0.04
    elif mode == "sys_id":
        scenario.simulation_duration = system_id_traj_time_horizon

    video_writers = []
    for _, camera in scenario.cameras.items():
        writer = VideoWriter(
            filename=f"{camera.name}.mp4",
            fps=camera.fps,
            backend="cv2",
        )
        video_writers.append(writer)

    def prefinalize_callback(parser: Parser):
        plant: MultibodyPlant = parser.plant()

        if mustard_grasped_mode:
            # Disable mustard gravity.
            mustard_instance = plant.GetModelInstanceByName("mustard")
            plant.set_gravity_enabled(
                model_instance=mustard_instance, is_enabled=False
            )

        if mode in ["sys_id"]:
            # Weld mustard to avoid it from slipping out of the gripper.
            X_GM = RigidTransform(
                p=[0.0, 0.12, 0.01],
                rpy=RollPitchYaw(-np.pi / 2, -0.3, np.pi / 2),
            )
            plant.WeldFrames(
                frame_on_parent_F=plant.GetFrameByName("body"),
                frame_on_child_M=plant.GetFrameByName("base_link_mustard"),
                X_FM=X_GM,
            )

    def prebuild_callback(builder, video_writers):
        # Add the camera(s).
        for camera, writer in zip(scenario.cameras.values(), video_writers):
            if args.still:
                camera.show_rgb = False
            name = camera.name
            sensor = builder.GetSubsystemByName(f"rgbd_sensor_{name}")
            if args.still:
                writer = builder.AddSystem(ImageWriter())
                writer.DeclareImageInputPort(
                    pixel_type=PixelType.kRgba8U,
                    port_name="color_image",
                    file_name_format=f"./{name}",
                    publish_period=1.0,
                    start_time=scenario.simulation_duration,
                )
                builder.Connect(
                    sensor.GetOutputPort("color_image"),
                    writer.GetInputPort("color_image"),
                )
            else:
                builder.AddSystem(writer)
                writer.ConnectRgbdSensor(builder=builder, sensor=sensor)

    # Create the scene.
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    station: RobotDiagram = builder.AddSystem(
        MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            package_xmls=[
                os.path.join(
                    os.path.dirname(__file__), "../blender_models/package.xml"
                )
            ],
            parser_prefinalize_callback=prefinalize_callback,
            prebuild_callback=partial(
                prebuild_callback, video_writers=video_writers
            ),
        )
    )

    # Connect iiwa and wsg position sources.
    if mode == "sys_id":
        # Add a trajectory source for the system ID trajectory.
        traj_attrs = FourierSeriesTrajectoryAttributes.load(
            Path(system_id_traj_parameter_path)
        )
        excitation_traj = FourierSeriesTrajectory(
            traj_attrs=traj_attrs,
            time_horizon=system_id_traj_time_horizon,
        )
        traj_source = builder.AddSystem(
            TrajectorySource(trajectory=excitation_traj)
        )
        builder.Connect(
            traj_source.get_output_port(),
            station.GetInputPort("iiwa.position"),
        )
        iiwa_positions[mode] = excitation_traj.value(0.0)
    else:
        iiwa_position_source = builder.AddSystem(
            ConstantVectorSource(iiwa_positions[mode])
        )
        builder.Connect(
            iiwa_position_source.get_output_port(),
            station.GetInputPort("iiwa.position"),
        )
    wsg_position_source = builder.AddSystem(
        ConstantVectorSource(wsg_positions[mode])
    )
    builder.Connect(
        wsg_position_source.get_output_port(),
        station.GetInputPort("wsg.position"),
    )

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant = station.plant()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    # Set the initial iiwa position.
    plant.SetPositions(
        context=plant_context,
        model_instance=plant.GetModelInstanceByName("iiwa"),
        q=iiwa_positions[mode],
    )

    # Set the initial mustard position.
    if mode in ["pick_bin_a"]:
        X_WG = plant.EvalBodyPoseInWorld(
            context=plant_context,
            body=plant.GetBodyByName("body"),
        )
        X_GM = RigidTransform(
            p=[0.0, 0.12, 0.0],
            rpy=RollPitchYaw(-np.pi / 2, 0.0, np.pi / 2),
        )
        X_WM = X_WG @ X_GM
        plant.SetFreeBodyPose(
            context=plant_context,
            body=plant.GetBodyByName("base_link_mustard"),
            X_PB=X_WM,
        )
    elif mode in ["place_bin_b"]:
        X_WG = plant.EvalBodyPoseInWorld(
            context=plant_context,
            body=plant.GetBodyByName("body"),
        )
        X_GM = RigidTransform(
            p=[0.0, 0.255, 0.0],
            rpy=RollPitchYaw(0.0, np.pi / 2, 0.0),
        )
        X_WM = X_WG @ X_GM
        plant.SetFreeBodyPose(
            context=plant_context,
            body=plant.GetBodyByName("base_link_mustard"),
            X_PB=X_WM,
        )

    # Create the simulator.
    simulator = Simulator(diagram, context=diagram_context)
    ApplySimulatorConfig(scenario.simulator_config, simulator)

    # Simulate.
    meshcat.StartRecording()
    if args.still:
        logging.info("Creating still image(s)")
        simulator.AdvanceTo(scenario.simulation_duration)
    else:
        logging.info("Creating video(s)")
        simulator.set_monitor(_ProgressBar(scenario.simulation_duration))
        simulator.AdvanceTo(scenario.simulation_duration)
        # for writer in video_writers:
        #     writer.Save()
    meshcat.PublishRecording()

    time.sleep(5.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--still",
        action="store_true",
        help="Don't create a video; instead, capture a single photograph of "
        "the initial conditions.",
    )
    parser.add_argument("--mode", type=str, default="neutral")
    args = parser.parse_args()

    # Run the demo.
    _run(args)


def _wrapped_main():
    # Do our best to clean up after ourselves, by advising Drake code to use
    # a directory other than /tmp.
    with tempfile.TemporaryDirectory(prefix="ball_bin_") as temp_dir:
        os.environ["TMPDIR"] = temp_dir
        main()


if __name__ == "__main__":
    # Create output files in $PWD, not runfiles.
    if "BUILD_WORKING_DIRECTORY" in os.environ:
        os.chdir(os.environ["BUILD_WORKING_DIRECTORY"])

    configure_logging()
    _wrapped_main()
