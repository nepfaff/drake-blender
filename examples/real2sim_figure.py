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
)

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
simulation_duration: 2.0
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
            rotation: !Rpy { deg: [-90, 0, 0]}

model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}

cameras:
    blender_camera:
        name: blender_camera
        renderer_name: blender
        renderer_class: !RenderEngineGltfClientParams
            base_url: http://127.0.0.1:8000
        width: 1024
        height: 1024
        focal: !FovDegrees { x: 40 }
        fps: 8.0
        X_PB:
            translation: [2.0, 0.0, 0.5]
            # rotation: !Rpy { deg: [90, 0, 90] } # Blender
            rotation: !Rpy { deg: [-90.0, 0.0, 90.0] } # OpenCV
    vtk_camera:
        name: vtk_camera
        renderer_name: vtk
        renderer_class: !RenderEngineVtkParams
            backend: EGL
        # For `show_rgb: True` you must also set the `backend: GLX` on prior line
        # and be running locally with an Xorg display server available.
        show_rgb: False
        width: 1024
        height: 1024
        focal: !FovDegrees { x: 40 }
        fps: 8.0
        # background: [0, 0, 0, 1]
        X_PB:
            translation: [2.0, 0.0, 0.5]
            rotation: !Rpy { deg: [-90.0, 0.0, 90.0] }
"""

iiwa_positions = {
    "neutral": [0, -0.5, 0, -1.5, 0, 1.6, 0],
    "pick_bin_a": [-1.57, 0.2, 0, -2, 0, 1, 0.9],
}


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

    video_writers = []
    for _, camera in scenario.cameras.items():
        writer = VideoWriter(
            filename=f"{camera.name}.mp4",
            fps=16,
            backend="cv2",
        )
        video_writers.append(writer)

    def prefinalize_callback(parser: Parser):
        if mustard_grasped_mode:
            # Disable mustard gravity.
            plant: MultibodyPlant = parser.plant()
            mustard_instance = plant.GetModelInstanceByName("mustard")
            plant.set_gravity_enabled(
                model_instance=mustard_instance, is_enabled=False
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
    iiwa_position_source = builder.AddSystem(ConstantVectorSource(iiwa_positions[mode]))
    builder.Connect(
        iiwa_position_source.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    wsg_position_source = builder.AddSystem(ConstantVectorSource([0.03]))
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

    # Create the simulator.
    simulator = Simulator(diagram, context=diagram_context)
    simulator.set_target_realtime_rate(1.0)
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
        for writer in video_writers:
            writer.Save()
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
