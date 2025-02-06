# SPDX-License-Identifier: MIT-0

"""
Demonstrates combining Drake with the Blender render server to create a
simulation video (or still image).

In this demo, moving objects (some balls) and a fixed object (a bin) are
simulated by Drake and the static visual background (a room with custom
lighting) is provided by a Blender file.
"""

import argparse
import dataclasses as dc
import logging
import os
from pathlib import Path
import tempfile
import typing
import time
from pydrake.common import configure_logging
from pydrake.common.yaml import yaml_load_typed
from pydrake.multibody.parsing import (
    ModelDirective,
)
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.sensors import (
    CameraConfig,
    ImageWriter,
    PixelType,
)
from pydrake.visualization import VideoWriter
from pydrake.all import StartMeshcat

# from python import runfiles
import tqdm
from manipulation.station import MakeHardwareStation, LoadScenario
from functools import partial


@dc.dataclass
class Scenario:
    """Defines the YAML format for a scenario to be simulated."""

    # The maximum simulation time (in seconds).
    simulation_duration: float = 1.0

    # Simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = SimulatorConfig()

    # All of the fully deterministic elements of the simulation.
    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # Cameras to add to the scene.
    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)


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
    scenario = yaml_load_typed(
        schema=Scenario, filename=args.scenario_file, defaults=Scenario()
    )

    video_writers = []
    for _, camera in scenario.cameras.items():
        writer = VideoWriter(
            filename=f"{camera.name}.mp4",
            fps=16,
            backend="cv2",
        )
        video_writers.append(writer)

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
                    publish_period=10.0,
                    start_time=0.0,
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
    station = MakeHardwareStation(
        scenario=LoadScenario(filename=args.scenario_file),
        meshcat=meshcat,
        package_xmls=[
            os.path.join(
                os.path.dirname(__file__), "../blender_models/package.xml"
            )
        ],
        prebuild_callback=partial(
            prebuild_callback, video_writers=video_writers
        ),
    )

    # Create the simulator.
    simulator = Simulator(station)
    simulator.set_target_realtime_rate(1.0)
    ApplySimulatorConfig(scenario.simulator_config, simulator)

    # Simulate.
    if args.still:
        logging.info("Creating still image(s)")
        simulator.AdvanceTo(1e-3)
    else:
        logging.info("Creating video(s)")
        simulator.set_monitor(_ProgressBar(scenario.simulation_duration))
        simulator.AdvanceTo(scenario.simulation_duration)
        for writer in video_writers:
            writer.Save()

    time.sleep(5.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--still",
        action="store_true",
        help="Don't create a video; instead, capture a single photograph of "
        "the initial conditions.",
    )
    parser.add_argument(
        "--scenario_file",
        type=Path,
        default="examples/ball_bin.yaml",
        help="The absolute path to a scenario file to construct a simulation. "
        "If not provided, `drake_blender/examples/ball_bin.yaml` will be used "
        "by default.",
    )
    parser.add_argument(
        "--no-server",
        dest="server",
        action="store_false",
        help="Don't automatically launch the blender server.",
    )
    parser.add_argument(
        "--bpy_settings_file",
        metavar="FILE",
        help="This flag is forward along to the server, unchanged. "
        "Refer to its documentation for details.",
    )
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
