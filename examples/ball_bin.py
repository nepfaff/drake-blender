# SPDX-License-Identifier: MIT-0

import argparse
import logging
import os
from pathlib import Path
import tempfile
import time
from pydrake.common import configure_logging
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
)
from pydrake.systems.sensors import ImageWriter, PixelType
from pydrake.visualization import VideoWriter
from pydrake.all import StartMeshcat, Simulator

# from python import runfiles
import tqdm
from manipulation.station import MakeHardwareStation, LoadScenario
from functools import partial


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
    scenario = LoadScenario(filename=args.scenario_file)

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
        scenario=scenario,
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
    parser = argparse.ArgumentParser()
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
