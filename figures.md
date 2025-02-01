1. Start the server: `python server.py --blend_file transparent.blend --bpy_settings_file enable_gpu_bpy_setting_file.py`
2. Run the Drake sim: `python examples/ball_bin.py --no-server --scenario_file examples/ball_bin.yaml --stil`

# Recommended camera and lighting editing workflow

Start the blender with the export scene option during rendering:
```
python server.py --blend_file transparent.blend \
--bpy_settings_file enable_gpu_bpy_setting_file.py --export_scenes export_dir
```
This will save a new blend file in the export directory every time a new image is
rendered. This scene will contain all the Drake objects.
Open this scene in Blender and adjust the camera and lighting as needed. This can also
be used to directly render images from Blender. It is best to adjust the camera by
clicking on its viewport and using the lock mode to move the camera while looking through
it.
You can then delete all Drake objects and save the scene with the new lighting. Take note
of the camera pose, run it through the `blender_to_opencv_camera_euler.py` script and
use the resulting pose in the Blender camera scenario config.
