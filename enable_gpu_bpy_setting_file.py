import bpy

# Need to set cycles for GPU acceleration but this might lead to worse performance in
# some instances.
bpy.data.scenes[0].render.engine = "CYCLES"

print("Render engine: ", bpy.data.scenes[0].render.engine)

if "cycles" in bpy.context.preferences.addons:
    print("Enabling GPU for Cycles")
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    cycles_prefs.compute_device_type = "OPTIX"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in cycles_prefs.devices:
        # d["use"] = 1 # bpy 3.6
        d.use = True # bpy 4.0

# elif bpy.data.scenes[0].render.engine == "BLENDER_EEVEE":
#     # Adjust Eevee settings
#     bpy.context.scene.eevee.use_gtao = True  # Enable ambient occlusion
#     bpy.context.scene.eevee.use_bloom = True  # Enable bloom effect
#     bpy.context.scene.eevee.use_ssr = True  # Enable screen space reflections
#     bpy.context.scene.eevee.use_ssr_refraction = (
#         True  # Enable refraction in SSR
#     )
#     bpy.context.scene.eevee.use_motion_blur = True  # Enable motion blur
#     bpy.context.scene.eevee.shadow_cube_size = (
#         "1024"  # Adjust shadow resolution
#     )
#     bpy.context.scene.eevee.use_soft_shadows = True  # Enable soft shadows
