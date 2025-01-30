import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R


def blender_to_opencv_euler(euler_blender):
    """
    Converts Euler angles from Blender (XYZ intrinsic) to OpenCV (XYZ intrinsic).

    Args:
        euler_blender (list): [roll, pitch, yaw] in degrees from Blender.

    Returns:
        list: Converted [roll, pitch, yaw] in degrees for OpenCV.
    """
    # Convert Blender Euler angles to rotation matrix
    R_blender = R.from_euler("xyz", euler_blender, degrees=True).as_matrix()

    # Blender-to-OpenCV coordinate transformation matrix
    # This represents the transformation from Blender to OpenCV coordinate system
    R_transform = np.array(
        [
            [1, 0, 0],  # X stays the same
            [0, -1, 0],  # Y is inverted
            [0, 0, -1],  # Z is inverted
        ]
    )

    # The correct order is: R_opencv = R_blender @ R_transform
    # This applies the coordinate system transformation after the rotation
    R_opencv = R_blender @ R_transform

    # Convert back to Euler angles (XYZ intrinsic for OpenCV)
    euler_opencv = R.from_matrix(R_opencv).as_euler("xyz", degrees=True)

    # Fix small numerical errors
    euler_opencv = np.round(euler_opencv, decimals=6).tolist()

    return euler_opencv


def main():
    parser = argparse.ArgumentParser(
        description="Convert Blender Euler angles to OpenCV."
    )
    parser.add_argument(
        "roll", type=float, help="Roll angle (Blender) in degrees"
    )
    parser.add_argument(
        "pitch", type=float, help="Pitch angle (Blender) in degrees"
    )
    parser.add_argument(
        "yaw", type=float, help="Yaw angle (Blender) in degrees"
    )

    args = parser.parse_args()
    blender_euler = [args.roll, args.pitch, args.yaw]
    opencv_euler = blender_to_opencv_euler(blender_euler)

    print("OpenCV Euler Angles (XYZ degrees):", opencv_euler)


if __name__ == "__main__":
    main()
