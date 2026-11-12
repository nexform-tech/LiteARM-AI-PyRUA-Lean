"""One station camera frame: +x right, +y up, viewing along -z.

OpenCV optical (+x right, +y down, +z forward) transforms are converted
once at the calibration/observation boundary, never by policy code.
"""

from __future__ import annotations

import numpy as np

CAMERA_FRAME = "opengl"
OPTICAL_FRAME = "opencv"
OPTICAL_TO_STATION = np.diag([1.0, -1.0, -1.0, 1.0])


def station_transform(transform, source_frame=CAMERA_FRAME):
    """Return a camera-to-parent matrix expressed in the station camera frame."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera transform must be a finite 4x4 matrix")
    if source_frame == CAMERA_FRAME:
        return matrix.copy()
    if source_frame == OPTICAL_FRAME:
        return matrix @ OPTICAL_TO_STATION
    raise ValueError("camera_frame must be 'opengl' or 'opencv'")


def camera_ray(intrinsic, row, col):
    k = np.asarray(intrinsic)
    return np.array([(col - k[0, 2]) / k[0, 0], -(row - k[1, 2]) / k[1, 1], -1.0])


def project_points(intrinsic, camera_to_world, points):
    k, pose = np.asarray(intrinsic), np.asarray(camera_to_world)
    local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    depth = -local[:, 2]
    safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    return (-k[1, 1] * local[:, 1] / safe + k[1, 2], k[0, 0] * local[:, 0] / safe + k[0, 2], depth)


def depth_points(intrinsic, camera_to_world, rows, cols, depth):
    k, pose = np.asarray(intrinsic), np.asarray(camera_to_world)
    local = np.stack(
        [(cols - k[0, 2]) / k[0, 0] * depth, -(rows - k[1, 2]) / k[1, 1] * depth, -depth], axis=-1
    )
    return local @ pose[:3, :3].T + pose[:3, 3]


def ray_plane(intrinsic, camera_to_world, row, col, plane_z):
    pose = np.asarray(camera_to_world)
    direction = pose[:3, :3] @ camera_ray(intrinsic, row, col)
    if abs(direction[2]) < 1e-9:
        return None
    distance = (plane_z - pose[2, 3]) / direction[2]
    return pose[:3, 3] + distance * direction if distance > 0 else None
