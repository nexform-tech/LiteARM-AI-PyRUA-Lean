"""Import the operator's PARK eye-in-hand result into a review-only station profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .camera_geometry import CAMERA_FRAME, OPTICAL_FRAME, station_transform
from .config import load_calibration, load_station


def import_park(calibration, document, *, camera="cam_right_wrist", source="operator PARK"):
    """Rebuild this camera from PARK, rather than flipping an existing transform."""
    if document.get("mode") != "eye_in_hand" or document.get("selected_method") != "PARK":
        raise ValueError("an eye_in_hand PARK result is required")
    info = document.get("conventions", {})
    if info.get("transform_notation") != "T_A_B maps points from frame B into frame A":
        raise ValueError("unsupported hand-eye transform notation")
    if info.get("units", {}).get("translation") != "m (per-pose residuals in mm)":
        raise ValueError("hand-eye transform translations must be metres")
    cam = calibration.camera(camera)
    if cam.mount != "tool" or cam.arm is None:
        raise ValueError("PARK eye-in-hand needs a tool-mounted camera")
    optical = np.asarray(document["transforms"]["T_flange_cam"]["matrix"], dtype=float)
    transform = np.linalg.inv(calibration.arm(cam.arm).flange_to_tcp) @ station_transform(
        optical, OPTICAL_FRAME
    )
    # This station's 640x480 stream is the centred crop of the 1280x720
    # calibration FOV after uniform height scaling. Record that mapping,
    # rather than retaining intrinsics from an unrelated RealSense profile.
    source_camera = document["camera"]
    width, height = int(source_camera["width"]), int(source_camera["height"])
    scale = cam.height / height
    crop_x = (width * scale - cam.width) / 2
    if crop_x < -1e-9:
        raise ValueError("target image needs padding; a measured pixel mapping is required")
    mapping = np.array([[scale, 0, -crop_x], [0, scale, 0], [0, 0, 1]])
    intrinsic = mapping @ np.asarray(source_camera["intrinsic_matrix"], dtype=float)
    raw = copy.deepcopy(dict(calibration.raw))
    for name, calibrated_camera in calibration.cameras.items():
        key = "camera_to_world" if calibrated_camera.mount == "world" else "camera_to_tool"
        raw["cameras"][name][key] = calibrated_camera.transform.tolist()
        raw["cameras"][name]["camera_frame"] = CAMERA_FRAME
    item = raw["cameras"][camera]
    item.update(
        camera_to_tool=transform.tolist(),
        camera_frame=CAMERA_FRAME,
        intrinsic_matrix=intrinsic.tolist(),
        distortion_coefficients=source_camera["distortion_coefficients"],
    )
    item["handeye_source"] = {
        "method": "PARK",
        "path": source,
        "created_at": document["created_at"],
        "sha256": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
        "source_camera_frame": OPTICAL_FRAME,
        "runtime_camera_frame": CAMERA_FRAME,
        "pixel_mapping": "uniform-height-scale-and-centred-horizontal-crop",
        "pixel_mapping_matrix": mapping.tolist(),
        "source_resolution": [height, width],
        "runtime_resolution": list(cam.resolution),
        "metrics": document.get("metrics", {}),
    }
    raw["executable"] = False
    raw["calibration_version"] = calibration.version + "-park-unified-review"
    raw["source"] = f"{calibration.source} | canonical PARK import from {source}"
    raw.setdefault("quality", {})["status"] = "park-unified-review-not-live-verified"
    raw["quality"]["wrist_camera_handeye"] = (
        "PARK is the only wrist extrinsic/intrinsic source. OpenCV-to-station axes are "
        "converted once. Pixel crop mapping and physical accuracy require field verification."
    )
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", type=Path, required=True)
    parser.add_argument("--handeye", type=Path, required=True)
    parser.add_argument("--out-station", type=Path, required=True)
    parser.add_argument("--out-calibration", type=Path, required=True)
    args = parser.parse_args(argv)
    station = load_station(args.station)
    if args.out_station.exists() or args.out_calibration.exists():
        parser.error("review outputs already exist; choose new output paths")
    document = json.loads(args.handeye.read_text())
    calibration = import_park(station.calibration, document, source=str(args.handeye.resolve()))
    args.out_calibration.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_calibration.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n")
    try:
        load_calibration(
            temporary, arms=station.execution.active_arms, cameras=tuple(station.cameras)
        )
        temporary.replace(args.out_calibration)
    finally:
        temporary.unlink(missing_ok=True)
    profile = copy.deepcopy(dict(station.raw))
    profile["calibration"] = str(args.out_calibration.resolve())
    profile.setdefault("execution", {})["mode"] = "read-only"
    profile.setdefault("manipulation", {}).update(enable_cube_grasp=False, enable_cube_stack=False)
    args.out_station.parent.mkdir(parents=True, exist_ok=True)
    args.out_station.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    load_station(args.out_station)
    print(
        json.dumps(
            {
                "station": str(args.out_station),
                "calibration": str(args.out_calibration),
                "execution_mode": "read-only",
                "cube_primitives_enabled": False,
            }
        )
    )


if __name__ == "__main__":
    main()
