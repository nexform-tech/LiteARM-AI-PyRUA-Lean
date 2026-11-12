"""Recompute perception and plan a lift from saved evidence; never connect to a robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyrualean.hardware.camera_geometry import CAMERA_FRAME, station_transform
from pyrualean.hardware.config import load_station
from pyrualean.hardware.kinematics import LitearmKinematics
from pyrualean.hardware.perception import FrameData, Perception, Segment, _depth_points
from pyrualean.hardware.planning import Planner, PlanningError


class CachedSegmenter:
    def __init__(self, directory: Path):
        self.directory = directory

    def segment(self, image, *, text_prompt, **kwargs):
        path = self.directory / (text_prompt.replace(" ", "_") + "_mask.npy")
        return Segment(True, mask=np.load(path).astype(bool))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--use-calibrated-camera-poses",
        action="store_true",
        help="rebuild camera poses/intrinsics from calibration and recorded capture joints",
    )
    args = parser.parse_args()
    station = load_station(args.config)
    step = args.episode / "steps" / f"{args.step:06d}"
    metadata = json.loads((step / "camera_meta.json").read_text())
    frames = {}
    for name, meta in metadata.items():
        with Image.open(step / "artifacts" / meta["color_artifact"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        depth_name = meta.get("depth_artifact")
        pose = station_transform(meta["camera_to_world"], meta.get("camera_frame", CAMERA_FRAME))
        intrinsic = np.asarray(meta["intrinsic_matrix"])
        if args.use_calibrated_camera_poses:
            cam = station.calibration.camera(name)
            if cam.resolution != rgb.shape[:2]:
                raise ValueError("replay image does not match the selected camera calibration")
            if cam.mount == "world":
                pose = cam.transform
            else:
                q = meta.get("q_model_at_capture")
                if q is None:
                    raise ValueError("calibrated wrist replay requires recorded capture joints")
                pose = LitearmKinematics(station.calibration).camera_pose(name, cam.arm, q)
            intrinsic = cam.intrinsic
        frames[name] = FrameData(
            name,
            rgb,
            np.load(step / "artifacts" / depth_name) if depth_name else None,
            intrinsic,
            pose,
            meta["captured_at_wall_ns"],
        )
    segmenter = CachedSegmenter(args.mask_dir)
    perception = Perception(station.calibration, station.sam3, segmenter=segmenter)
    camera = station.primary_camera
    report = {
        "episode": str(args.episode.resolve()),
        "step": args.step,
        "config_sha256": station.sha256,
        "calibration_sha256": station.calibration_sha256,
        "device_actions": 0,
        "camera_frame": CAMERA_FRAME,
        "camera_calibration_replayed": args.use_calibrated_camera_poses,
        "perception": {},
        "planning": {},
    }
    for prompt in args.prompt:
        mask = segmenter.segment(frames[camera].rgb, text_prompt=prompt).mask
        points, ratio = _depth_points(frames[camera], mask)
        report["perception"][prompt] = {
            "raw_depth_valid_ratio": ratio,
            "raw_world_z_quantiles": np.percentile(points[:, 2], [2, 50, 97]).tolist()
            if len(points)
            else None,
            "corrected": perception.locate({camera: frames[camera]}, prompt, camera),
        }
    device = json.loads((step / "device.json").read_text())
    arm = station.execution.active_arms[0]
    q = device[arm]["q_model"]
    planner = Planner(station.calibration, station.execution, station.safety)
    target = planner.kinematics.fk(arm, q)[:3, 3] + [0, 0, 0.08]
    sight = report["perception"][args.prompt[0]]["corrected"]
    variants = {"position_only_lift": {}}
    if sight.get("grasp_yaw") is not None:
        variants["top_grasp_lift"] = {"pitch": 90, "yaw": sight["grasp_yaw"], "roll": 0}
    for name, orientation in variants.items():
        try:
            plan = planner.plan_move(arm, q, target, path="line", **orientation)
            report["planning"][name] = {
                "waypoints": len(plan.waypoints),
                "path": plan.path,
                "note": plan.note,
                "duration_s": plan.estimated_duration_s,
                "target_angles": plan.target_angles,
                "validation_errors": planner.validate_waypoints(arm, q, plan.waypoints),
            }
        except PlanningError as exc:
            report["planning"][name] = {"error": str(exc), "detail": exc.detail}
    if "top_grasp_lift" in variants:
        try:
            lift = planner.plan_move(arm, q, target, path="line")
            lifted_q = lift.waypoints[-1]
            lifted_xyz = planner.kinematics.fk(arm, lifted_q)[:3, 3]
            orient = planner.plan_move(
                arm, lifted_q, lifted_xyz, path="auto", **variants["top_grasp_lift"]
            )
            report["planning"]["reorient_after_lift"] = {
                "path": orient.path,
                "waypoints": len(orient.waypoints),
                "duration_s": orient.estimated_duration_s,
                "note": orient.note,
                "validation_errors": planner.validate_waypoints(arm, lifted_q, orient.waypoints),
            }
        except PlanningError as exc:
            report["planning"]["reorient_after_lift"] = {"error": str(exc), "detail": exc.detail}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
