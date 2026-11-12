"""Compare SDK conventions, official URDF and recorded plans without opening hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from pyrualean.hardware.config import load_calibration, load_station
from pyrualean.hardware.kinematics import (
    LitearmKinematics,
    _origin,
    matrix_to_rpy,
    orientation,
    rotation_error,
    rpy_matrix,
    slerp,
)
from pyrualean.hardware.planning import Planner, PlanningError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--official-description", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from litearm import _rot, tool

    sdk_file = Path(_rot.__file__).parent / "arm.py"
    source = args.sdk_root / "src/litearm/arm.py"
    report = {
        "device_actions": 0,
        "sdk_source": str(args.sdk_root.resolve()),
        "installed_sdk": str(sdk_file),
        "arm_source_matches_installed": source.read_bytes() == sdk_file.read_bytes(),
        "arm_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    station = load_station(args.config)
    kin = LitearmKinematics(station.calibration)
    planner = Planner(station.calibration, station.execution, station.safety, kin)
    with zipfile.ZipFile(args.official_description) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".urdf"))
        official = ET.fromstring(archive.read(member))
    packaged = ET.parse(kin.urdf).getroot()
    official_joints = {joint.get("name"): joint for joint in official.findall("joint")}
    local_joints = {joint.get("name"): joint for joint in packaged.findall("joint")}
    report["urdf_joint_comparison"] = {}
    for name in [*(f"joint{i}" for i in range(1, 8)), "ee_link"]:
        a, b = official_joints[name], local_joints[name]
        axis_a, axis_b = a.find("axis"), b.find("axis")
        report["urdf_joint_comparison"][name] = {
            "origin_max_abs_diff": float(np.max(np.abs(_origin(a) - _origin(b)))),
            "axis_equal": (axis_a.attrib if axis_a is not None else None)
            == (axis_b.attrib if axis_b is not None else None),
        }
    arm = station.execution.active_arms[0]
    cal = station.calibration.arm(arm)
    offset, rotation = tool.as_offset(
        cal.flange_to_tcp[:3, 3], matrix_to_rpy(cal.flange_to_tcp[:3, :3])
    )
    rng = np.random.default_rng(34)
    limits = np.asarray(cal.joint_limits_rad)
    errors = []
    for _ in range(30):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        flange = np.linalg.inv(cal.base_pose) @ kin.flange_pose(arm, q)
        pose = [*flange[:3, 3], *matrix_to_rpy(flange[:3, :3])]
        sdk_pose = tool.to_tool(pose, offset, rotation)
        expected = np.linalg.inv(cal.base_pose) @ kin.fk(arm, q)
        other = rpy_matrix(rng.uniform(-2, 2, 3))
        errors.append(
            [
                np.linalg.norm(sdk_pose[:3] - expected[:3, 3]),
                rotation_error(rpy_matrix(sdk_pose[3:]), expected[:3, :3]),
                np.max(
                    np.abs(
                        slerp(flange[:3, :3], other, 0.25)
                        - _rot.rot_slerp(flange[:3, :3], other, 0.25)
                    )
                ),
            ]
        )
    report["sdk_transform_check_30_poses"] = dict(
        zip(
            ["max_tcp_position_error_m", "max_tcp_rotation_error_deg", "max_slerp_matrix_error"],
            np.max(errors, axis=0).tolist(),
            strict=True,
        )
    )
    q = json.loads((args.episode / "steps" / f"{args.step:06d}" / "device.json").read_text())[arm][
        "q_model"
    ]
    report["source_episode"] = str(args.episode.resolve())
    report["source_step"] = args.step
    report["tool_table_constraint"] = planner.collision.tool_table_constraint(
        arm, orientation(90, 14.4, 0)
    )
    report["height_candidates"] = []
    for z in [0.162, 0.168, 0.170, 0.171, 0.196]:
        for path in ["line", "auto"]:
            result = {"target_xyz": [-0.036, 0.432, z], "requested_path": path}
            try:
                plan = planner.plan_move(
                    arm, q, result["target_xyz"], pitch=90, yaw=14.4, roll=0, path=path
                )
                result.update(
                    {
                        "feasible": True,
                        "selected_path": plan.path,
                        "waypoint_count": len(plan.waypoints),
                        "note": plan.note,
                        "duration_s": plan.estimated_duration_s,
                        "planned_xyz": kin.fk(arm, plan.waypoints[-1])[:3, 3].tolist(),
                        "validation_errors": planner.validate_waypoints(arm, q, plan.waypoints),
                    }
                )
            except PlanningError as exc:
                result.update({"feasible": False, "error": str(exc), "detail": exc.detail})
            report["height_candidates"].append(result)
    root = Path(__file__).resolve().parents[1]
    probe_dir = root / "runs_hw/bringup-single-arm"
    probe = json.loads((probe_dir / "read-only-probe/20261106-093838/arm_probe.json").read_text())
    old = LitearmKinematics(load_calibration(probe_dir / "offline-fk/identity_candidate.json"))
    flange = old.flange_pose("left", probe["state"]["q"])
    report["recorded_firmware_fk_check"] = {
        "position_error_m": float(np.linalg.norm(flange[:3, 3] - probe["tcp_value"][:3])),
        "rotation_error_deg": rotation_error(flange[:3, :3], rpy_matrix(probe["tcp_value"][3:])),
        "note": "Single measured pose from 2026-11-06, not a new firmware IK comparison.",
        "joint_limits_match_recorded": all(
            (param["q_min"], param["q_max"]) == tuple(cal.joint_limits_rad[index])
            for index, param in enumerate(probe["joint_params"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
