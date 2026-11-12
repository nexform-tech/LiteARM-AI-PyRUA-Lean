"""A bounded cube stack attempt with support geometry frozen before grasping."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .grasp import _hold_sample, grasp_cube
from .kinematics import rotation_error
from .motion import _require_active, _require_gripper, move_pose, place_on
from .planning import PlanningError
from .rpc import RpcError

POSITION_TOL_M = 0.003
HEIGHT_TOL_M = 0.003
ANGLE_TOL_DEG = 3.0


def _measurement(toolkit, prompt, camera):
    result = toolkit._locate({"text_prompt": prompt, "camera": camera})
    quality = result.get("measurement_quality") or {}
    if (
        not result.get("found")
        or result.get("measurement_failed", True)
        or result.get("source") not in ("depth", "multiview")
        or any(
            quality.get(k) is False
            for k in ("height_reliable", "center_reliable", "footprint_reliable")
        )
    ):
        raise ValueError(f"{prompt}: reliable 3-D measurement required; no pixel/plane fallback")
    try:
        xyz = np.asarray(result["center_xyz"], dtype=float)
        values = [
            float(result[k]) for k in ("top_z", "height_m", "length_m", "width_m", "grasp_yaw")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{prompt}: incomplete object geometry") from exc
    if (
        xyz.shape != (3,)
        or not np.all(np.isfinite(xyz))
        or not all(math.isfinite(v) for v in values)
    ):
        raise ValueError(f"{prompt}: non-finite object geometry")
    if any(v <= 0 for v in values[1:4]):
        raise ValueError(f"{prompt}: positive dimensions required")
    result["center_xyz"] = xyz.tolist()
    for name, value in zip(
        ("top_z", "height_m", "length_m", "width_m", "grasp_yaw"), values, strict=True
    ):
        result[name] = value
    return copy.deepcopy(result)


def _consistent(first, second):
    return (
        np.linalg.norm(np.asarray(first["center_xyz"][:2]) - second["center_xyz"][:2])
        <= POSITION_TOL_M
        and abs(first["top_z"] - second["top_z"]) <= HEIGHT_TOL_M
        and abs(first["width_m"] - second["width_m"]) <= POSITION_TOL_M
        and abs(first["length_m"] - second["length_m"]) <= POSITION_TOL_M
    )


def stack_cube(
    toolkit: Any,
    source_prompt: str,
    support_prompt: str,
    *,
    pad_height_m: float,
    arm: str = "right",
    camera: str = "cam_head",
    alignment_camera: str | None = None,
    carry_clearance_m: float = 0.03,
    overlap_fraction: float = 0.65,
) -> dict[str, Any]:
    """Measure twice before grasp; lift, move horizontally once, verify and place.

    New support/held-object reads after transport are verification only. They
    never replace the frozen landing target or cause exploratory corrections.
    A failed alignment keeps the object held and refuses the final descent.
    """
    if toolkit.config.raw.get("manipulation", {}).get("enable_cube_stack") is not True:
        raise ValueError(
            "stack_cube is disabled; explicit manipulation.enable_cube_stack is required"
        )
    if toolkit.config.raw.get("manipulation", {}).get("enable_cube_grasp") is not True:
        raise ValueError("grasp_cube is disabled; enable both cube primitives explicitly")
    _require_active(toolkit, arm)
    _require_gripper(toolkit, arm)
    if not isinstance(source_prompt, str) or not isinstance(support_prompt, str):
        raise ValueError("source_prompt and support_prompt must be strings")
    if not source_prompt.strip() or not support_prompt.strip() or source_prompt == support_prompt:
        raise ValueError("distinct, non-empty source and support prompts are required")
    carry_clearance_m = float(carry_clearance_m)
    if not math.isfinite(carry_clearance_m) or not 0.02 <= carry_clearance_m <= 0.10:
        raise ValueError("carry_clearance_m must be in [0.02, 0.10]")
    alignment_camera = alignment_camera or f"cam_{arm}_wrist"
    result = {
        "arm": arm,
        "completed": False,
        "failure_stage": None,
        "note": None,
        "support_xyz": None,
        "support_top": None,
        "landing_tcp_xy": None,
        "held_offset_xy": None,
        "carry_z": None,
        "aligned": False,
        "placed": False,
        "released": False,
        "safe_to_home": False,
        "steps_used": 0,
        "measured": True,
        "stages": [],
        "error_code": None,
    }
    start_steps = toolkit.device_steps
    stage = "measure"
    try:
        # Four reads occur with the empty arm, before any grip or move.
        source = _measurement(toolkit, source_prompt, camera)
        support = _measurement(toolkit, support_prompt, camera)
        source_check = _measurement(toolkit, source_prompt, camera)
        support_check = _measurement(toolkit, support_prompt, camera)
        if not _consistent(source, source_check) or not _consistent(support, support_check):
            raise ValueError("pre-grasp object measurements disagree; no motion was issued")
        for item in (source, support):
            table = toolkit.calibration.table
            x, y = item["center_xyz"][:2]
            half = max(item["length_m"], item["width_m"]) / 2
            if not (
                table.x_min_m + half <= x <= table.x_max_m - half
                and table.y_min_m + half <= y <= table.y_max_m - half
            ):
                result["error_code"] = "outside_calibrated_table"
                result["stages"].append(
                    {
                        "stage": "table_bounds",
                        "center_xyz": item["center_xyz"],
                        "table_xy_bounds": [
                            [table.x_min_m, table.x_max_m],
                            [table.y_min_m, table.y_max_m],
                        ],
                    }
                )
                raise ValueError(
                    f"object measured at ({x:.4f}, {y:.4f}) is outside the calibrated table "
                    f"footprint (y=[{table.y_min_m:.4f}, {table.y_max_m:.4f}]); "
                    "this is an operating-region refusal, not an IK reachability result. "
                    "Confirm/recalibrate the physical support or move the object into that region."
                )
        distance = np.linalg.norm(np.asarray(source["center_xyz"][:2]) - support["center_xyz"][:2])
        if (
            distance
            <= (
                max(source["length_m"], source["width_m"])
                + max(support["length_m"], support["width_m"])
            )
            / 2
        ):
            raise ValueError("source and support measurements overlap or identify the same object")
        result["support_xyz"] = support["center_xyz"]
        result["support_top"] = support["top_z"]
        result["stages"].append(
            {
                "stage": "frozen_geometry",
                "source": source,
                "support": support,
                "calibration_sha256": toolkit.config.calibration_sha256,
            }
        )
        stage = "grasp"
        grasp = grasp_cube(
            toolkit,
            source["center_xyz"][:2],
            source["top_z"],
            toolkit.calibration.table.z_m,
            source["width_m"],
            source["grasp_yaw"],
            pad_height_m=pad_height_m,
            arm=arm,
            overlap_fraction=overlap_fraction,
        )
        result["stages"].append({"stage": stage, "result": grasp})
        if not grasp["grasped"]:
            raise ValueError(f"grasp failed: {grasp.get('failure_stage')}: {grasp.get('note')}")
        offset = np.asarray(source["center_xyz"][:2]) - np.asarray(grasp["grasp_xyz"][:2])
        landing = np.asarray(support["center_xyz"][:2]) - offset
        bottom_offset = grasp["tcp_to_bottom"]
        result["held_offset_xy"] = offset.tolist()
        result["landing_tcp_xy"] = landing.tolist()
        obs = toolkit.refresh_observation(tool="stack_cube_grasped")
        pose = np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)
        fixed_rotation = pose[:3, :3].copy()
        carry_z = max(
            float(pose[2, 3]),
            max(source["top_z"], support["top_z"]) + bottom_offset + carry_clearance_m,
        )
        result["carry_z"] = carry_z

        def validate(plan, before, *, horizontal=False):
            start_pose = np.asarray(before.arm(arm).tcp_pose).reshape(4, 4)
            start_xy = start_pose[:2, 3]
            travel = landing - start_xy
            denominator = float(travel @ travel)
            for q in plan.waypoints:
                at = toolkit.kinematics.fk(arm, q)
                if rotation_error(at[:3, :3], fixed_rotation) > ANGLE_TOL_DEG:
                    raise PlanningError("carry path changes the frozen held-object orientation")
                if horizontal:
                    progress = (
                        float((at[:2, 3] - start_xy) @ travel / denominator) if denominator else 0
                    )
                    on_line = start_xy + np.clip(progress, 0, 1) * travel
                    if (
                        abs(at[2, 3] - carry_z) > 0.002
                        or np.linalg.norm(at[:2, 3] - on_line) > POSITION_TOL_M
                    ):
                        raise PlanningError("carry is not a horizontal line at the safe height")
                elif np.linalg.norm(at[:2, 3] - start_xy) > POSITION_TOL_M:
                    raise PlanningError("clearance lift is not vertical")
                if at[2, 3] - bottom_offset < toolkit.calibration.table.z_m + 0.003:
                    raise PlanningError("held-object bottom violates support clearance")

        stage = "clearance_lift"
        up = [*pose[:2, 3], carry_z]
        moved = move_pose(
            toolkit, up, arm=arm, path="line", _half_turn=False, _validate_plan=validate
        )
        result["stages"].append({"stage": stage, "result": moved})
        if not moved["reached"] or moved["blocked"] or moved["slip"]:
            raise ValueError("safe vertical lift did not arrive; horizontal carry refused")
        stage = "horizontal_carry"
        moved = move_pose(
            toolkit,
            [*landing, carry_z],
            arm=arm,
            path="line",
            _half_turn=False,
            _validate_plan=lambda p, o: validate(p, o, horizontal=True),
        )
        result["stages"].append({"stage": stage, "result": moved})
        if not moved["reached"] or moved["blocked"] or moved["slip"]:
            raise ValueError("horizontal carry did not arrive; placement refused")
        stage = "alignment"
        obs, held = _hold_sample(toolkit, arm)
        at = np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)
        estimated_center = at[:2, 3] + offset
        if (
            not held["valid"]
            or abs(held["width_m"] - grasp["baseline_width_m"]) > 0.001
            or np.linalg.norm(estimated_center - support["center_xyz"][:2]) > POSITION_TOL_M
            or abs(at[2, 3] - carry_z) > 0.002
            or rotation_error(at[:3, :3], fixed_rotation) > ANGLE_TOL_DEG
        ):
            raise ValueError("measured TCP/hold does not align the frozen object centre")
        # Exactly one check of each object. Missing/contradictory geometry
        # stops here; no pixels or exploratory viewpoint moves are requested.
        support_now = _measurement(toolkit, support_prompt, camera)
        source_now = _measurement(toolkit, source_prompt, alignment_camera)
        if not _consistent(support, support_now):
            raise ValueError("support changed or cannot be confirmed at its frozen location")
        if (
            np.linalg.norm(np.asarray(source_now["center_xyz"][:2]) - support["center_xyz"][:2])
            > POSITION_TOL_M
        ):
            raise ValueError("visual object centres are not aligned; no automatic correction")
        expected_top = float(
            at[2, 3] - bottom_offset + source["top_z"] - toolkit.calibration.table.z_m
        )
        if abs(source_now["top_z"] - expected_top) > HEIGHT_TOL_M:
            raise ValueError("held-object height/offset cannot be confirmed")
        result["aligned"] = True
        result["stages"].append(
            {
                "stage": stage,
                "support_check": support_now,
                "held_check": source_now,
                "estimated_center_xy": estimated_center.tolist(),
            }
        )
        stage = "place"
        placed = place_on(
            toolkit,
            *landing,
            support["top_z"],
            bottom_offset,
            arm=arm,
            clearance=0.003,
            lift_m=0.08,
            retreat_xy=(0.07, 0),
            path="line",
        )
        result["stages"].append({"stage": stage, "result": placed})
        result.update({key: bool(placed[key]) for key in ("placed", "released", "safe_to_home")})
        if not result["placed"] or not result["released"] or not result["safe_to_home"]:
            raise ValueError(placed.get("note") or "placement/release/retreat was incomplete")
        result["completed"] = True
    except (ValueError, PlanningError, RuntimeError, RpcError) as exc:
        result["failure_stage"] = stage
        result["note"] = str(exc)
    finally:
        result["steps_used"] = toolkit.device_steps - start_steps
    return result
