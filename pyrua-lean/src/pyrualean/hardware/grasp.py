"""Centred vertical cube pinches with measured engagement and a settled hold baseline.

Object and usable pad dimensions are supplied from measurements. This module
does not infer physical contact from collision capsules or import a device SDK.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import numpy as np

from .kinematics import orientation
from .motion import _require_active, _require_gripper, grip, move_pose
from .planning import PlanningError
from .rpc import RpcError

ALIGN_M = 0.002
ALIGN_DEG = 3.0
WIDTH_TOL_M = 0.003
STABLE_WIDTH_M = 0.0005
SAMPLE_S = 0.25
MIN_WAIT_S = 1.0
STABLE_WINDOW_S = 0.5


def cube_geometry(
    center_xy: Any,
    top_z: float,
    support_top: float,
    side_m: float,
    yaw: float,
    *,
    pad_height_m: float,
    overlap_fraction: float = 0.65,
    clearance_m: float = 0.003,
    approach_m: float = 0.06,
    lift_m: float = 0.015,
) -> dict[str, Any]:
    """Compute a fingertip target; reserve 1 mm for measured arrival error."""
    xy = tuple(float(v) for v in center_xy)
    values = {
        name: float(value)
        for name, value in (
            ("top_z", top_z),
            ("support_top", support_top),
            ("side_m", side_m),
            ("yaw", yaw),
            ("pad_height_m", pad_height_m),
            ("overlap_fraction", overlap_fraction),
            ("clearance_m", clearance_m),
            ("approach_m", approach_m),
            ("lift_m", lift_m),
        )
    }
    if len(xy) != 2 or not all(math.isfinite(v) for v in (*xy, *values.values())):
        raise ValueError("cube geometry must contain finite numbers and two centre coordinates")
    (
        top_z,
        support_top,
        side_m,
        yaw,
        pad_height_m,
        overlap_fraction,
        clearance_m,
        approach_m,
        lift_m,
    ) = (
        values[name]
        for name in (
            "top_z",
            "support_top",
            "side_m",
            "yaw",
            "pad_height_m",
            "overlap_fraction",
            "clearance_m",
            "approach_m",
            "lift_m",
        )
    )
    height = top_z - support_top
    if height <= 0 or side_m <= 0 or pad_height_m <= 0:
        raise ValueError("object height, side_m and pad_height_m must be positive")
    if not 0.5 < overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be greater than 0.5 and at most 1")
    if clearance_m < 0.003 or approach_m < 0.01 or not 0.005 <= lift_m <= 0.03:
        raise ValueError(
            "clearance_m >= 0.003, approach_m >= 0.01, lift_m in [0.005, 0.03] required"
        )
    required = height * overlap_fraction
    if pad_height_m < required:
        raise ValueError("usable pad height cannot engage the requested fraction of the object")
    tip_z = top_z - required - 0.001
    if tip_z < support_top + clearance_m:
        raise ValueError("support clearance conflicts with the requested engagement depth")
    return {
        "center_xy": list(xy),
        "top_z": top_z,
        "support_top": support_top,
        "height_m": height,
        "side_m": side_m,
        "yaw": yaw,
        "pad_height_m": pad_height_m,
        "required_overlap_m": required,
        "clearance_m": clearance_m,
        "target_xyz": [*xy, tip_z],
        "approach_xyz": [*xy, top_z + approach_m],
        "lift_m": lift_m,
    }


def _pose_check(pose: Any, geometry: dict[str, Any], width: float, *, engage: bool) -> dict:
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    point, jaw = pose[:3, 0], pose[:3, 1]
    wanted_jaw = orientation(90.0, geometry["yaw"], 0.0)[:, 1]
    vertical_deg = math.degrees(math.acos(np.clip(-point[2], -1.0, 1.0)))
    parallel_deg = math.degrees(math.acos(np.clip(abs(np.dot(jaw, wanted_jaw)), 0.0, 1.0)))
    centre_error = float(np.linalg.norm(pose[:2, 3] - geometry["center_xy"]))
    # A tilted jaw puts its two tips at different heights. Use the lower tip
    # for support clearance and the higher tip for the minimum engagement.
    tip_low = float(pose[2, 3] - abs(jaw[2]) * width / 2)
    tip_high = float(pose[2, 3] + abs(jaw[2]) * width / 2)
    overlap = max(0.0, min(geometry["pad_height_m"] * -point[2], geometry["top_z"] - tip_high))
    valid = centre_error <= ALIGN_M and max(vertical_deg, parallel_deg) <= ALIGN_DEG
    valid = valid and tip_low >= geometry["support_top"] + geometry["clearance_m"] - 1e-9
    if engage:
        valid = valid and overlap >= geometry["required_overlap_m"] - 1e-9
    return {
        "valid": bool(valid),
        "center_error_m": centre_error,
        "vertical_error_deg": vertical_deg,
        "parallel_error_deg": parallel_deg,
        "tip_low_z": tip_low,
        "overlap_m": overlap,
        "overlap_fraction": overlap / geometry["height_m"],
    }


def _hold_sample(toolkit: Any, arm: str) -> tuple[Any, dict[str, Any]]:
    obs = toolkit.refresh_observation(tool="grasp_cube_hold")
    a = obs.arm(arm)
    values = (a.gripper_width_m, a.gripper_opening, a.gripper_force_n)
    valid = (
        not toolkit._closed
        and not obs.fault
        and not a.fault
        and a.enabled
        and a.gripper_enabled
        and a.gripper_fresh
        and a.gripper_command == "closed"
        and all(v is not None and math.isfinite(v) for v in values)
    )
    return obs, {
        "valid": bool(valid),
        "step": obs.step,
        "width_m": a.gripper_width_m,
        "opening": a.gripper_opening,
        "nominal_force_n": a.gripper_force_n,
    }


def grasp_cube(
    toolkit: Any,
    center_xy: Any,
    top_z: float,
    support_top: float,
    side_m: float,
    yaw: float,
    *,
    pad_height_m: float,
    arm: str = "right",
    overlap_fraction: float = 0.65,
    clearance_m: float = 0.003,
    approach_m: float = 0.06,
    lift_m: float = 0.015,
    stable_timeout_s: float = 5.0,
) -> dict[str, Any]:
    """One attempt: align, descend, close, settle, baseline, vertical test lift.

    Failures before lifting leave the object on its support and release a
    newly started squeeze when the service is healthy. Failures during/after
    lifting preserve the squeeze: this call never drops or homes a held cube.
    """
    if toolkit.config.raw.get("manipulation", {}).get("enable_cube_grasp") is not True:
        raise ValueError(
            "grasp_cube is disabled; explicit manipulation.enable_cube_grasp is required"
        )
    _require_active(toolkit, arm)
    _require_gripper(toolkit, arm)
    geometry = cube_geometry(
        center_xy,
        top_z,
        support_top,
        side_m,
        yaw,
        pad_height_m=pad_height_m,
        overlap_fraction=overlap_fraction,
        clearance_m=clearance_m,
        approach_m=approach_m,
        lift_m=lift_m,
    )
    support_top, yaw = geometry["support_top"], geometry["yaw"]
    stable_timeout_s = float(stable_timeout_s)
    if not math.isfinite(stable_timeout_s) or not 1.5 <= stable_timeout_s <= 10.0:
        raise ValueError("stable_timeout_s must be in [1.5, 10]")
    cal = toolkit.calibration.arm(arm)
    result = {
        "arm": arm,
        "grasped": False,
        "failure_stage": None,
        "note": None,
        "target_xyz": geometry["target_xyz"],
        "final_xyz": None,
        "overlap_fraction": None,
        "baseline_opening": None,
        "baseline_width_m": None,
        "lift_reached": False,
        "slip": False,
        "holding": False,
        "tcp_to_bottom": None,
        "steps_used": 0,
        "measured": True,
        "stages": [],
    }
    start_steps = toolkit.device_steps
    stage = "geometry"
    squeeze_started = False
    lift_started = False
    try:
        reference = toolkit.calibration.raw["arms"][arm].get("tcp_reference")
        if reference != "fingertip_midpoint" or abs(cal.tool_collision.tip_offset_m) > 1e-9:
            raise ValueError("grasp_cube requires a fingertip_midpoint TCP with zero tip offset")
        table = toolkit.calibration.table
        half_side = geometry["side_m"] / 2
        x, y = geometry["center_xy"]
        if not (
            table.x_min_m + half_side <= x <= table.x_max_m - half_side
            and table.y_min_m + half_side <= y <= table.y_max_m - half_side
        ):
            raise ValueError("cube footprint is outside the calibrated table bounds")
        constraint = toolkit.planner.collision.tool_table_constraint(arm, orientation(90, yaw, 0))
        floor = max(constraint["tcp_min_z_m"], table.z_floor_m)
        if geometry["target_xyz"][2] < floor:
            raise ValueError("engagement target is below the configured tool/table bound")
        open_width = geometry["side_m"] + 0.012
        if open_width > cal.gripper.open_m:
            raise ValueError("cube plus approach clearance exceeds the calibrated gripper stroke")
        if geometry["side_m"] <= cal.gripper.closed_m + cal.gripper.min_holding_m + WIDTH_TOL_M:
            raise ValueError("cube width is too close to the empty-close limit for verification")
        obs = toolkit.refresh_observation(tool="grasp_cube_start")
        if toolkit._closed or obs.fault or obs.arm(arm).fault or not obs.arm(arm).enabled:
            raise ValueError("hardware is not ready for a grasp")
        measured_grip = obs.arm(arm)
        if arm in toolkit.held_openings or (
            measured_grip.gripper_command == "closed"
            and measured_grip.gripper_width_m is not None
            and measured_grip.gripper_width_m > cal.gripper.closed_m + cal.gripper.min_holding_m
        ):
            raise ValueError(
                "grasp_cube requires an empty gripper; release the existing hold first"
            )

        def validate_plan(plan: Any, before: Any, *, vertical: bool, engage: bool) -> None:
            poses = [toolkit.kinematics.fk(arm, q) for q in plan.waypoints]
            checked = poses if vertical else poses[-1:]
            if any(
                not _pose_check(pose, geometry, open_width, engage=False)["valid"]
                for pose in checked
            ):
                raise PlanningError(
                    "grasp path does not preserve centre, face alignment or support clearance"
                )
            if engage and not _pose_check(poses[-1], geometry, open_width, engage=True)["valid"]:
                raise PlanningError("planned descent cannot engage the required object height")
            if (
                vertical
                and not _pose_check(before.arm(arm).tcp_pose, geometry, open_width, engage=False)[
                    "valid"
                ]
            ):
                raise PlanningError("vertical path must start centred and aligned")

        # Check the approach before opening; every actual move is re-planned
        # and its final/vertical geometry checked again before submission.
        stage = "approach"
        other = next((a for a in obs.arms if a != arm), None)
        plan = toolkit.planner.plan_move(
            arm,
            obs.arm(arm).q_model,
            geometry["approach_xyz"],
            pitch=90,
            yaw=yaw,
            roll=0,
            path="auto",
            half_turn=True,
            other_arm=other,
            other_q=obs.arm(other).q_model if other else None,
        )
        validate_plan(plan, obs, vertical=False, engage=False)
        stage = "open"
        opened = grip(
            toolkit, arm=arm, gripper=-1, opening=cal.gripper.width_to_opening(open_width)
        )
        result["stages"].append({"stage": stage, "result": opened})
        obs = toolkit.refresh_observation(tool="grasp_cube_open")
        actual_width = obs.arm(arm).gripper_width_m
        if (
            not opened["settled"]
            or actual_width is None
            or not math.isfinite(actual_width)
            or abs(actual_width - open_width) > 0.001
        ):
            raise ValueError("gripper did not reach the required approach width")

        for stage, target in (
            ("approach", geometry["approach_xyz"]),
            ("descent", geometry["target_xyz"]),
        ):
            vertical = stage == "descent"
            moved = move_pose(
                toolkit,
                target,
                arm=arm,
                path="line" if vertical else "auto",
                pitch=None if vertical else 90,
                yaw=None if vertical else yaw,
                roll=None if vertical else 0,
                _half_turn=not vertical,
                _validate_plan=lambda p, o, v=vertical: validate_plan(p, o, vertical=v, engage=v),
            )
            result["stages"].append({"stage": stage, "result": moved})
            if not moved["reached"] or moved["blocked"] or moved["slip"]:
                raise ValueError(f"{stage} did not arrive safely")
            obs = toolkit.refresh_observation(tool="grasp_cube_alignment")
            check = _pose_check(obs.arm(arm).tcp_pose, geometry, open_width, engage=vertical)
            result["stages"].append({"stage": stage + "_geometry", **check})
            if not check["valid"]:
                raise ValueError("measured pose does not satisfy centre/parallelism/engagement")
        result["overlap_fraction"] = check["overlap_fraction"]
        result["tcp_to_bottom"] = (
            float(np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)[2, 3]) - support_top
        )
        stage = "close"
        squeeze_started = True
        closed = grip(toolkit, arm=arm, gripper=1)
        result["stages"].append({"stage": stage, "result": closed})
        if not closed["holding"]:
            raise ValueError("close did not detect an object")

        stage = "settle"
        window: deque = deque()
        started = time.monotonic()
        baseline = None
        force_min = 0.95 * cal.gripper.force_n  # nominal SDK hold, not measured contact force
        while time.monotonic() - started <= stable_timeout_s:
            obs, sample = _hold_sample(toolkit, arm)
            elapsed = time.monotonic() - started
            sample["elapsed_s"] = elapsed
            result["stages"].append({"stage": stage, **sample})
            width = sample["width_m"]
            if (
                not sample["valid"]
                or abs(width - geometry["side_m"]) > WIDTH_TOL_M
                or width <= cal.gripper.closed_m + cal.gripper.min_holding_m
            ):
                raise ValueError("hold feedback is invalid or width does not match the cube")
            window.append(sample)
            while len(window) > 3 and window[1]["elapsed_s"] <= elapsed - STABLE_WINDOW_S:
                window.popleft()
            if (
                elapsed >= MIN_WAIT_S
                and len(window) >= 3
                and elapsed - window[0]["elapsed_s"] >= STABLE_WINDOW_S
                and max(s["width_m"] for s in window) - min(s["width_m"] for s in window)
                <= STABLE_WIDTH_M
                and all(s["nominal_force_n"] >= force_min for s in window)
                and max(s["nominal_force_n"] for s in window)
                - min(s["nominal_force_n"] for s in window)
                <= max(0.5, 0.05 * cal.gripper.force_n)
            ):
                baseline = sample
                break
            time.sleep(SAMPLE_S)
        if baseline is None:
            raise ValueError("hold did not remain stable for the required window")
        check = _pose_check(obs.arm(arm).tcp_pose, geometry, baseline["width_m"], engage=True)
        if not check["valid"]:
            raise ValueError("measured engagement changed before the lift")
        result["overlap_fraction"] = check["overlap_fraction"]
        result["tcp_to_bottom"] = (
            float(np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)[2, 3]) - support_top
        )
        stage = "baseline"
        toolkit.held_openings[arm] = baseline["opening"]
        if hasattr(toolkit, "sync_target_holds"):
            toolkit.sync_target_holds(obs)
        result["baseline_opening"] = baseline["opening"]
        result["baseline_width_m"] = baseline["width_m"]
        result["grasp_xyz"] = np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)[:3, 3].tolist()
        result["stages"].append({"stage": stage, **baseline})
        stage = "lift"
        lift_started = True
        target = np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)[:3, 3].copy()
        target[2] += geometry["lift_m"]
        lifted = move_pose(
            toolkit,
            target,
            arm=arm,
            path="line",
            _half_turn=False,
            _validate_plan=lambda p, o: validate_plan(p, o, vertical=True, engage=False),
        )
        result["stages"].append({"stage": stage, "result": lifted})
        result["lift_reached"] = lifted["reached"]
        result["slip"] = lifted["slip"]
        obs, held = _hold_sample(toolkit, arm)
        result["stages"].append({"stage": "lift_hold", **held})
        pose = np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)
        actual_lift = float(pose[2, 3] - (target[2] - geometry["lift_m"]))
        width_ok = (
            held["valid"]
            and abs(held["width_m"] - geometry["side_m"]) <= WIDTH_TOL_M
            and abs(held["width_m"] - baseline["width_m"]) <= 0.001
            and held["nominal_force_n"] >= force_min
        )
        if (
            not lifted["reached"]
            or lifted["blocked"]
            or lifted["slip"]
            or not width_ok
            or abs(actual_lift - geometry["lift_m"]) > 0.002
            or not _pose_check(pose, geometry, held["width_m"] or 0, engage=False)["valid"]
        ):
            raise ValueError("test lift did not preserve the measured hold and vertical path")
        result["grasped"] = True
        result["holding"] = True
    except (ValueError, PlanningError, RuntimeError, RpcError) as exc:
        result["failure_stage"] = stage
        result["note"] = str(exc)
        if squeeze_started and not lift_started:
            obs = toolkit.current_observation(refresh=True)
            if (
                not toolkit._closed
                and not obs.fault
                and not obs.arm(arm).fault
                and obs.arm(arm).enabled
                and obs.arm(arm).gripper_enabled
            ):
                try:
                    released = grip(toolkit, arm=arm, gripper=-1)
                    result["stages"].append({"stage": "release_on_support", "result": released})
                except (RuntimeError, RpcError) as release_error:
                    result["note"] += f"; release on support failed: {release_error}"
    finally:
        result["steps_used"] = toolkit.device_steps - start_steps
        obs = toolkit.current_observation(refresh=False)
        result["final_xyz"] = np.asarray(obs.arm(arm).tcp_pose).reshape(4, 4)[:3, 3].tolist()
        result["holding"] = arm in toolkit.held_openings
    return result
