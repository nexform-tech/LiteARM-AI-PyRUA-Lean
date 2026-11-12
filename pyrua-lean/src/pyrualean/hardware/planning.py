# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Joint, straight-line TCP and home planning for the real LiteArm.

Planning is done on the runner side for fast feedback and is repeated by the
device service before anything moves.  The service is the authority: it checks
the actual start state, the calibrated limits, the other arm's measured pose,
the table and the configured obstacles again.

The first hardware version executes each waypoint with a blocking
``movej_sync`` segment.  The firmware owns the S-curve; the SDK exposes only a
0..1 speed fraction.  This module therefore limits segment length, estimates
duration from a configured joint-speed ceiling and refuses a segment whose
estimate exceeds the configured maximum.  It does not claim to enforce an
acceleration bound that the SDK cannot command; that remains a field
verification item.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import Calibration, ExecutionConfig, SafetyConfig
from .kinematics import (
    ARM_NAMES,
    MEASURED_LIMIT_TOL_RAD,
    CollisionModel,
    LitearmKinematics,
    angles,
    orientation,
    rotation_error,
)

#: A target missed by at most this much is reported as a near miss and reached
#: at the closest feasible pose (the caller must see the note).
NEAR_M = 0.02
NEAR_DEG = 8.0


class PlanningError(ValueError):
    """No collision-free, within-limit trajectory exists for this request."""

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


@dataclass(frozen=True)
class PlannedTrajectory:
    arm: str
    waypoints: tuple[tuple[float, ...], ...]
    path: str
    target_xyz: tuple[float, float, float]
    target_angles: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    target_pose: tuple[float, ...] | None = None
    note: str | None = None
    estimated_duration_s: float | None = None
    path_detail: dict = field(default_factory=dict)

    def to_request(self) -> dict:
        return {
            "arm": self.arm,
            "waypoints": [list(w) for w in self.waypoints],
            "path": self.path,
            "target_xyz": list(self.target_xyz),
            "target_angles": list(self.target_angles),
            "start_xyz": list(self.start_xyz),
        }


def _arm(name: str) -> str:
    if name not in ARM_NAMES:
        raise PlanningError(f"arm must be one of {ARM_NAMES}, got {name!r}")
    return name


def _target_pose(
    current_rotation: np.ndarray, xyz: Sequence[float], angles_deg: Sequence[float]
) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = orientation(*angles_deg)
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    _ = current_rotation
    return pose


def _followable(
    q_start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    step_rad: float,
    *,
    joint_limits_rad: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    """Subdivide from measured start, keeping every command inside exact limits.

    A readback just outside a limit remains the measured start. Interpolation
    must not copy that noise into a command, even when IK keeps that joint at
    its limit. Only interpolated samples are projected; invalid IK/home targets
    and starts beyond the observation tolerance are refused.
    """
    out: list[tuple[float, ...]] = []
    prev = np.asarray(q_start, dtype=float)
    limits = np.asarray(joint_limits_rad, dtype=float)
    if prev.shape != (7,) or not np.all(np.isfinite(prev)):
        raise PlanningError("q_start is not a finite 7-vector")
    if np.any(prev < limits[:, 0] - MEASURED_LIMIT_TOL_RAD) or np.any(
        prev > limits[:, 1] + MEASURED_LIMIT_TOL_RAD
    ):
        raise PlanningError("q_start is outside the calibrated joint limits")
    for way in waypoints:
        target = np.asarray(way, dtype=float)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise PlanningError("trajectory target is not a finite 7-vector")
        if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
            raise PlanningError("trajectory target is outside the calibrated joint limits")
        count = max(1, int(math.ceil(float(np.max(np.abs(target - prev))) / step_rad)))
        for i in range(1, count + 1):
            sample = np.clip(prev + (target - prev) * (i / count), limits[:, 0], limits[:, 1])
            out.append(tuple(float(v) for v in sample))
        prev = target
    return out


def _segment_duration(q0: Sequence[float], q1: Sequence[float], joint_speed_rad_s: float) -> float:
    delta = float(np.max(np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))))
    return delta / max(1e-9, joint_speed_rad_s)


class Planner:
    def __init__(
        self,
        calibration: Calibration,
        execution: ExecutionConfig,
        safety: SafetyConfig,
        kinematics: LitearmKinematics | None = None,
    ) -> None:
        self.calibration = calibration
        self.execution = execution
        self.safety = safety
        self.kinematics = kinematics or LitearmKinematics(calibration)
        self.collision = CollisionModel(calibration, self.kinematics)

    # -- validation ---------------------------------------------------------

    def validate_waypoints(
        self,
        arm: str,
        q_start: Sequence[float],
        waypoints: Sequence[Sequence[float]],
        *,
        other_arm: str | None = None,
        other_q: Sequence[float] | None = None,
        max_joint_step_rad: float | None = None,
    ) -> list[str]:
        arm = _arm(arm)
        cal = self.calibration.arm(arm)
        errors: list[str] = []
        if len(q_start) != 7:
            errors.append("q_start must have 7 joints")
        if not waypoints:
            errors.append("trajectory has no waypoints")
        prev = np.asarray(q_start, dtype=float)
        if prev.shape != (7,) or not np.all(np.isfinite(prev)):
            errors.append("q_start is not a finite 7-vector")
            return errors
        limits = np.asarray(cal.joint_limits_rad)
        if np.any(prev < limits[:, 0] - MEASURED_LIMIT_TOL_RAD) or np.any(
            prev > limits[:, 1] + MEASURED_LIMIT_TOL_RAD
        ):
            errors.append("q_start is outside the calibrated joint limits")
            return errors
        floor_bad = self.collision.z_floor_violations(arm, prev)
        if floor_bad:
            errors.append(f"q_start violates the hard table floor: {'; '.join(floor_bad)}")
        max_step = (
            self.execution.max_joint_step_rad
            if max_joint_step_rad is None
            else float(max_joint_step_rad)
        )
        for index, way in enumerate(waypoints):
            q = np.asarray(way, dtype=float)
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                errors.append(f"waypoint {index} is not a finite 7-vector")
                return errors
            if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
                errors.append(f"waypoint {index} is outside the calibrated joint limits")
                return errors
            if float(np.max(np.abs(q - prev))) > max_step + 1e-6:
                errors.append(
                    f"waypoint {index} moves a joint by "
                    f"{float(np.max(np.abs(q - prev))):.4f} rad, over {max_step:.4f}"
                )
            floor_bad = self.collision.z_floor_violations(arm, q)
            if floor_bad:
                errors.append(
                    f"waypoint {index} violates the hard table floor: {'; '.join(floor_bad)}"
                )
            if not self._collision_free(arm, q, other_arm=other_arm, other_q=other_q):
                errors.append(
                    f"waypoint {index} collides with the table, an obstacle or the other arm"
                )
            # The hard floor is cheap (FK only) and the joint-space sweep is not
            # straight in Cartesian space, so sample it densely.
            for t in np.linspace(0.05, 0.95, 19):
                mid = prev + (q - prev) * float(t)
                floor_bad = self.collision.z_floor_violations(arm, mid)
                if floor_bad:
                    errors.append(
                        f"segment {index} violates the hard table floor: {'; '.join(floor_bad)}"
                    )
                    break
            # Sample the joint-space segment too; a primitive collision model can
            # miss a light graze between two waypoints.
            for t in (0.25, 0.5, 0.75):
                mid = prev + (q - prev) * t
                if not self._collision_free(arm, mid, other_arm=other_arm, other_q=other_q):
                    errors.append(f"segment {index} collides between waypoints")
                    break
            prev = q
            if errors:
                break
        return errors

    def _collision_free(
        self,
        arm: str,
        q: Sequence[float],
        *,
        other_arm: str | None,
        other_q: Sequence[float] | None,
    ) -> bool:
        try:
            bad = self.collision.all_violations(arm, q, other_arm=other_arm, other_q=other_q)
        except ValueError:
            return False
        return not bad

    def estimate_duration(
        self, q_start: Sequence[float], waypoints: Sequence[Sequence[float]]
    ) -> float:
        speed = max(
            1e-6,
            self.execution.max_joint_speed_rad_s * max(self.execution.max_speed, 1e-6),
        )
        total = 0.0
        prev = q_start
        for way in waypoints:
            total += _segment_duration(prev, way, speed)
            prev = way
        return total

    def validate_tcp_speed(
        self, arm: str, q_start: Sequence[float], waypoints: Sequence[Sequence[float]]
    ) -> None:
        """Reject a plan whose *upper-bound* average TCP speed exceeds the limit.

        The firmware owns the S-curve; this uses the configured joint-speed
        ceiling as a lower bound on segment duration, so the check is
        conservative and is not a claim that acceleration is enforced.
        """

        speed = max(
            1e-6,
            self.execution.max_joint_speed_rad_s * max(self.execution.max_speed, 1e-6),
        )
        previous = self.kinematics.fk(arm, q_start)[:3, 3]
        prev_q = np.asarray(q_start, dtype=float)
        for index, way in enumerate(waypoints):
            current = self.kinematics.fk(arm, way)[:3, 3]
            distance = float(np.linalg.norm(current - previous))
            duration = _segment_duration(prev_q, way, speed)
            if duration > 1e-9 and distance / duration > self.execution.max_tcp_speed_m_s:
                raise PlanningError(
                    f"segment {index} would average {distance / duration:.2f} m/s at the "
                    f"configured joint ceiling, over the {self.execution.max_tcp_speed_m_s:.2f} "
                    "m/s TCP limit; split the motion or lower execution.max_speed",
                    detail={"segment": index, "tcp_speed_m_s": distance / duration},
                )
            previous, prev_q = current, np.asarray(way, dtype=float)

    def _validate_duration(
        self, q_start: Sequence[float], waypoints: Sequence[Sequence[float]]
    ) -> None:
        speed = max(
            1e-6,
            self.execution.max_joint_speed_rad_s * max(self.execution.max_speed, 1e-6),
        )
        prev = q_start
        for index, way in enumerate(waypoints):
            duration = _segment_duration(prev, way, speed)
            if duration > self.execution.max_segment_duration_s + 1e-9:
                raise PlanningError(
                    f"segment {index} would take {duration:.2f} s at the configured "
                    f"{speed:.3f} rad/s joint ceiling, over "
                    f"{self.execution.max_segment_duration_s:.2f} s; split the motion "
                    "or lower the target distance",
                    detail={"segment": index, "duration_s": duration},
                )
            prev = way
        total = self.estimate_duration(q_start, waypoints)
        if total > self.execution.command_timeout_s:
            raise PlanningError(
                f"estimated trajectory duration {total:.1f} s exceeds the command "
                f"timeout {self.execution.command_timeout_s:.1f} s"
            )

    # -- move planning ------------------------------------------------------

    def plan_move(
        self,
        arm: str,
        q_start: Sequence[float],
        xyz: Sequence[float],
        *,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
        half_turn: bool = True,
        other_arm: str | None = None,
        other_q: Sequence[float] | None = None,
        holding: bool = False,
        step_rad: float | None = None,
    ) -> PlannedTrajectory:
        arm = _arm(arm)
        if path not in ("auto", "line", "joint"):
            raise PlanningError("path must be 'auto', 'line' or 'joint'")
        xyz = tuple(float(v) for v in xyz)
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise PlanningError("xyz must be three finite numbers")
        current = self.kinematics.fk(arm, q_start)
        # The current pitch/yaw/roll must be read in the same way the public API
        # reports them; ``angles`` returns (pitch, yaw, roll).
        cp, cy, cr = angles(current[:3, :3])
        want = (
            float(cp if pitch is None else pitch),
            float(cy if yaw is None else yaw),
            float(cr if roll is None else roll),
        )
        target = _target_pose(current[:3, :3], xyz, want)
        if pitch is None and yaw is None and roll is None:
            # Near vertical, the display angles canonicalize roll to zero.
            # Preserve the measured matrix, not that lossy display round trip.
            target[:3, :3] = current[:3, :3]
        flipped = target.copy()
        flipped[:3, 1:3] *= -1.0
        candidates = [target]
        if half_turn and not holding:
            candidates.append(flipped)
        options = []
        first_error: PlanningError | None = None
        for candidate in candidates:
            try:
                options.append(self._plan_one(arm, q_start, candidate, path, step_rad=step_rad))
            except PlanningError as exc:
                # Do not overwrite a reachable target's line/collision refusal
                # with an unreachable half-turn's error. For two IK misses,
                # report the closer one rather than the last attempted pose.
                if first_error is None:
                    first_error = exc
                elif "pos_err_m" in first_error.detail:
                    if "pos_err_m" not in exc.detail or (
                        exc.detail["pos_err_m"] + 0.1 * math.radians(exc.detail["rot_err_deg"])
                        < first_error.detail["pos_err_m"]
                        + 0.1 * math.radians(first_error.detail["rot_err_deg"])
                    ):
                        first_error = exc
        if not options:
            assert first_error is not None
            raise first_error
        # Prefer exact solutions; among them the shortest joint travel.
        options.sort(
            key=lambda item: (
                float(np.linalg.norm(item[3][:3, 3] - item[0][:3, 3])) > 1e-6
                or rotation_error(item[3][:3, :3], item[0][:3, :3]) > 1e-3,
                item[1],
            )
        )
        selected_target, travel, plan, goal, taken = options[0]
        _ = travel
        errors = self.validate_waypoints(
            arm,
            q_start,
            plan,
            other_arm=other_arm,
            other_q=other_q,
        )
        if errors:
            raise PlanningError("; ".join(errors), detail={"violations": errors})
        self._validate_duration(q_start, plan)
        self.validate_tcp_speed(arm, q_start, plan)
        note = None
        target_dist = float(np.linalg.norm(np.asarray(goal[:3, 3]) - np.asarray(xyz)))
        target_turn = rotation_error(goal[:3, :3], selected_target[:3, :3])
        if target_dist > 1e-6 or target_turn > 1e-3:
            note = (
                f"the exact pose is out of reach; went to the closest one, "
                f"{target_dist:.3f} m and {target_turn:.0f} deg from it"
            )
        if selected_target is flipped and half_turn and not holding:
            half = (
                "took the pose turned half a turn about the pointing direction "
                "(the empty fingers are interchangeable): yaw / roll read 180 off"
            )
            note = f"{note}; {half}" if note else half
        end_angles = angles(goal[:3, :3])
        return PlannedTrajectory(
            arm=arm,
            waypoints=tuple(plan),
            path=taken,
            target_xyz=tuple(float(v) for v in xyz),
            target_angles=tuple(float(v) for v in end_angles),
            start_xyz=tuple(float(v) for v in current[:3, 3]),
            target_pose=tuple(float(v) for v in goal.reshape(-1)),
            note=note,
            estimated_duration_s=self.estimate_duration(q_start, plan),
        )

    def _plan_one(
        self,
        arm: str,
        q_start: Sequence[float],
        target: np.ndarray,
        path: str,
        *,
        step_rad: float | None,
    ) -> tuple[np.ndarray, float, list[tuple[float, ...]], np.ndarray, str]:
        cal = self.calibration.arm(arm)
        target_xyz = target[:3, 3]
        if np.any(target_xyz < np.asarray(cal.base_pose)[:3, 3] - 1.2) or np.any(
            target_xyz > np.asarray(cal.base_pose)[:3, 3] + 1.2
        ):
            raise PlanningError("target is outside the arm's working envelope")
        res = self.kinematics.best_solution(arm, target, q_start, extra_seeds=[cal.safe_home_rad])
        near = res.pos_err_m <= NEAR_M and res.rot_err_deg <= NEAR_DEG
        if not res.ok and not near:
            raise PlanningError(
                f"out of reach: the closest pose the arm can take is "
                f"{res.pos_err_m:.3f} m and {res.rot_err_deg:.0f} deg away",
                detail={"pos_err_m": res.pos_err_m, "rot_err_deg": res.rot_err_deg},
            )
        if res.collides:
            violations = self.collision.self_violations(arm, res.q)
            detail = {"violations": violations}
            if f"{arm}_tool_body" in violations or f"{arm}_tcp_table_plane" in violations:
                detail["tool_table_constraint"] = self.collision.tool_table_constraint(
                    arm, target[:3, :3]
                )
            raise PlanningError(
                "every solution puts part of the arm into the table or an obstacle: "
                + "; ".join(violations),
                detail=detail,
            )
        goal = target
        if not res.ok:
            goal = self.kinematics.fk(arm, res.q)
        ways: list[tuple[float, ...]] = []
        taken = "joint"
        if path in ("line", "auto"):
            line, why = self.kinematics.line_path(
                arm,
                q_start,
                goal,
                max_joint_step_rad=max(self.execution.max_joint_step_rad, 0.05),
            )
            if why is None:
                ways = line
                taken = "line"
            elif path == "line":
                raise PlanningError(
                    why + " (the target itself is reachable: path='auto' swings)",
                    detail={"path": "line", "why": why},
                )
        if not ways:
            if path == "joint" or path == "auto":
                joint_clear = self._joint_path_clear(arm, q_start, res.q)
                if not joint_clear:
                    raise PlanningError(
                        "the joint-space swing to this pose passes through the table or an obstacle"
                    )
                ways = [tuple(float(v) for v in res.q)]
                taken = "joint"
        if not ways:
            raise PlanningError("no path to the target")
        step = self.execution.max_joint_step_rad if step_rad is None else float(step_rad)
        followable = _followable(
            q_start, ways, max(0.005, step), joint_limits_rad=cal.joint_limits_rad
        )
        # The last point must be the exact IK solution, not a rounding artefact.
        followable[-1] = tuple(float(v) for v in ways[-1])
        travel = float(np.sum(np.abs(np.asarray(res.q) - np.asarray(q_start, dtype=float))))
        return target, travel, followable, goal, taken

    def _joint_path_clear(self, arm: str, q0: Sequence[float], q1: Sequence[float]) -> bool:
        a = np.asarray(q0, dtype=float)
        b = np.asarray(q1, dtype=float)
        for t in np.linspace(0.0, 1.0, 21)[1:]:
            if not self._collision_free(arm, a + (b - a) * t, other_arm=None, other_q=None):
                return False
        return True

    # -- home planning ------------------------------------------------------

    def plan_home(
        self,
        arm: str,
        q_start: Sequence[float],
        *,
        other_arm: str | None = None,
        other_q: Sequence[float] | None = None,
    ) -> PlannedTrajectory:
        arm = _arm(arm)
        cal = self.calibration.arm(arm)
        current = self.kinematics.fk(arm, q_start)
        lift = float(self.safety.home_lift_m)
        errors: list[str] = []
        for attempt in (lift, lift * 2.0, lift * 1.5):
            if attempt <= 0.0:
                break
            up = current.copy()
            up[2, 3] += attempt
            try:
                line, why = self.kinematics.line_path(arm, q_start, up)
            except Exception as exc:  # noqa: BLE001 - converted below
                why = str(exc)
                line = []
            if why is not None or not line:
                errors.append(why or "no lift path")
                continue
            lifted_q = line[-1]
            joint_ways = _followable(
                lifted_q,
                [cal.safe_home_rad],
                self.execution.max_joint_step_rad,
                joint_limits_rad=cal.joint_limits_rad,
            )
            waypoints = list(line) + joint_ways
            if self._joint_path_clear(arm, q_start, lifted_q) and self._joint_path_clear(
                arm, lifted_q, cal.safe_home_rad
            ):
                violations = self.validate_waypoints(
                    arm,
                    q_start,
                    waypoints,
                    other_arm=other_arm,
                    other_q=other_q,
                )
                if not violations:
                    self._validate_duration(q_start, waypoints)
                    self.validate_tcp_speed(arm, q_start, waypoints)
                    home_pose = self.kinematics.fk(arm, cal.safe_home_rad)
                    return PlannedTrajectory(
                        arm=arm,
                        waypoints=tuple(waypoints),
                        path="lift+joint",
                        target_xyz=tuple(float(v) for v in home_pose[:3, 3]),
                        target_angles=tuple(float(v) for v in angles(home_pose[:3, :3])),
                        start_xyz=tuple(float(v) for v in current[:3, 3]),
                        note=f"lifted {attempt:.3f} m before the joint-space return",
                        estimated_duration_s=self.estimate_duration(q_start, waypoints),
                        path_detail={"lift_m": attempt},
                    )
                errors.extend(violations)
        raise PlanningError(
            "no safe lift-and-return path to the configured home pose: "
            + ("; ".join(errors[-4:]) if errors else "the joint swing is not clear"),
            detail={"errors": errors},
        )


__all__ = [
    "NEAR_DEG",
    "NEAR_M",
    "PlannedTrajectory",
    "Planner",
    "PlanningError",
]
