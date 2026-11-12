"""Episode-local target identity and measured-pose search bounds.

Predictions gate observations; they never replace missing measured geometry.
Failed observations cannot move an anchor or refresh a grasp offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def target_key(prompt: str) -> str:
    return " ".join(prompt.lower().split()).removeprefix("the ")


@dataclass
class TargetExpectation:
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    held: bool = False
    tolerance_m: float = 0.008

    @property
    def extent(self) -> np.ndarray:
        return np.abs(self.rotation) @ self.size

    @property
    def top_z(self) -> float:
        return float(self.center[2] + self.extent[2] / 2)

    @property
    def upright(self) -> bool:
        return abs(float(self.rotation[2, 2])) >= np.cos(np.deg2rad(5))

    def contains(self, points: np.ndarray) -> np.ndarray:
        local = (points - self.center) @ self.rotation
        return np.all(np.abs(local) <= self.size / 2 + self.tolerance_m, axis=1)


@dataclass
class _Track:
    expectation: TargetExpectation
    last_capture_ns: int
    confirmations: int = 1
    reference_camera: str = "cam_head"
    arm: str | None = None
    center_in_tcp: np.ndarray | None = None
    rotation_in_tcp: np.ndarray | None = None
    baseline_width_m: float | None = None
    hold_valid: bool = True


class TargetTracker:
    """Lock geometry after two independent captures of the same text target."""

    def __init__(self) -> None:
        self.tracks: dict[str, _Track] = {}
        self.active_holds: set[str] = set()
        self.possibly_held: dict[str, set[str]] = {}

    def expectation(self, prompt: str) -> TargetExpectation | None:
        track = self.tracks.get(target_key(prompt))
        return track.expectation if track is not None else None

    def failure(self, prompt: str) -> str | None:
        track = self.tracks.get(target_key(prompt))
        if any(target_key(prompt) in keys for keys in self.possibly_held.values()):
            return "grasped target identity was not confirmed before lifting"
        if track is not None and track.arm and not track.hold_valid:
            return "held-object offset cannot be confirmed after hold feedback changed"
        if track is None and self.active_holds:
            return "no pregrasp target reference; cannot identify a new object while carrying"
        return None

    def validate(self, prompt: str, result: dict[str, Any]) -> str | None:
        expected = self.expectation(prompt)
        if expected is None:
            return None
        center = np.asarray(result["center_xyz"], dtype=float)
        if np.linalg.norm(center - expected.center) > expected.tolerance_m:
            return "target centre jumped outside its recorded/predicted geometry"
        if abs(float(result["top_z"]) - expected.top_z) > 0.004:
            return "target top changed beyond the measured reference"
        size = np.array([result["length_m"], result["width_m"], result["height_m"]])
        if np.any(np.abs(size - expected.size) > np.maximum(0.004, expected.size * 0.30)):
            return "target dimensions contradict the measured reference"
        return None

    def accept(self, prompt: str, result: dict[str, Any], capture_ns: int) -> None:
        key = target_key(prompt)
        quality = result["measurement_quality"]
        if not all(
            quality.get(k, False)
            for k in ("height_reliable", "center_reliable", "footprint_reliable")
        ):
            quality.update(temporal_confirmed=False, confirmations=0)
            result["geometry_confirmed"] = False
            return
        track = self.tracks.get(key)
        if track is None:
            angle = np.deg2rad(float(result["long_axis_deg"]))
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            track = _Track(
                TargetExpectation(
                    np.asarray(result["center_xyz"], dtype=float),
                    np.array([result["length_m"], result["width_m"], result["height_m"]]),
                    rotation,
                ),
                capture_ns,
                reference_camera=result.get("camera", "cam_head"),
            )
            self.tracks[key] = track
        elif capture_ns > track.last_capture_ns:
            track.confirmations = min(2, track.confirmations + 1)
            track.last_capture_ns = capture_ns
        # The original anchor is immutable; repeated measurements cannot drift it.
        confirmed = track.confirmations >= 2
        quality.update(temporal_confirmed=confirmed, confirmations=track.confirmations)
        result["geometry_confirmed"] = confirmed
        result["tracking"] = {
            "mode": "held" if track.arm else "supported",
            "reference_center_xyz": track.expectation.center.tolist(),
            "prediction_is_measurement": False,
        }

    def reject(
        self, prompt: str, capture_ns: int, *, camera: str | None = None, force: bool = False
    ) -> None:
        track = self.tracks.get(target_key(prompt))
        if (
            track is not None
            and camera is not None
            and camera != track.reference_camera
            and not force
        ):
            return  # Missing evidence in another camera does not contradict measured depth.
        if track is not None and capture_ns > track.last_capture_ns:
            track.confirmations = 0
            track.last_capture_ns = capture_ns

    def sync_holds(
        self, observation: Any, held_openings: dict[str, float], *, allow_bind: bool = True
    ) -> None:
        """Associate one nearby confirmed object at settled close, before lifting."""
        active = set(held_openings)
        for arm, keys in list(self.possibly_held.items()):
            if arm not in active:
                for key in keys:
                    self.tracks.pop(key, None)
                del self.possibly_held[arm]
        for key, track in list(self.tracks.items()):
            if track.arm and track.arm not in active:
                # Release invalidates the original supported location.
                del self.tracks[key]
        newly_held = active - self.active_holds
        self.active_holds = active
        for arm in newly_held:
            measured = observation.arm(arm)
            pose = np.asarray(measured.tcp_pose).reshape(4, 4)
            width = measured.gripper_width_m
            candidates = []
            nearby = set()
            for key, track in self.tracks.items():
                expected = track.expectation
                if track.arm or width is None:
                    continue
                if (
                    np.linalg.norm(pose[:2, 3] - expected.center[:2]) <= 0.015
                    and expected.center[2] - expected.size[2] / 2 - 0.005
                    <= pose[2, 3]
                    <= expected.top_z + 0.005
                    and abs(float(width) - expected.size[1]) <= 0.008
                ):
                    nearby.add(key)
                    if track.confirmations >= 2:
                        candidates.append(track)
            if len(candidates) != 1 or not allow_bind:
                self.possibly_held[arm] = nearby
                continue  # No guessed identity on an ambiguous or unmeasured grasp.
            track = candidates[0]
            track.arm = arm
            track.center_in_tcp = pose[:3, :3].T @ (track.expectation.center - pose[:3, 3])
            track.rotation_in_tcp = pose[:3, :3].T @ track.expectation.rotation
            track.baseline_width_m = float(width)
            track.expectation.held = True
            track.confirmations = 0
        for track in self.tracks.values():
            if track.arm is None:
                continue
            measured = observation.arm(track.arm)
            pose = np.asarray(measured.tcp_pose).reshape(4, 4)
            track.hold_valid = track.hold_valid and bool(
                measured.gripper_fresh
                and not measured.fault
                and measured.gripper_width_m is not None
                and abs(float(measured.gripper_width_m) - track.baseline_width_m) <= 0.001
            )
            track.expectation.center = pose[:3, 3] + pose[:3, :3] @ track.center_in_tcp
            track.expectation.rotation = pose[:3, :3] @ track.rotation_in_tcp
