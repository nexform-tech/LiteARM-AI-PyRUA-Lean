# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Perception over measured camera images: SAM3 plus calibrated geometry.

Two measurement paths are implemented:

* **RGB-D**: a robust top-plane fit rejects foreground depth. The measured
  top-plane footprint supplies x/y; RGB projections remain separate estimates.
* **RGB only**: masks of the same text prompt in at least two calibrated views
  are intersected as a visual hull on a voxel grid.  One view cannot determine
  a height; the result says ``measurement_failed`` instead of inventing one.
  A separate ``table_projection`` exposes approximate x/y and heading without
  depth, even when height measurement fails. Explicit table queries report the
  configured support plane, labelled as calibration rather than measurement.

The packet is not a simulator: no RoboDojo task layout, object pose or fixed
table height is read.  Table bounds and camera calibration come from the
station calibration.  Occlusion, missing depth and insufficient views produce
an explicit failure, not a value filled from a constant.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .camera_geometry import (
    CAMERA_FRAME,
    depth_points,
    project_points,
    ray_plane,
    station_transform,
)
from .config import Calibration, Sam3Config
from .kinematics import wrap_deg
from .target_tracking import TargetTracker


class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Segment:
    found: bool
    score: float | None = None
    mask: np.ndarray | None = None
    box: list[float] | None = None
    reason: str | None = None


@dataclass
class FrameData:
    name: str
    rgb: np.ndarray
    depth: np.ndarray | None
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    captured_at_wall_ns: int

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.rgb.shape[0]), int(self.rgb.shape[1]))


@dataclass
class Measurement:
    center_xyz: tuple[float, float, float]
    top_z: float
    height_m: float
    length_m: float
    width_m: float
    long_axis_deg: float
    grasp_yaw: float
    source: str
    views: int
    points: int
    quality: dict[str, Any] = field(default_factory=dict)


def frame_from_observation(
    observation: Any,
    artifacts: Mapping[str, bytes],
    camera: str,
) -> FrameData:
    from PIL import Image

    meta = observation.camera_state.get(camera)
    if not isinstance(meta, dict):
        raise PerceptionError(f"camera {camera!r} has no observation metadata")
    color_name = meta.get("color_artifact")
    if not color_name:
        raise PerceptionError(f"camera {camera!r} has no colour artifact")
    try:
        raw = artifacts[color_name]
    except KeyError:
        raise PerceptionError(f"colour artifact {color_name!r} was not retained") from None
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    depth = None
    depth_name = meta.get("depth_artifact")
    if depth_name:
        try:
            payload = artifacts[depth_name]
        except KeyError:
            raise PerceptionError(f"depth artifact {depth_name!r} was not retained") from None
        depth = np.load(io.BytesIO(payload)).astype(np.float32)
        if depth.shape != rgb.shape[:2]:
            raise PerceptionError(
                f"camera {camera}: depth shape {depth.shape} does not match colour {rgb.shape[:2]}"
            )
    intrinsic = np.asarray(meta.get("intrinsic_matrix"), dtype=float)
    if intrinsic.shape != (3, 3):
        raise PerceptionError(f"camera {camera}: invalid intrinsic matrix")
    c2w = np.asarray(meta.get("camera_to_world"), dtype=float)
    if c2w.shape != (4, 4):
        raise PerceptionError(f"camera {camera}: invalid camera-to-world transform")
    return FrameData(
        name=camera,
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        camera_to_world=station_transform(c2w, meta.get("camera_frame", CAMERA_FRAME)),
        captured_at_wall_ns=int(meta.get("captured_at_wall_ns", 0)),
    )


def _project(frame: FrameData, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to (row, col, depth)."""

    return project_points(frame.intrinsic, frame.camera_to_world, points)


def _depth_points(frame: FrameData, mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    if frame.depth is None:
        return np.zeros((0, 3)), 0.0
    depth = frame.depth
    valid = np.isfinite(depth) & (depth > 0.0)
    if mask is not None:
        valid &= mask.astype(bool)
    rows, cols = np.nonzero(valid)
    total = int(rows.size)
    if total == 0:
        return np.zeros((0, 3)), 0.0
    d = depth[rows, cols].astype(float)
    world = depth_points(frame.intrinsic, frame.camera_to_world, rows, cols, d)
    denominator = (
        float(np.sum(mask))
        if mask is not None and np.any(mask)
        else float(np.isfinite(depth).sum())
    )
    ratio = total / max(1.0, denominator)
    return world, ratio


def _ray_plane(frame: FrameData, row: int, col: int, z: float) -> np.ndarray | None:
    return ray_plane(frame.intrinsic, frame.camera_to_world, row, col, z)


class Sam3Segmenter:
    """Lazy wrapper over the existing local SAM3 client (agent environment)."""

    def __init__(self, config: Sam3Config) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from ..hosts import sam3_local  # noqa: PLC0415 - agent-side dependency

            if not self.config.checkpoint:
                raise PerceptionError(
                    "SAM3 checkpoint is not configured; set perception.sam3.checkpoint "
                    "to an existing file in the agent environment"
                )
            self._model = sam3_local.shared(
                checkpoint=self.config.checkpoint, cuda_device=self.config.cuda_device
            )
        return self._model

    def segment(
        self,
        image: np.ndarray,
        *,
        text_prompt: str | None = None,
        point: Sequence[int] | None = None,
        min_score: float = 0.2,
    ) -> Segment:
        model = self._load()
        try:
            result = model.segment(
                image,
                text_prompt=text_prompt,
                point=list(point) if point else None,
                min_score=min_score,
            )
        except Exception as exc:  # noqa: BLE001 - service failure is reported as a result
            raise PerceptionError(f"SAM3 segmentation failed: {exc}") from exc
        return Segment(
            found=bool(result.found),
            score=float(result.score) if result.score is not None else None,
            mask=np.asarray(result.mask, dtype=bool) if result.mask is not None else None,
            box=list(result.box) if result.box is not None else None,
            reason=result.reason,
        )


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    rows, cols = np.nonzero(mask)
    return {
        "area_px": int(rows.size),
        "centroid_rc": [int(rows.mean()), int(cols.mean())] if rows.size else None,
    }


def _interior_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Avoid aligned-depth spillover at RGB silhouette boundaries."""
    mask = np.asarray(mask, dtype=bool)
    radius = max(1, min(4, min(mask.shape) // 120))
    while radius:
        padded = np.pad(mask, radius, constant_values=False)
        interior = np.ones_like(mask)
        h, w = mask.shape
        for row in range(2 * radius + 1):
            for col in range(2 * radius + 1):
                interior &= padded[row : row + h, col : col + w]
        if interior.sum() >= max(12, 0.2 * mask.sum()):
            return interior, radius
        radius -= 1
    return mask, 0


def _robust_depth_points(frame: FrameData, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    interior, radius = _interior_mask(mask)
    points, ratio = _depth_points(frame, interior)
    quality: dict[str, Any] = {
        "depth_valid_ratio": round(ratio, 3),
        "depth_interior_margin_px": radius,
    }
    if len(points):
        median = float(np.median(points[:, 2]))
        mad = float(np.median(np.abs(points[:, 2] - median)))
        # A small, disconnected high tail often belongs to the occluding arm.
        # Keep broad height distributions; report all rejected samples.
        keep = np.abs(points[:, 2] - median) <= max(0.025, 6.0 * mad)
        quality["depth_rejected_ratio"] = round(float(1.0 - keep.mean()), 3)
        points = points[keep]
    return points, quality


def _table_prompt(prompt: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:the\s+)?(?:white\s+)?(?:tabletop|table(?:\s+(?:top|surface))?)|(?:白色)?桌面",
            prompt.strip().lower(),
        )
    )


def _footprint(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Long-axis heading (deg), length, width and centre of a 2-D point set."""

    if len(xy) < 3:
        centre = xy.mean(axis=0) if len(xy) else np.zeros(2)
        return 0.0, 0.0, 0.0, centre
    centre = np.median(xy, axis=0)
    cov = np.cov((xy - centre).T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axis = vectors[:, order[0]]
    heading = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    u = np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
    v = np.array([-u[1], u[0]])
    lo_u, hi_u = np.percentile(xy @ u, [1.0, 99.0])
    lo_v, hi_v = np.percentile(xy @ v, [1.0, 99.0])
    length = float(hi_u - lo_u)
    width = float(hi_v - lo_v)
    if width > length:
        length, width = width, length
        heading += 90.0
    mid = u * (lo_u + hi_u) / 2 + v * (lo_v + hi_v) / 2
    heading = wrap_deg(heading)
    if heading < -90.0:
        heading += 180.0
    if heading > 90.0:
        heading -= 180.0
    return heading, length, width, mid


def _bounded_top_plane(points: np.ndarray, max_slope: float) -> np.ndarray:
    """Least squares within the same slope bound used by RANSAC hypotheses."""
    xy_mean = points[:, :2].mean(axis=0)
    z_mean = float(points[:, 2].mean())
    xy = points[:, :2] - xy_mean
    z = points[:, 2] - z_mean
    gram, rhs = xy.T @ xy, xy.T @ z
    slope = np.linalg.lstsq(xy, z, rcond=None)[0]
    if np.linalg.norm(slope) > max_slope:
        lo, hi = 0.0, max(float(np.trace(gram)), 1e-12)
        while np.linalg.norm(np.linalg.solve(gram + hi * np.eye(2), rhs)) > max_slope:
            hi *= 2
        for _ in range(50):
            penalty = (lo + hi) / 2
            candidate = np.linalg.solve(gram + penalty * np.eye(2), rhs)
            if np.linalg.norm(candidate) > max_slope:
                lo = penalty
            else:
                hi = penalty
        slope = np.linalg.solve(gram + hi * np.eye(2), rhs)
    return np.r_[slope, z_mean - xy_mean @ slope]


def _box_mask_overlap(frame: FrameData, corners: np.ndarray, mask: np.ndarray) -> float:
    """Check a fitted box against the complete silhouette, not just occupied rays."""
    rows, cols, depth = _project(frame, corners)
    if np.any(depth <= 0) or not np.all(np.isfinite(rows)):
        return 0.0
    points = sorted(set(zip(cols.tolist(), rows.tolist(), strict=True)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for ordered, hull in ((points, lower), (points[::-1], upper)):
        for point in ordered:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    h, w = frame.shape
    r0, r1 = max(0, int(np.floor(rows.min()))), min(h, int(np.ceil(rows.max())) + 1)
    c0, c1 = max(0, int(np.floor(cols.min()))), min(w, int(np.ceil(cols.max())) + 1)
    if r1 <= r0 or c1 <= c0:
        return 0.0
    rr, cc = np.mgrid[r0:r1, c0:c1]
    inside = np.ones(rr.shape, dtype=bool)
    for a, b in zip(hull, hull[1:] + hull[:1], strict=True):
        inside &= (b[0] - a[0]) * (rr - a[1]) - (b[1] - a[1]) * (cc - a[0]) >= -0.5
    overlap = int(np.count_nonzero(inside & mask[r0:r1, c0:c1]))
    union = int(inside.sum()) + int(mask.sum()) - overlap
    return overlap / max(1, union)


class Perception:
    def __init__(
        self,
        calibration: Calibration,
        sam3: Sam3Config,
        *,
        segmenter: Any | None = None,
    ) -> None:
        self.calibration = calibration
        self.sam3 = sam3
        self.tracker = TargetTracker()
        if segmenter is not None:
            self._segmenter = segmenter
        elif sam3.mode == "http":
            # SAM3 runs on the Jetson Thor (or another host); the agent holds a
            # thin stdlib HTTP client and needs neither torch nor the sam3 wheel.
            from ..hosts.sam3_http import Sam3Http  # noqa: PLC0415

            if not sam3.endpoint:
                raise PerceptionError("perception.sam3.mode='http' needs perception.sam3.endpoint")
            self._segmenter = Sam3Http(endpoint=sam3.endpoint, timeout_s=sam3.timeout_s)
        else:
            self._segmenter = Sam3Segmenter(sam3)

    # -- segment -----------------------------------------------------------

    def _target_segment(self, frame, prompt, score_floor, expected=None):
        """Run SAM3 only inside a target's projected 3-D search box."""
        roi = None
        image = frame.rgb
        if expected is not None:
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            corners = (
                expected.center
                + (signs * (expected.size / 2 + expected.tolerance_m)) @ expected.rotation.T
            )
            rows, cols, depths = _project(frame, corners)
            if not np.all(np.isfinite(rows)) or np.any(depths <= 0):
                return Segment(False, reason="target search volume is behind the camera"), None
            h, w = frame.shape
            r0, r1 = max(0, int(np.floor(rows.min())) - 3), min(h, int(np.ceil(rows.max())) + 4)
            c0, c1 = max(0, int(np.floor(cols.min())) - 3), min(w, int(np.ceil(cols.max())) + 4)
            if r1 - r0 < 3 or c1 - c0 < 3:
                return Segment(False, reason="target search volume is outside the image"), None
            roi = [r0, c0, r1, c1]
            image = frame.rgb[r0:r1, c0:c1]
        result = self._segmenter.segment(image, text_prompt=prompt, min_score=score_floor)
        if not result.found or result.mask is None:
            return result, roi
        if result.score is None or not math.isfinite(result.score) or result.score < score_floor:
            return Segment(
                False, result.score, reason="mask score is below the configured floor"
            ), roi
        mask = np.asarray(result.mask, dtype=bool)
        if mask.shape != image.shape[:2] or mask.sum() < 12:
            return Segment(False, result.score, reason="invalid or insufficient target mask"), roi
        if roi is not None:
            if mask.mean() > 0.80 or np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
                return Segment(
                    False, result.score, reason="target mask is oversized or truncated"
                ), roi
            full = np.zeros(frame.shape, dtype=bool)
            full[r0:r1, c0:c1] = mask
            mask = full
        rows, cols = np.nonzero(mask)
        box = [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
        return Segment(True, result.score, mask, box), roi

    def _score_floor(self, min_score):
        requested = self.sam3.min_score if min_score is None else float(min_score)
        if not math.isfinite(requested) or not 0 <= requested <= 1:
            raise ValueError("min_score must be finite and in [0, 1]")
        return max(self.sam3.min_score, requested)

    def _unconfirmed(self, out, reason):
        if out.get("tracking_enabled"):
            self.tracker.reject(
                out["text_prompt"], out["captured_at_wall_ns"], camera=out.get("camera")
            )
        for key in (
            "center_xyz",
            "top_z",
            "height_m",
            "length_m",
            "width_m",
            "long_axis_deg",
            "grasp_yaw",
        ):
            out.pop(key, None)
        out.update(
            measurement_failed=True,
            geometry_confirmed=False,
            error_code="target_unconfirmed",
            note="target unconfirmed: " + reason,
            geometry_error="target unconfirmed: " + reason,
        )
        quality = out.setdefault("measurement_quality", {})
        quality.update(
            height_reliable=bool(out.get("partial_measurement")),
            center_reliable=False,
            footprint_reliable=False,
        )
        return out

    def segment(
        self,
        frames: Mapping[str, FrameData],
        text_prompt: str,
        camera: str,
        *,
        min_score: float | None = None,
        track: bool = False,
    ) -> dict[str, Any]:
        if not text_prompt.strip():
            raise ValueError("text_prompt must not be empty")
        frame = frames.get(camera)
        if frame is None:
            raise PerceptionError(f"camera {camera!r} is not in this snapshot")
        score_floor = self._score_floor(min_score)
        expected = self.tracker.expectation(text_prompt) if track else None
        reason = (
            self.tracker.failure(text_prompt) if track and not _table_prompt(text_prompt) else None
        )
        result, roi = (
            (Segment(False, reason=reason), None)
            if reason
            else self._target_segment(frame, text_prompt, score_floor, expected)
        )
        out: dict[str, Any] = {
            "camera": camera,
            "found": bool(result.found),
            "text_prompt": text_prompt,
            "score": result.score,
            "median_score": result.score,
            "box_px": result.box,
            "mask_shape": list(result.mask.shape) if result.mask is not None else None,
            "measurement_failed": not bool(result.found),
            "search_roi_rc": roi,
            "captured_at_wall_ns": frame.captured_at_wall_ns,
            "tracking_enabled": track,
        }
        if result.mask is not None:
            out.update(_mask_stats(result.mask))
        if not result.found:
            out["note"] = result.reason or "SAM3 found no mask above the score threshold"
            if track:
                out["error_code"] = "target_unconfirmed"
                self.tracker.reject(text_prompt, frame.captured_at_wall_ns, camera=camera)
        return out

    # -- measurement -------------------------------------------------------

    def locate(
        self,
        frames: Mapping[str, FrameData],
        text_prompt: str,
        camera: str = "cam_head",
        *,
        min_score: float | None = None,
        track: bool = False,
        geometry_mode: str = "supported",
        support_z_m: float | None = None,
        object_height_m: float | None = None,
    ) -> dict[str, Any]:
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        primary = frames.get(camera)
        if primary is None:
            raise PerceptionError(f"camera {camera!r} is not in this snapshot")
        if geometry_mode not in ("supported", "held"):
            raise ValueError("geometry_mode must be 'supported' or 'held'")
        support_z = self.calibration.table.z_m if support_z_m is None else float(support_z_m)
        if not math.isfinite(support_z):
            raise ValueError("support_z_m must be finite")
        if object_height_m is not None and (
            not math.isfinite(float(object_height_m)) or float(object_height_m) <= 0
        ):
            raise ValueError("object_height_m must be positive and finite")
        score_floor = self._score_floor(min_score)
        expected = self.tracker.expectation(prompt) if track else None
        if expected is not None and expected.held:
            geometry_mode = "held"
            object_height_m = float(expected.size[2])
        reason = self.tracker.failure(prompt) if track and not _table_prompt(prompt) else None
        result, roi = (
            (Segment(False, reason=reason), None)
            if reason
            else self._target_segment(primary, prompt, score_floor, expected)
        )
        out: dict[str, Any] = {
            "camera": camera,
            "found": bool(result.found),
            "text_prompt": prompt,
            "score": result.score,
            "views": 1,
            "measurement_failed": True,
            "geometry_confirmed": False,
            "search_roi_rc": roi,
            "captured_at_wall_ns": primary.captured_at_wall_ns,
            "geometry_mode": geometry_mode,
            "tracking_enabled": track,
        }
        if not result.found or result.mask is None:
            out["note"] = result.reason or "SAM3 found no mask above the score threshold"
            return self._unconfirmed(out, out["note"])
        out.update(_mask_stats(result.mask))
        mask = result.mask
        projection = self._table_projection(primary, mask)
        out["table_projection"] = projection
        if _table_prompt(prompt):
            # This is the configured support surface, not an object whose
            # height should be inferred from outliers inside a table mask.
            table = self.calibration.table
            out.update(
                {
                    "source": "calibrated_table",
                    "measurement_failed": False,
                    "center_xyz": [
                        (table.x_min_m + table.x_max_m) / 2,
                        (table.y_min_m + table.y_max_m) / 2,
                        table.z_m,
                    ],
                    "top_z": table.z_m,
                    "height_m": 0.0,
                    "length_m": table.x_max_m - table.x_min_m,
                    "width_m": table.y_max_m - table.y_min_m,
                    "note": "Configured table plane and bounds; not a depth measurement.",
                    "measurement_quality": {"source": "calibration"},
                }
            )
            return out
        if geometry_mode == "held" and object_height_m is None:
            return self._unconfirmed(out, "held objects require a measured pregrasp height")
        if expected is not None and expected.held and not expected.upright:
            return self._unconfirmed(out, "tilted held-object top geometry cannot be confirmed")
        points, depth_quality = _robust_depth_points(primary, mask)
        depth_ratio = float(depth_quality["depth_valid_ratio"])
        if (
            points.shape[0] >= 12
            and depth_ratio >= self.sam3.depth_min_valid_ratio
            and depth_quality.get("depth_rejected_ratio", 0.0) <= 0.25
        ):
            measurement = self._measure_depth(
                primary,
                mask,
                points,
                support_z,
                geometry_mode,
                object_height_m,
                expected,
            )
            if measurement is not None:
                measurement.quality.update(depth_quality)
                return self._finish_target(out, measurement, prompt, primary, track, mask)
            # A tracked target with contradictory depth is not rescued by another fit.
            if expected is not None:
                return self._unconfirmed(out, "no credible top plane in the target search volume")
        masks: dict[str, np.ndarray] = {camera: mask}
        scores: dict[str, float | None] = {camera: result.score}
        for name, frame in frames.items():
            if name == camera:
                continue
            other, _ = self._target_segment(frame, prompt, score_floor, expected)
            if not other.found or other.mask is None:
                continue
            share = float(other.mask.mean())
            if share > 0.30 or other.mask.sum() < 12:
                continue
            masks[name] = other.mask
            scores[name] = other.score
        if len(masks) < 2:
            if len(points) >= 12 and depth_ratio >= self.sam3.depth_min_valid_ratio:
                out["note"] = (
                    "measurement_failed: depth cannot establish a credible top plane; "
                    "no second matched calibrated view"
                )
            else:
                out["note"] = (
                    "measurement_failed: one view cannot determine an object's height; no second "
                    "calibrated view produced a mask"
                )
            out["geometry_error"] = out["note"]
            out["depth_valid_ratio"] = round(depth_ratio, 3)
            out["measurement_quality"] = depth_quality
            return self._unconfirmed(out, out["note"])
        z_bounds = None
        if geometry_mode == "held" and expected is None:
            return self._unconfirmed(
                out, "RGB held-object geometry requires a tracked 3-D search volume"
            )
        if expected is not None:
            z_bounds = (
                float(expected.center[2] - expected.extent[2] / 2 - 0.008),
                expected.top_z + 0.008,
            )
        elif support_z_m is not None:
            z_bounds = (support_z + 0.002, support_z + self.sam3.max_object_height_m)
        carved = self._carve(frames, masks, z_bounds=z_bounds)
        if carved is None or carved[0].shape[0] < 12:
            out["note"] = "measurement_failed: the calibrated views do not intersect on an object"
            out["geometry_error"] = out["note"]
            out["views"] = len(masks)
            return self._unconfirmed(out, out["note"])
        points, quality = carved
        measurement = self._measure_points(
            points,
            source="multiview",
            views=len(masks),
            support_z=support_z,
            object_height_m=object_height_m if geometry_mode == "held" else None,
        )
        if measurement is None:
            out["note"] = "measurement_failed: too few voxels above the table"
            out["geometry_error"] = out["note"]
            out["views"] = len(masks)
            return self._unconfirmed(out, out["note"])
        measurement.quality.update(quality)
        self._multiview_quality(measurement, frames, masks, support_z)
        out["view_scores"] = {k: v for k, v in scores.items()}
        return self._finish_target(out, measurement, prompt, primary, track, mask)

    @staticmethod
    def _multiview_quality(measurement, frames, masks, support_z):
        """Assess visual-hull resolution and geometry; RGB is not always invalid."""
        quality = measurement.quality
        voxel = float(quality["voxel_m"])
        size = np.array([measurement.length_m, measurement.width_m, measurement.height_m])
        center = np.asarray(measurement.center_xyz)
        rays = [center - frames[name].camera_to_world[:3, 3] for name in masks]
        rays = [ray / np.linalg.norm(ray) for ray in rays if np.linalg.norm(ray) > 1e-9]
        parallax = max(
            (
                float(np.rad2deg(np.arccos(np.clip(a @ b, -1, 1))))
                for i, a in enumerate(rays)
                for b in rays[i + 1 :]
            ),
            default=0.0,
        )
        fill = float(quality["volume_m3"] / max(float(np.prod(size)), 1e-12))
        angle = np.deg2rad(measurement.long_axis_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        corners = (
            center
            + np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            * size
            / 2
            @ rotation.T
        )
        overlaps = {}
        for name, mask in masks.items():
            overlaps[name] = _box_mask_overlap(frames[name], corners, mask)
        resolved = bool(
            parallax >= 10
            and np.all(size >= 3 * voxel)
            and 0.25 <= fill <= 1.5
            and min(overlaps.values(), default=0) >= 0.50
        )
        bottom_reference = (
            measurement.top_z - measurement.height_m
            if quality.get("height_reference") == "measured_pregrasp_height"
            else support_z
        )
        bottom_gap = abs(float(quality["observed_bottom_z_m"]) - bottom_reference)
        quality.update(
            parallax_deg=round(parallax, 2),
            hull_fill_ratio=round(fill, 3),
            box_mask_iou=overlaps,
            bottom_reference_z_m=float(bottom_reference),
            quantization_uncertainty_m={
                "center": float(np.sqrt(3) * voxel / 2),
                "height": voxel / 2,
                "footprint": float(np.sqrt(2) * voxel / 2),
            },
            height_reliable=resolved and voxel / 2 <= 0.004 and bottom_gap <= 2 * voxel,
            center_reliable=bool(resolved and np.sqrt(3) * voxel / 2 <= 0.008),
            footprint_reliable=bool(resolved and np.sqrt(2) * voxel / 2 <= 0.004),
        )

    def _finish_target(self, out, measurement, prompt, frame, track, mask):
        result = self._finish(out, measurement)
        result["rgb_footprint"] = self._silhouette_projection(frame, mask, measurement.top_z)
        quality = measurement.quality
        if measurement.source == "depth" and not (
            quality["center_reliable"] and quality["footprint_reliable"]
        ):
            result["partial_measurement"] = {
                "top_z": result["top_z"],
                "height_m": result["height_m"],
                "source": "depth_top_plane",
                "authorizes_target": False,
            }
            return self._unconfirmed(
                result, "top plane is visible but the centre/footprint is occluded"
            )
        if track:
            reason = self.tracker.validate(prompt, result)
            if reason:
                self.tracker.reject(prompt, frame.captured_at_wall_ns, force=True)
                return self._unconfirmed(result, reason)
            self.tracker.accept(prompt, result, frame.captured_at_wall_ns)
        return result

    def _measure_depth(self, frame, mask, points, support_z, mode, known_height, expected):
        """Fit a measured horizontal top and its depth-only footprint.

        A partial patch cannot establish the centre of an occluded object.
        Held objects use their measured pregrasp height, never the table gap.
        """
        pts = points[np.all(np.isfinite(points), axis=1)]
        if expected is not None:
            pts = pts[expected.contains(pts)]
            pts = pts[np.abs(pts[:, 2] - expected.top_z) <= 0.005]
        elif mode == "supported":
            pts = pts[
                (pts[:, 2] > support_z + 0.004)
                & (pts[:, 2] < support_z + self.sam3.max_object_height_m)
            ]
        if len(pts) < 12:
            return None
        # Deterministic bounded RANSAC: favour support, then fit all inliers.
        rng = np.random.default_rng(0)
        candidates = [np.array([0.0, 0.0, float(np.median(pts[:, 2]))])]
        for _ in range(80):
            sample = pts[rng.choice(len(pts), 3, replace=False)]
            design = np.column_stack([sample[:, :2], np.ones(3)])
            if np.linalg.matrix_rank(design) == 3:
                fit = np.linalg.solve(design, sample[:, 2])
                if np.linalg.norm(fit[:2]) <= np.tan(np.deg2rad(10)):
                    candidates.append(fit)
        design = np.column_stack([pts[:, :2], np.ones(len(pts))])
        fit = max(
            candidates, key=lambda p: np.count_nonzero(np.abs(design @ p - pts[:, 2]) <= 0.0025)
        )
        inliers = np.abs(design @ fit - pts[:, 2]) <= 0.0025
        if inliers.sum() < max(12, int(0.4 * len(pts))):
            return None
        fit = _bounded_top_plane(pts[inliers], np.tan(np.deg2rad(10)))
        residual = design[inliers] @ fit - pts[inliers, 2]
        rms = float(np.sqrt(np.mean(residual**2)))
        if rms > 0.002:
            return None
        # Use raw-mask depth only if it lies on the fitted plane. This recovers
        # measured edges eroded for fitting without projecting an RGB silhouette.
        raw, _ = _depth_points(frame, mask)
        raw_design = np.column_stack([raw[:, :2], np.ones(len(raw))])
        raw_top = raw[np.abs(raw_design @ fit - raw[:, 2]) <= 0.003]
        if expected is not None:
            inside = expected.contains(raw_top)
            if len(inside) and inside.mean() < 0.90:
                return None  # Cropping depth must not shrink an oversized target into a valid one.
        top_points = raw_top
        coverage = len(top_points) / max(1, int(mask.sum()))
        if len(top_points) < 12:
            return None
        heading, length, width, mid = _footprint(top_points[:, :2])
        if width < 0.003 or length < 0.003:
            return None
        at_xy = expected.center[:2] if expected is not None else mid
        top = float(np.r_[at_xy, 1.0] @ fit)
        height = float(known_height) if mode == "held" else top - support_z
        if height < 0.004:
            return None
        # A slanted view legitimately contains vertical side pixels. Verify
        # them against the depth-derived upright box instead of treating every
        # non-top pixel as an occluder. Missing/foreground depth and an unseen
        # half-top cannot establish a full box centre.
        angle = np.deg2rad(heading)
        rotation_xy = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local_xy = (raw[:, :2] - mid) @ rotation_xy
        on_object = (
            np.all(np.abs(local_xy) <= np.array([length, width]) / 2 + 0.004, axis=1)
            & (raw[:, 2] >= top - height + 0.002)
            & (raw[:, 2] <= top + 0.003)
        )
        object_coverage = float(on_object.sum() / max(1, int(mask.sum())))
        complete = object_coverage >= 0.75 and coverage >= 0.20
        return Measurement(
            center_xyz=(float(mid[0]), float(mid[1]), top - height / 2),
            top_z=top,
            height_m=height,
            length_m=length,
            width_m=width,
            long_axis_deg=heading,
            grasp_yaw=wrap_deg(heading + 90),
            source="depth",
            views=1,
            points=len(top_points),
            quality={
                "source": "depth",
                "views": 1,
                "points": len(top_points),
                "height_reference": "measured_pregrasp_height"
                if mode == "held"
                else "configured_table_plane"
                if support_z == self.calibration.table.z_m
                else "explicit_support_plane",
                "height_source": "pregrasp_measurement"
                if mode == "held"
                else "depth_top_plane_and_support",
                "height_reliable": True,
                "center_reliable": complete,
                "footprint_reliable": complete,
                "center_z_source": "measured_top_minus_half_height",
                "center_source": "depth_top_plane_footprint",
                "footprint_source": "depth_top_plane",
                "footprint_is_estimate": False,
                "top_plane": {
                    "z_xy_coefficients": fit.tolist(),
                    "rms_m": rms,
                    "inliers": int(inliers.sum()),
                    "coverage": round(coverage, 3),
                },
                "object_depth_coverage": round(object_coverage, 3),
                "foreground_rejected_ratio": round(
                    1 - len(top_points) / max(1, int(mask.sum())), 3
                ),
            },
        )

    def _table_projection(self, frame: FrameData, mask: np.ndarray) -> dict[str, Any] | None:
        return self._silhouette_projection(frame, mask, self.calibration.table.z_m)

    def _silhouette_projection(
        self, frame: FrameData, mask: np.ndarray, z: float
    ) -> dict[str, Any] | None:
        rows, cols = np.nonzero(mask)
        if len(rows) < 12:
            return None
        stride = max(1, len(rows) // 5000)
        hits = [
            _ray_plane(frame, int(r), int(c), z)
            for r, c in zip(rows[::stride], cols[::stride], strict=True)
        ]
        xy = np.asarray([hit[:2] for hit in hits if hit is not None])
        if len(xy) < 12:
            return None
        heading, length, width, mid = _footprint(xy)
        table = self.calibration.table
        within = (
            table.x_min_m <= mid[0] <= table.x_max_m and table.y_min_m <= mid[1] <= table.y_max_m
        )
        return {
            "center_xy": [float(v) for v in mid],
            "length_m": length,
            "width_m": width,
            "long_axis_deg": heading,
            "grasp_yaw": wrap_deg(heading + 90.0),
            "plane_z_m": z,
            "source": "rgb_plane_projection",
            "is_estimate": True,
            "within_table_bounds": bool(within),
            "note": "Silhouette projected onto a horizontal plane; perspective and occlusion "
            "can shift/shorten this footprint. It does not measure object height.",
        }

    def _finish(self, out: dict[str, Any], measurement: Measurement) -> dict[str, Any]:
        table = self.calibration.table
        x, y = measurement.center_xyz[:2]
        half = max(measurement.length_m, measurement.width_m) / 2
        within = table.x_min_m <= x <= table.x_max_m and table.y_min_m <= y <= table.y_max_m
        footprint_within = (
            table.x_min_m + half <= x <= table.x_max_m - half
            and table.y_min_m + half <= y <= table.y_max_m - half
        )
        out.update(
            {
                "center_xyz": [round(v, 4) for v in measurement.center_xyz],
                "top_z": round(measurement.top_z, 4),
                "height_m": round(measurement.height_m, 4),
                "length_m": round(measurement.length_m, 4),
                "width_m": round(measurement.width_m, 4),
                "long_axis_deg": round(measurement.long_axis_deg, 1),
                "grasp_yaw": round(measurement.grasp_yaw, 1),
                "views": measurement.views,
                "source": measurement.source,
                "measurement_failed": False,
                "measurement_quality": measurement.quality,
                "within_table_bounds": bool(within),
                "footprint_within_table_bounds": bool(footprint_within),
                "bounds_error": None if footprint_within else "outside_calibrated_table",
            }
        )
        if measurement.source == "multiview":
            out["note"] = (
                "RGB-only multi-view measurement: height comes from intersecting calibrated "
                "silhouettes; hidden or thin parts remain uncertain"
            )
        elif measurement.source == "depth":
            out["note"] = (
                "Centre and footprint use measured top-plane depth, not RGB silhouette projection. "
                "Check component reliability and temporal confirmation before using this geometry."
            )
        elif measurement.quality.get("footprint_is_estimate"):
            out["note"] = (
                "Height uses robust interior depth; x/y and footprint use the RGB silhouette "
                "at estimated mid-height. Occlusion and perspective can bias the footprint; "
                "table_projection gives an independent, depth-free table-plane estimate."
            )
        if not footprint_within:
            out["note"] = (
                (out.get("note") or "")
                + " Object was measured outside the calibrated table footprint. "
                "Height is relative to the configured table plane, extrapolated here; "
                "this measurement does not authorize a grasp. Confirm the physical support "
                "and its calibration, or move the object into the calibrated operating region."
            ).strip()
        return out

    def _measure_points(
        self,
        points: np.ndarray,
        *,
        source: str,
        views: int,
        support_z: float | None = None,
        object_height_m: float | None = None,
    ) -> Measurement | None:
        table = self.calibration.table
        support_z = table.z_m if support_z is None else support_z
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 12:
            return None
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        if object_height_m is None:
            pts = pts[
                (pts[:, 2] > support_z + 0.002)
                & (pts[:, 2] < support_z + self.sam3.max_object_height_m)
            ]
        if pts.shape[0] < 12:
            return None
        top = float(np.percentile(pts[:, 2], 97))
        low = float(np.percentile(pts[:, 2], 2))
        height = top - support_z if object_height_m is None else float(object_height_m)
        if height < 0.004:
            return None
        # A visual hull's full extent is needed for resolution/fill assessment;
        # a thin bottom slice can underestimate its footprint and move its centre.
        heading, length, width, mid = _footprint(pts[:, :2])
        return Measurement(
            center_xyz=(float(mid[0]), float(mid[1]), float(top - height / 2)),
            top_z=top,
            height_m=height,
            length_m=length,
            width_m=width,
            long_axis_deg=heading,
            grasp_yaw=wrap_deg(heading + 90.0),
            source=source,
            views=views,
            points=int(pts.shape[0]),
            quality={
                "source": source,
                "views": views,
                "points": int(pts.shape[0]),
                "height_quantile": 0.97,
                "height_reference": "configured_table_plane"
                if object_height_m is None
                else "measured_pregrasp_height",
                "observed_bottom_z_m": low,
                "center_source": "multiview_visual_hull",
                "footprint_source": "multiview_visual_hull",
            },
        )

    def _carve(
        self,
        frames: Mapping[str, FrameData],
        masks: Mapping[str, np.ndarray],
        *,
        z_bounds: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        table = self.calibration.table
        primary = next(iter(masks))
        frame = frames[primary]
        rows, cols = np.nonzero(masks[primary])
        if rows.size == 0:
            return None
        # Bounds from the primary mask's rays at the table and at maximum height.
        pts = []
        z_lo, z_hi = z_bounds or (table.z_m + 0.002, table.z_m + self.sam3.max_object_height_m)
        for z in (z_lo, z_hi):
            for row, col in zip(
                rows[:: max(1, rows.size // 200)], cols[:: max(1, cols.size // 200)], strict=True
            ):
                p = _ray_plane(frame, int(row), int(col), z)
                if p is not None:
                    pts.append(p)
        if not pts:
            return None
        cloud = np.asarray(pts)
        x_lo, y_lo = cloud[:, 0].min(), cloud[:, 1].min()
        x_hi, y_hi = cloud[:, 0].max(), cloud[:, 1].max()
        margin = max(0.02, self.sam3.voxel_m * 2)
        # Perception searches the calibrated camera rays, independently of
        # the permitted table footprint. Cropping to the motion region used
        # to discard valid outside objects and report a false triangulation
        # failure. The voxel budget still bounds this search.
        x_lo, x_hi = x_lo - margin, x_hi + margin
        y_lo, y_hi = y_lo - margin, y_hi + margin
        if (
            not all(math.isfinite(v) for v in (x_lo, x_hi, y_lo, y_hi))
            or x_hi <= x_lo
            or y_hi <= y_lo
        ):
            return None
        step = float(self.sam3.voxel_m)
        nx = max(1, int(math.ceil((x_hi - x_lo) / step)))
        ny = max(1, int(math.ceil((y_hi - y_lo) / step)))
        nz = max(1, int(math.ceil((z_hi - z_lo) / step)))
        while nx * ny * nz > self.sam3.max_voxels:
            step *= 1.25
            nx = max(1, int(math.ceil((x_hi - x_lo) / step)))
            ny = max(1, int(math.ceil((y_hi - y_lo) / step)))
            nz = max(1, int(math.ceil((z_hi - z_lo) / step)))
        xs = x_lo + (np.arange(nx) + 0.5) * step
        ys = y_lo + (np.arange(ny) + 0.5) * step
        zs = z_lo + (np.arange(nz) + 0.5) * step
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        voxels = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        keep = np.ones(len(voxels), dtype=bool)
        for name, mask in masks.items():
            view = frames[name]
            r, c, depth = _project(view, voxels)
            valid = depth > 0
            ri = np.rint(r).astype(np.int64)
            ci = np.rint(c).astype(np.int64)
            inside = np.zeros(len(voxels), dtype=bool)
            ok = valid & (ri >= 0) & (ri < mask.shape[0]) & (ci >= 0) & (ci < mask.shape[1])
            inside[ok] = mask[ri[ok], ci[ok]]
            # One-pixel dilation accounts for silhouette quantisation.
            dilated = np.zeros_like(mask)
            dilated[1:-1, 1:-1] = (
                mask[:-2, 1:-1]
                | mask[2:, 1:-1]
                | mask[1:-1, :-2]
                | mask[1:-1, 2:]
                | mask[1:-1, 1:-1]
            )
            inside_d = np.zeros(len(voxels), dtype=bool)
            inside_d[ok] = dilated[ri[ok], ci[ok]]
            keep &= inside_d
            if not np.any(keep):
                return None
        occupied = voxels[keep]
        if occupied.shape[0] < 12:
            return None
        quality = {
            "voxel_m": step,
            "voxels": int(occupied.shape[0]),
            "grid": [nx, ny, nz],
            "views": len(masks),
            "volume_m3": float(occupied.shape[0] * step**3),
            "search_xy_bounds": [[float(x_lo), float(x_hi)], [float(y_lo), float(y_hi)]],
            "table_bounds_constrained": False,
        }
        return occupied, quality

    # -- per-pixel ---------------------------------------------------------

    def back_project(
        self,
        frames: Mapping[str, FrameData],
        row: int,
        col: int,
        camera: str = "cam_head",
    ) -> dict[str, Any]:
        frame = frames.get(camera)
        if frame is None:
            raise PerceptionError(f"camera {camera!r} is not in this snapshot")
        height, width = frame.shape
        if not (0 <= int(row) < height and 0 <= int(col) < width):
            raise ValueError(f"pixel ({row}, {col}) is outside {width}x{height}")
        wall = frame.captured_at_wall_ns
        if frame.depth is not None:
            depth = float(frame.depth[int(row), int(col)])
            if math.isfinite(depth) and depth > 0:
                world = depth_points(
                    frame.intrinsic,
                    frame.camera_to_world,
                    np.asarray(row),
                    np.asarray(col),
                    np.asarray(depth),
                )
                surface = "object" if world[2] > self.calibration.table.z_m + 0.003 else "table"
                return {
                    "pixel": [int(row), int(col)],
                    "world_xyz": [round(float(v), 5) for v in world],
                    "depth_m": round(depth, 5),
                    "camera": camera,
                    "surface": surface,
                    "measurement_failed": False,
                    "captured_at_wall_ns": wall,
                }
        # A generic mask in another view cannot establish correspondence to
        # this pixel. Never turn an unknown surface into a table/top-plane hit.
        raise PerceptionError(
            "no reliable depth at this pixel; use locate with an explicit object prompt "
            "for matched multi-view geometry or the calibrated table projection"
        )


__all__ = [
    "FrameData",
    "Measurement",
    "Perception",
    "PerceptionError",
    "Sam3Segmenter",
    "Segment",
    "frame_from_observation",
]
