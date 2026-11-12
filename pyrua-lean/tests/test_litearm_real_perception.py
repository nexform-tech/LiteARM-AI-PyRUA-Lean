"""Calibrated RGB-D and multi-view measurement tests with known geometry."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace

import numpy as np
import pytest

from pyrualean.hardware.perception import (
    FrameData,
    Measurement,
    Perception,
    PerceptionError,
    Segment,
    _bounded_top_plane,
    frame_from_observation,
)

H, W = 48, 64
K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
BOX = {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (0.0, 0.10)}


def _look_at(position, target) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    z = position - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        # Looking straight down/up: any horizontal x is valid; pick world +x.
        x = np.array([1.0, 0.0, 0.0])
        x = x - z * np.dot(x, z)
        norm = np.linalg.norm(x)
    x /= norm
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x, y, z, position
    return t


def _project(c2w, points):
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    depth = -local[:, 2]
    safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    rows = -K[1, 1] * local[:, 1] / safe + K[1, 2]
    cols = K[0, 0] * local[:, 0] / safe + K[0, 2]
    return rows, cols, depth


def _box_mask(c2w) -> np.ndarray:
    import cv2

    corners = np.array([[x, y, z] for x in BOX["x"] for y in BOX["y"] for z in BOX["z"]])
    rows, cols, depth = _project(c2w, corners)
    pts = np.stack([cols, rows], axis=1)
    pts = np.round(pts).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    return mask.astype(bool)


def _rgb(mask):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[mask] = (220, 30, 30)
    return image


def _top_depth():
    c2w = _look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    depth = np.full((H, W), np.nan, dtype=np.float32)
    rows, cols = np.mgrid[0:H, 0:W]
    local = np.stack(
        [
            (cols - K[0, 2]) / K[0, 0],
            -(rows - K[1, 2]) / K[1, 1],
            -np.ones_like(rows),
        ],
        axis=-1,
    ).astype(float)
    # Each ray meets the box top (z=0.10) at t=0.90 and the table (z=0) at
    # t=1.0.  A pixel inside the box footprint reports the top's depth; every
    # other pixel reports the table.
    for r in range(H):
        for c in range(W):
            direction = local[r, c]
            p = c2w[:3, 3] + direction * 0.90
            if BOX["x"][0] <= p[0] <= BOX["x"][1] and BOX["y"][0] <= p[1] <= BOX["y"][1]:
                depth[r, c] = 0.90
            else:
                depth[r, c] = 1.0
    return depth, c2w


@dataclass
class _Scripted:
    masks: dict[str, np.ndarray]

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        raise AssertionError("the scripted segmenter is called through the camera map")


class _PerCameraSegmenter:
    def __init__(self, masks):
        self.masks = masks

    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        # The test encodes the camera in the red channel intensity of the image's
        # top-left pixel: 1 -> head, 2 -> side, 3 -> other.
        key = int(image[0, 0, 1])
        mask = self.masks[key]
        if point is not None:
            row, col = int(point[0]), int(point[1])
            if 0 <= row < H and 0 <= col < W and mask[row, col]:
                return Segment(True, 0.9, mask, [0, 0, W, H])
            return Segment(False, reason="no object at point")
        return Segment(True, 0.9, mask, [0, 0, W, H])


def _frame(name, key, c2w, mask, depth=None):
    rgb = _rgb(mask)
    if key:
        rgb[0, 0, 1] = key
    return FrameData(name, rgb, depth, K.copy(), c2w, 1)


def test_depth_measurement_uses_measured_top_and_footprint():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    segmenter = _PerCameraSegmenter({1: mask})
    perception = Perception(
        _calibration(),
        _sam3(),
        segmenter=segmenter,
    )
    frame = _frame("cam_head", 1, c2w, mask, depth)
    result = perception.locate({"cam_head": frame}, "the red block")
    assert result["found"] and not result["measurement_failed"]
    assert result["source"] == "depth"
    assert result["top_z"] == pytest.approx(0.10, abs=0.005)
    assert result["height_m"] == pytest.approx(0.10, abs=0.005)
    assert result["center_xyz"][0] == pytest.approx(0.0, abs=0.005)
    assert result["center_xyz"][1] == pytest.approx(0.0, abs=0.005)
    assert result["width_m"] == pytest.approx(0.04, abs=0.01)
    assert result["length_m"] == pytest.approx(0.06, abs=0.01)
    assert result["views"] >= 1
    assert result["measurement_quality"]["depth_valid_ratio"] > 0.9


def test_one_rgb_view_cannot_measure_height():
    c2w = _look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    mask = _box_mask(c2w)
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate({"cam_head": _frame("cam_head", 1, c2w, mask)}, "the red block")
    assert result["found"] is True
    assert result["measurement_failed"] is True
    assert result.get("center_xyz") is None
    assert "one view" in result["geometry_error"]


def test_outside_table_depth_is_measured_and_separately_flagged():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    cal = _calibration()
    cal = replace(cal, table=replace(cal.table, y_min_m=0.2))
    perception = Perception(cal, _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate({"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "red block")
    assert not result["measurement_failed"] and result["source"] == "depth"
    assert result["top_z"] == pytest.approx(0.10, abs=0.005)
    assert not result["within_table_bounds"] and not result["footprint_within_table_bounds"]
    assert result["bounds_error"] == "outside_calibrated_table"
    assert "extrapolated" in result["note"]
    assert result["measurement_quality"]["height_reference"] == "configured_table_plane"
    assert cal.table.y_min_m == 0.2


def test_outside_table_multiview_uses_camera_rays_without_table_crop():
    top = _look_at([0.0, -0.25, 0.60], [0.0, 0.0, 0.05])
    side = _look_at([0.0, -0.55, 0.35], [0.0, 0.0, 0.05])
    masks = {1: _box_mask(top), 2: _box_mask(side)}
    cal = _calibration()
    cal = replace(cal, table=replace(cal.table, y_min_m=0.2))
    perception = Perception(cal, _sam3(), segmenter=_PerCameraSegmenter(masks))
    result = perception.locate(
        {
            "cam_head": _frame("cam_head", 1, top, masks[1]),
            "cam_l": _frame("cam_l", 2, side, masks[2]),
        },
        "red block",
    )
    assert not result["measurement_failed"] and result["source"] == "multiview"
    assert result["bounds_error"] == "outside_calibrated_table"
    assert result["measurement_quality"]["table_bounds_constrained"] is False
    assert result["center_xyz"][1] < 0.1  # known object remains outside the y>=0.2 region


def test_inside_table_depth_preserves_bounds_status():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    cal = _calibration()
    cal = replace(cal, table=replace(cal.table, y_min_m=-0.1))
    perception = Perception(cal, _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate({"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "red block")
    assert result["within_table_bounds"] and result["footprint_within_table_bounds"]
    assert result["bounds_error"] is None


def test_two_views_measure_a_visual_hull():
    top = _look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    side = _look_at([0.0, -0.55, 0.35], [0.0, 0.0, 0.05])
    masks = {1: _box_mask(top), 2: _box_mask(side)}
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter(masks))
    result = perception.locate(
        {
            "cam_head": _frame("cam_head", 1, top, masks[1]),
            "cam_l": _frame("cam_l", 2, side, masks[2]),
        },
        "the red block",
    )
    assert result["found"] and not result["measurement_failed"]
    assert result["source"] == "multiview"
    assert result["views"] == 2
    assert result["top_z"] == pytest.approx(0.10, abs=0.035)
    assert result["center_xyz"][0] == pytest.approx(0.0, abs=0.02)
    assert result["measurement_quality"]["height_reliable"]
    assert result["measurement_quality"]["center_reliable"]
    assert result["measurement_quality"]["footprint_reliable"]


def test_invalid_depth_falls_back_or_fails_explicitly():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    depth[:] = np.nan
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate(
        {"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "the red block"
    )
    assert result["measurement_failed"] is True
    assert result.get("center_xyz") is None


def test_back_project_uses_valid_depth_and_marks_table():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    frame = _frame("cam_head", 1, c2w, mask, depth)
    row, col = np.argwhere(mask)[0]
    point = perception.back_project({"cam_head": frame}, int(row), int(col))
    assert point["surface"] == "object"
    assert point["world_xyz"][2] == pytest.approx(0.10, abs=0.005)
    table_pixel = (0, 0)
    table = perception.back_project({"cam_head": frame}, *table_pixel)
    assert table["surface"] == "table"
    assert table["world_xyz"][2] == pytest.approx(0.0, abs=0.005)


def test_edge_depth_from_an_occluding_arm_does_not_become_the_object_top():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    rows, cols = np.nonzero(mask)
    depth[rows[rows == rows.min()], cols[rows == rows.min()]] = 0.65
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate({"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "can")
    assert not result["measurement_failed"]
    assert result["top_z"] == pytest.approx(0.10, abs=0.005)
    assert result["measurement_quality"]["depth_interior_margin_px"] > 0
    assert result["measurement_quality"]["footprint_source"] == "depth_top_plane"


def test_table_projection_survives_missing_depth_without_inventing_height():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    depth[:] = np.nan
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate({"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "can")
    assert result["measurement_failed"]
    assert result.get("top_z") is None and result.get("center_xyz") is None
    assert result["table_projection"]["center_xy"] == pytest.approx([0, 0], abs=0.01)
    assert result["table_projection"]["is_estimate"]


def test_table_query_uses_the_calibrated_support_not_object_depth():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    depth[:] = 0.65
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    result = perception.locate(
        {"cam_head": _frame("cam_head", 1, c2w, mask, depth)}, "the white tabletop"
    )
    assert result["source"] == "calibrated_table"
    assert result["height_m"] == 0.0
    assert result["top_z"] == _calibration().table.z_m


def test_table_projection_does_not_depend_on_depth():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    frame = _frame("cam_head", 1, c2w, mask, depth)
    first = perception.locate({"cam_head": frame}, "can")
    frame.depth = np.full_like(depth, np.nan)
    second = perception.locate({"cam_head": frame}, "can")
    assert first["table_projection"] == second["table_projection"]
    assert first["measurement_failed"] is False and second["measurement_failed"] is True


def test_missing_pixel_depth_never_falls_back_to_a_generic_object_or_table():
    depth, c2w = _top_depth()
    mask = _box_mask(c2w)
    depth[:] = np.nan
    perception = Perception(_calibration(), _sam3(), segmenter=_PerCameraSegmenter({1: mask}))
    frame = _frame("cam_head", 1, c2w, mask, depth)
    with pytest.raises(PerceptionError, match="no reliable depth"):
        perception.back_project({"cam_head": frame}, H // 2, W // 2)


def test_frame_loader_refuses_mismatched_depth_and_missing_artifacts():
    class Obs:
        camera_state = {
            "cam_head": {
                "color_artifact": "a.png",
                "depth_artifact": "a_depth.npy",
                "intrinsic_matrix": K.tolist(),
                "camera_to_world": np.eye(4).tolist(),
                "captured_at_wall_ns": 1,
            }
        }

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (W, H), (1, 2, 3)).save(buf, format="PNG")
    depth = np.zeros((H + 1, W), dtype=np.float32)
    dbuf = io.BytesIO()
    np.save(dbuf, depth)
    with pytest.raises(PerceptionError, match="depth shape"):
        frame_from_observation(
            Obs(), {"a.png": buf.getvalue(), "a_depth.npy": dbuf.getvalue()}, "cam_head"
        )
    with pytest.raises(PerceptionError, match="not retained"):
        frame_from_observation(Obs(), {"a_depth.npy": dbuf.getvalue()}, "cam_head")


def _calibration():
    from pathlib import Path

    from pyrualean.hardware.config import load_station

    root = Path(__file__).resolve().parents[1]
    return load_station(root / "configs" / "litearm_real.example.json").calibration


def _sam3():
    from pathlib import Path

    from pyrualean.hardware.config import load_station

    root = Path(__file__).resolve().parents[1]
    return load_station(root / "configs" / "litearm_real.example.json").sam3


@pytest.mark.parametrize("target,expected_top", [("green", 0.1397), ("black", 0.1376)])
def test_recorded_horaud_cube_depth_survives_constrained_plane_refinement(target, expected_top):
    """Unconstrained refinement rejected these 100%-valid recorded depth masks."""
    from pathlib import Path

    saved = np.load(Path(__file__).parent / "data/park_horaud_cube_depth.npz")
    h, w = saved["shape"]
    mask = np.zeros((h, w), dtype=bool)
    depth = np.full((h, w), np.nan, dtype=np.float32)
    r, c = saved[target + "_origin"]
    mh, mw = saved[target + "_mask"].shape
    mask[r : r + mh, c : c + mw] = saved[target + "_mask"]
    depth[r : r + mh, c : c + mw] = saved[target + "_depth"]
    frame = FrameData(
        "cam_head",
        np.zeros((h, w, 3), dtype=np.uint8),
        depth,
        saved["intrinsic"],
        saved["camera_to_world"],
        1,
    )

    class SavedMask:
        def segment(self, image, **kwargs):
            return Segment(True, 0.9, mask)

    calibration = _calibration()
    calibration = replace(
        calibration, table=replace(calibration.table, z_m=float(saved["table_z"]))
    )
    perception = Perception(calibration, _sam3(), segmenter=SavedMask())
    result = perception.locate({"cam_head": frame}, "the " + target + " block")
    assert result["source"] == "depth" and not result["measurement_failed"]
    assert result["top_z"] == pytest.approx(expected_top, abs=0.0002)
    quality = result["measurement_quality"]
    assert quality["depth_valid_ratio"] == 1.0
    assert all(quality[k] for k in ("height_reliable", "center_reliable", "footprint_reliable"))
    slope = np.linalg.norm(quality["top_plane"]["z_xy_coefficients"][:2])
    assert slope <= np.tan(np.deg2rad(10)) + 1e-10
    assert quality["top_plane"]["rms_m"] < 0.002


@pytest.mark.parametrize(
    "condition", ["resolved", "same_view", "coarse", "floating", "wrong_silhouette", "held"]
)
def test_multiview_reliability_depends_on_resolution_parallax_and_support(condition):
    pose_a = _look_at([0, -0.25, 0.6], [0, 0, 0.05])
    pose_b = _look_at([-0.25, 0, 0.6], [0, 0, 0.05]) if condition != "same_view" else pose_a
    masks = {"a": _box_mask(pose_a), "b": _box_mask(pose_b)}
    frames = {"a": _frame("a", 0, pose_a, masks["a"]), "b": _frame("b", 0, pose_b, masks["b"])}
    measurement = Measurement(
        (0, 0, 0.05),
        0.10,
        0.10,
        0.06,
        0.04,
        0,
        90,
        "multiview",
        2,
        64,
        {
            "voxel_m": 0.02 if condition == "coarse" else 0.005,
            "volume_m3": 0.00024,
            "observed_bottom_z_m": 0.03 if condition == "floating" else 0.0025,
        },
    )
    if condition == "wrong_silhouette":
        measurement.center_xyz = (0, 0, 0.175)
        measurement.top_z = measurement.height_m = 0.35
    elif condition == "held":
        for frame in frames.values():
            frame.camera_to_world[2, 3] += 0.2
        measurement.center_xyz = (0, 0, 0.25)
        measurement.top_z = 0.3
        measurement.quality.update(
            height_reference="measured_pregrasp_height", observed_bottom_z_m=0.2025
        )
    Perception._multiview_quality(measurement, frames, masks, 0)
    q = measurement.quality
    if condition in ("resolved", "held"):
        assert all(q[k] for k in ("height_reliable", "center_reliable", "footprint_reliable"))
    elif condition == "floating":
        assert not q["height_reliable"] and q["center_reliable"]
    else:
        assert not any(q[k] for k in ("height_reliable", "center_reliable", "footprint_reliable"))
    assert all(
        type(q[k]) is bool for k in ("height_reliable", "center_reliable", "footprint_reliable")
    )


def test_top_plane_refinement_cannot_escape_its_slope_constraint():
    x, y = np.meshgrid(np.linspace(-0.01, 0.01, 8), np.linspace(-0.01, 0.01, 8))
    points = np.column_stack([x.ravel(), y.ravel(), (5 * x + 9 * y + 0.15).ravel()])
    bound = float(np.tan(np.deg2rad(10)))
    fitted = _bounded_top_plane(points, bound)
    assert np.linalg.norm(fitted[:2]) <= bound + 1e-12
    # Such a genuinely incompatible surface remains far outside the RMS gate.
    residual = points[:, 2] - np.column_stack([points[:, :2], np.ones(len(points))]) @ fitted
    assert np.sqrt(np.mean(residual**2)) > 0.002


def test_http_sam3_endpoint_is_used_and_a_thor_style_mask_is_honoured():
    """SAM3 runs on Thor; the agent only sends the RGB image over HTTP."""

    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from pyrualean.hosts.sam3_http import _pack_mask

    depth, c2w = _top_depth()
    mask = _box_mask(c2w)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            if self.path != "/segment":
                self.send_response(404)
                self.end_headers()
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["text_prompt"] == "the red block"
            payload = {
                "found": True,
                "score": 0.9,
                "box": [0, 0, 64, 48],
                "mask": _pack_mask(mask),
                "reason": None,
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        from dataclasses import replace

        sam3 = replace(_sam3(), mode="http", endpoint=endpoint, checkpoint=None)
        perception = Perception(_calibration(), sam3)
        frame = _frame("cam_head", 1, c2w, mask, depth)
        result = perception.locate({"cam_head": frame}, "the red block")
        assert result["found"] is True
        assert result["measurement_failed"] is False
        assert result["top_z"] == pytest.approx(0.10, abs=0.005)
    finally:
        server.shutdown()
        server.server_close()
