"""Optical-to-station camera conversion, PARK import, and no double conversion."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrualean.hardware.camera_geometry import (
    OPTICAL_TO_STATION,
    depth_points,
    project_points,
    ray_plane,
    station_transform,
)
from pyrualean.hardware.config import ConfigError, load_calibration, load_station
from pyrualean.hardware.handeye_import import import_park, main
from pyrualean.hardware.kinematics import LitearmKinematics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/data/stations/calibration/handeye_park_20261111.source.json"
PROFILE = ROOT / "tests/data/stations/litearm_single_right.measured-fingertip.json"
K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])


def test_opencv_projection_and_depth_share_the_station_conversion():
    optical = OPTICAL_TO_STATION.copy()
    optical[:3, 3] = [0, 0, 1]
    station = station_transform(optical, "opencv")
    points = np.array([[0.02, 0.03, 0.1], [-0.03, -0.02, 0.2]])
    rows, cols, depth = project_points(K, station, points)
    # Independent OpenCV optical calculation.
    local = (points - optical[:3, 3]) @ optical[:3, :3]
    np.testing.assert_allclose(cols, 100 * local[:, 0] / local[:, 2] + 32)
    np.testing.assert_allclose(rows, 100 * local[:, 1] / local[:, 2] + 24)
    np.testing.assert_allclose(depth_points(K, station, rows, cols, depth), points)
    for point, row, col in zip(points, rows, cols, strict=True):
        np.testing.assert_allclose(ray_plane(K, station, row, col, point[2]), point)
    np.testing.assert_allclose(station_transform(station, "opengl"), station)
    with pytest.raises(ValueError, match="camera_frame"):
        station_transform(station, "unknown")


def test_loader_normalizes_optical_calibration_once(tmp_path):
    raw = json.loads((ROOT / "configs/calibration/litearm_real.example.json").read_text())
    original = np.asarray(raw["cameras"]["cam_head"]["camera_to_world"])
    item = raw["cameras"]["cam_head"]
    item["camera_to_world"] = (original @ OPTICAL_TO_STATION).tolist()
    item["camera_frame"] = "opencv"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(raw))
    cal = load_calibration(path)
    np.testing.assert_allclose(cal.camera("cam_head").transform, original)
    assert cal.camera("cam_head").camera_frame == "opengl"
    item["camera_frame"] = "typo"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="camera_frame"):
        load_calibration(path)


def test_operator_park_import_has_one_source_and_reimport_does_not_flip(tmp_path):
    station = load_station(PROFILE)
    document = json.loads(SOURCE.read_text())
    imported = import_park(station.calibration, document, source=str(SOURCE))
    path = tmp_path / "park.json"
    path.write_text(json.dumps(imported))
    cal = load_calibration(path, arms=("right",), cameras=tuple(station.cameras))
    cam = cal.camera("cam_right_wrist")
    optical = np.asarray(document["transforms"]["T_flange_cam"]["matrix"])
    flange_camera = cal.arm("right").flange_to_tcp @ cam.transform
    np.testing.assert_allclose(flange_camera, optical @ OPTICAL_TO_STATION, atol=1e-12)
    assert cam.intrinsic[0, 0] == pytest.approx(910.37 * 480 / 720)
    assert cam.intrinsic[0, 2] == pytest.approx(647.85 * 480 / 720 - (1280 * 480 / 720 - 640) / 2)
    assert not cal.executable
    assert imported["cameras"]["cam_right_wrist"]["handeye_source"]["method"] == "PARK"
    again = import_park(cal, document)
    np.testing.assert_allclose(again["cameras"]["cam_right_wrist"]["camera_to_tool"], cam.transform)
    kinematics = LitearmKinematics(cal)
    q = cal.arm("right").safe_home_rad
    expected_world = kinematics.flange_pose("right", q) @ optical @ OPTICAL_TO_STATION
    np.testing.assert_allclose(
        kinematics.camera_pose("cam_right_wrist", "right", q), expected_world
    )
    bad = copy.deepcopy(document)
    bad["selected_method"] = "TSAI"
    with pytest.raises(ValueError, match="PARK"):
        import_park(cal, bad)


def test_import_cli_only_writes_disabled_review_profile(tmp_path):
    config_out = tmp_path / "profile.json"
    calibration_out = tmp_path / "calibration.json"
    original = PROFILE.read_bytes()
    main(
        [
            "--station",
            str(PROFILE),
            "--handeye",
            str(SOURCE),
            "--out-station",
            str(config_out),
            "--out-calibration",
            str(calibration_out),
        ]
    )
    reviewed = load_station(config_out)
    assert reviewed.execution.mode.value == "read-only"
    assert not reviewed.calibration.executable
    assert reviewed.raw["manipulation"] == {"enable_cube_grasp": False, "enable_cube_stack": False}
    assert PROFILE.read_bytes() == original
