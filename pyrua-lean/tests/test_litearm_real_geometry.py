"""Independent FK/IK, joint mapping, TCP, camera and collision checks."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from pyrualean.hardware.config import ConfigError, load_calibration, load_station
from pyrualean.hardware.kinematics import (
    CollisionModel,
    LitearmKinematics,
    angles,
    axis_angle,
    homogeneous,
    orientation,
    rotation_error,
    rotvec,
    rpy_matrix,
    slerp,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "litearm_real.example.json"


def _rpy(text: str) -> np.ndarray:
    values = [float(v) for v in text.split()]
    return rpy_matrix(values)


def _urdf_chain(path: Path):
    robot = ET.parse(path).getroot()
    parent_of = {j.find("child").get("link"): j for j in robot.findall("joint")}
    chain, link = [], "ee_link"
    while link != "base_link":
        joint = parent_of[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    return list(reversed(chain))


def _independent_fk(chain, q_model, base):
    pose = base.copy()
    moving = 0
    for joint in chain:
        origin = joint.find("origin")
        pose = pose @ homogeneous(
            _rpy(origin.get("rpy", "0 0 0")), [float(v) for v in origin.get("xyz", "0 0 0").split()]
        )
        if joint.get("type") == "revolute":
            axis = [float(v) for v in joint.find("axis").get("xyz").split()]
            pose = pose @ homogeneous(axis_angle(axis, q_model[moving]), (0, 0, 0))
            moving += 1
    return pose


@pytest.fixture(scope="module")
def station():
    return load_station(CONFIG)


@pytest.fixture(scope="module")
def kin(station):
    return LitearmKinematics(station.calibration)


def test_urdf_chain_and_fk_match_an_independent_reader(station, kin):
    chain = _urdf_chain(ROOT / "src" / "pyrualean" / "hardware" / "assets" / "litearm.urdf")
    rng = np.random.default_rng(17)
    for arm in ("left", "right"):
        cal = station.calibration.arm(arm)
        lo = np.array([pair[0] for pair in cal.joint_limits_rad])
        hi = np.array([pair[1] for pair in cal.joint_limits_rad])
        for _ in range(30):
            q = rng.uniform(lo, hi)
            expected = _independent_fk(chain, q, np.asarray(cal.base_pose)) @ np.asarray(
                cal.flange_to_tcp
            )
            got = kin.fk(arm, q)
            assert np.linalg.norm(got[:3, 3] - expected[:3, 3]) < 1e-6
            assert rotation_error(got[:3, :3], expected[:3, :3]) < 1e-4


def test_zero_pose_and_tcp_are_above_the_base(station, kin):
    cal = station.calibration.arm("left")
    zero = kin.fk("left", [0.0] * 7)
    assert zero[:3, 3] == pytest.approx(
        cal.base_pose[:3, 3] + np.array([0.0, 0.0, 0.9136]), abs=1e-4
    )
    assert kin.tcp_point("left", [0.0] * 7) == pytest.approx(zero[:3, 3])


def test_joint_map_round_trips(station):
    rng = np.random.default_rng(3)
    for arm in ("left", "right"):
        cal = station.calibration.arm(arm)
        q = rng.uniform(-1.0, 1.0, 7)
        sdk = cal.sdk_from_model(q)
        back = cal.model_from_sdk(sdk)
        assert back == pytest.approx(q, abs=1e-12)
        # The example map is the identity; the test still exercises both directions.
        assert sdk == pytest.approx(q)


def test_orientation_angle_round_trip():
    rng = np.random.default_rng(5)
    for _ in range(100):
        pitch = rng.uniform(-80, 85)
        yaw = rng.uniform(-179, 179)
        roll = rng.uniform(-179, 179)
        rot = orientation(pitch, yaw, roll)
        got = angles(rot)
        rebuilt = orientation(*got)
        assert rotation_error(rot, rebuilt) < 1e-4
    down = orientation(90, 33, 0)
    pitch, yaw, roll = angles(down)
    assert pitch == pytest.approx(90, abs=1e-4)
    assert yaw == pytest.approx(33, abs=1e-3)
    assert roll == pytest.approx(0, abs=1e-3)


def test_rotation_interpolation_follows_the_shortest_arc_at_constant_speed():
    start = orientation(37, -12, 18)
    axis = np.array([1.0, 2.0, -1.0])
    axis /= np.linalg.norm(axis)
    for turn in [0.2, 2.1, np.pi - 1e-8, np.pi]:
        end = start @ axis_angle(axis, turn)
        for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
            actual = slerp(start, end, fraction)
            assert np.allclose(actual.T @ actual, np.eye(3), atol=1e-10)
            assert np.linalg.det(actual) == pytest.approx(1.0, abs=1e-10)
            assert rotation_error(start, actual) == pytest.approx(
                np.degrees(turn * fraction), abs=2e-5
            )
        assert np.linalg.norm(rotvec(start.T @ end)) == pytest.approx(turn, abs=1e-8)


def test_tool_body_table_constraint_reports_existing_model_limit():
    station = load_station(ROOT / "tests/data/stations" / "litearm_single_right.safety-test.json")
    kin = LitearmKinematics(station.calibration)
    collision = CollisionModel(station.calibration, kin)
    result = collision.tool_table_constraint("right", orientation(90, 14.4, 0))
    assert result["tcp_min_z_m"] == pytest.approx(0.1658, abs=1e-9)
    assert result["model_tip_offset_m"] == 0.046
    assert result["source"] == "calibrated_tool_body_capsule"


def test_tcp_plane_profile_allows_lower_grasp_but_preserves_table_and_joint_checks():
    from dataclasses import replace

    from pyrualean.hardware.planning import Planner

    cfg = load_station(ROOT / "tests/data/stations/litearm_single_right.table-plane.json")
    planner = Planner(cfg.calibration, cfg.execution, cfg.safety)
    q = [
        0.0185985565,
        0.3476333618,
        0.0463495255,
        -1.2415122986,
        -1.5917062759,
        0.1897840500,
        1.5497446060,
    ]
    plan = planner.plan_move("right", q, [-0.036, 0.432, 0.150], pitch=90, yaw=14.4, roll=0)
    assert not planner.validate_waypoints("right", q, plan.waypoints)
    goal = plan.waypoints[-1]
    pose = planner.kinematics.fk("right", goal)
    assert float(pose[2, 3]) == pytest.approx(0.150, abs=0.002)
    bound = planner.collision.tool_table_constraint("right", pose[:3, :3])
    assert bound["source"] == "tcp_table_plane"
    assert bound["tcp_min_z_m"] == pytest.approx(0.127)
    segments = planner.collision._link_segments("right", goal)
    assert min(a[2] for a, _b, _r, name in segments if "finger" in name) < 0.125
    assert not planner.collision.z_floor_violations("right", goal)
    strict_arm = replace(
        cfg.calibration.arm("right"),
        tool_collision=replace(
            cfg.calibration.arm("right").tool_collision, table_check_mode="geometry"
        ),
    )
    strict_cal = replace(cfg.calibration, arms={"right": strict_arm})
    strict_collision = CollisionModel(strict_cal, LitearmKinematics(strict_cal))
    assert "right_tool_body" in strict_collision.table_violations("right", goal)
    assert strict_collision.z_floor_violations("right", goal)
    # Move the plane just above this measured TCP; both checks must refuse it.
    high_cal = replace(
        cfg.calibration,
        table=replace(
            cfg.calibration.table,
            z_m=float(pose[2, 3]) + 0.001,
            z_floor_m=float(pose[2, 3]) + 0.001,
        ),
    )
    high_collision = CollisionModel(high_cal, LitearmKinematics(high_cal))
    assert "right_tcp_table_plane" in high_collision.table_violations("right", goal)
    assert high_collision.z_floor_violations("right", goal)
    invalid = np.asarray(goal).copy()
    invalid[0] = cfg.calibration.arm("right").joint_limits_rad[0][1] + 0.01
    assert any(
        "joint limits" in error for error in planner.validate_waypoints("right", q, [invalid])
    )


def test_recorded_pregrasp_descent_refines_a_continuous_line_near_wrist_limit():
    from pyrualean.hardware.planning import Planner, PlanningError

    station = load_station(ROOT / "tests/data/stations" / "litearm_single_right.safety-test.json")
    planner = Planner(station.calibration, station.execution, station.safety)
    # Saved pregrasp at step 44 of live-openai-full-103151.
    q = [
        0.018598556518554688,
        0.34763336181640625,
        0.046349525451660156,
        -1.2415122985839844,
        -1.5917062759399414,
        0.18978404998779297,
        1.5497446060180664,
    ]
    for z in [0.168, 0.170, 0.171]:
        plan = planner.plan_move(
            "right", q, [-0.036, 0.432, z], pitch=90, yaw=14.4, roll=0, path="line"
        )
        assert plan.path == "line"
        assert not planner.validate_waypoints("right", q, plan.waypoints)
        start = planner.kinematics.fk("right", q)[:3, 3]
        target = np.array([-0.036, 0.432, z])
        direction = target - start
        for way in plan.waypoints:
            point = planner.kinematics.fk("right", way)[:3, 3]
            fraction = np.dot(point - start, direction) / np.dot(direction, direction)
            assert np.linalg.norm(point - (start + fraction * direction)) < 0.003
    with pytest.raises(PlanningError, match="right_tool_body") as failure:
        planner.plan_move(
            "right", q, [-0.036, 0.432, 0.162], pitch=90, yaw=14.4, roll=0, path="auto"
        )
    assert failure.value.detail["tool_table_constraint"]["tcp_min_z_m"] == pytest.approx(0.1658)


def test_tcp_and_wrist_camera_transforms_compose(station, kin):
    q = np.array(station.calibration.arm("left").safe_home_rad)
    tcp = kin.fk("left", q)
    cam = station.calibration.camera("cam_left_wrist")
    expected = tcp @ np.asarray(cam.transform)
    got = kin.camera_pose("cam_left_wrist", "left", q)
    assert got == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        kin.camera_pose("cam_left_wrist", "right", q)
    head = kin.camera_pose("cam_head", "left", q)
    assert head == pytest.approx(np.asarray(station.calibration.camera("cam_head").transform))


def test_fingertip_tcp_retarget_preserves_cameras_and_collision_geometry(tmp_path):
    from pyrualean.hardware.calibrate import retarget_tcp

    cfg = load_station(ROOT / "tests/data/stations/litearm_single_right.table-plane.json")
    old = cfg.calibration
    raw = retarget_tcp(old, "right", 0.046)
    assert raw["executable"] is False
    assert raw["arms"]["right"]["flange_to_tcp"][2][3] == pytest.approx(0.1766)
    assert old.arm("right").flange_to_tcp[2, 3] == pytest.approx(0.1306)
    path = tmp_path / "fingertip.json"
    path.write_text(json.dumps(raw))
    new = load_calibration(path, arms=tuple(old.arms), cameras=tuple(old.cameras))
    old_kin, new_kin = LitearmKinematics(old), LitearmKinematics(new)
    old_collision, new_collision = CollisionModel(old, old_kin), CollisionModel(new, new_kin)
    shift = np.eye(4)
    shift[0, 3] = 0.046
    for q in [old.arm("right").safe_home_rad, [0.1, 0.3, -0.2, -1.1, 0.4, -0.6, 0.2]]:
        assert new_kin.fk("right", q) == pytest.approx(old_kin.fk("right", q) @ shift, abs=1e-12)
        assert new_kin.camera_pose("cam_right_wrist", "right", q) == pytest.approx(
            old_kin.camera_pose("cam_right_wrist", "right", q), abs=1e-12
        )
        old_segments = old_collision._link_segments("right", q)
        new_segments = new_collision._link_segments("right", q)
        for (a, b, radius, name), (c, d, new_radius, new_name) in zip(
            old_segments, new_segments, strict=True
        ):
            assert name == new_name and radius == new_radius
            assert a == pytest.approx(c, abs=1e-12)
            assert b == pytest.approx(d, abs=1e-12)
    measured = retarget_tcp(old, "right", 0.029, measured=True)
    assert measured["executable"] == old.executable
    assert measured["arms"]["right"]["flange_to_tcp"][2][3] == pytest.approx(0.1596)
    for offset in [0.0, float("nan"), 0.4, 0.05]:
        with pytest.raises(ConfigError):
            retarget_tcp(old, "right", offset)


def test_ik_reaches_a_feasible_pose_and_checks_limits(station, kin):
    home = station.calibration.arm("left").safe_home_rad
    target = kin.fk("left", home)
    res = kin.best_solution("left", target, [0.0] * 7, extra_seeds=[home])
    assert res.ok and not res.collides
    assert res.pos_err_m < 0.002 and res.rot_err_deg < 1.0
    q = np.asarray(res.q)
    for value, (lo, hi) in zip(q, station.calibration.arm("left").joint_limits_rad, strict=True):
        assert lo - 1e-9 <= value <= hi + 1e-9
    with pytest.raises(ValueError, match="limits"):
        kin.fk("left", [9.0] * 7)


def test_collision_model_refuses_the_table_and_accepts_home(station, kin):
    collision = CollisionModel(station.calibration, kin)
    home = station.calibration.arm("left").safe_home_rad
    assert collision.self_violations("left", home) == []
    # A wrist-down pose with the grasp point below the table is a violation.
    target = np.eye(4)
    target[:3, :3] = orientation(90, 0, 0)
    target[:3, 3] = [-0.30, 0.30, -0.05]
    res = kin.solve("left", target, home, collision_check=True)
    assert res.collides or not res.ok


def test_ik_at_wrist_limits_does_not_diverge_from_a_short_lift():
    station = load_station(ROOT / "tests/data/stations" / "litearm_single_right.safety-test.json")
    kin = LitearmKinematics(station.calibration)
    # Measured start from the failed episode; J6 is 0.00075 rad from its limit.
    q = np.array(
        [0.23220348, 0.72498798, -0.33894062, -0.75207901, -1.21099377, -1.55279636, 1.42996120]
    )
    target = kin.fk("right", q)
    target[2, 3] += 0.08
    result = kin.best_solution("right", target, q, collision_check=False)
    # The old clipped solver diverged by 0.256 m / 88 degrees. Do not change
    # the acceptance tolerances: this regression checks numerical convergence.
    assert result.pos_err_m < 0.02
    assert result.rot_err_deg < 8.0
    limits = np.asarray(station.calibration.arm("right").joint_limits_rad)
    assert np.all(np.asarray(result.q) >= limits[:, 0])
    assert np.all(np.asarray(result.q) <= limits[:, 1])

    from pyrualean.hardware.planning import Planner

    planner = Planner(station.calibration, station.execution, station.safety)
    plan = planner.plan_move("right", q, target[:3, 3], path="line")
    assert plan.path == "line"
    assert not planner.validate_waypoints("right", q, plan.waypoints)
    assert "closest one" in plan.note
    assert "half a turn" not in plan.note


def test_planner_reports_line_refusal_instead_of_an_unreachable_half_turn(station, monkeypatch):
    from pyrualean.hardware.planning import Planner, PlanningError

    planner = Planner(station.calibration, station.execution, station.safety)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PlanningError("the line passes a singular pose", detail={"path": "line"})
        raise PlanningError("out of reach", detail={"pos_err_m": 0.1, "rot_err_deg": 43})

    monkeypatch.setattr(planner, "_plan_one", refuse)
    with pytest.raises(PlanningError, match="singular pose"):
        planner.plan_move(
            "left", station.calibration.arm("left").safe_home_rad, [0.0, 0.4, 0.3], path="line"
        )
    assert len(calls) == 2


def test_hard_table_z_floor_is_enforced(tmp_path):
    raw = json.loads((ROOT / "configs" / "calibration" / "litearm_real.example.json").read_text())
    raw["table"]["z_floor_m"] = 0.7
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(raw))
    calibration = load_calibration(path)
    kin = LitearmKinematics(calibration)
    collision = CollisionModel(calibration, kin)
    home = calibration.arm("left").safe_home_rad
    below = collision.z_floor_violations("left", home)
    assert below and "hard table floor" in below[0]
    assert "hard table floor" in " ".join(collision.self_violations("left", home))
    assert collision.z_floor_violations("left", [0.0] * 7) == []


def test_limit_readback_tolerance_retains_measurement_but_targets_stay_strict():
    from pyrualean.hardware.planning import Planner

    station = load_station(ROOT / "tests/data/stations" / "litearm_single_right.safety-test.json")
    kin = LitearmKinematics(station.calibration)
    q = np.asarray(station.calibration.arm("right").safe_home_rad).copy()
    lo = station.calibration.arm("right").joint_limits_rad[5][0]
    q[5] = lo - 0.0000123
    measured = kin.fk("right", q)
    at_limit = q.copy()
    at_limit[5] = lo
    assert not np.array_equal(measured, kin.fk("right", at_limit))
    planner = Planner(station.calibration, station.execution, station.safety)
    assert not planner.validate_waypoints("right", q, [at_limit])
    errors = planner.validate_waypoints("right", at_limit, [q])
    assert any("outside the calibrated joint limits" in err for err in errors)
    q[5] = lo - 0.002
    with pytest.raises(ValueError, match="outside the calibrated limits"):
        kin.fk("right", q)


def test_table_floor_allows_bounded_penetration(tmp_path):
    raw = json.loads((ROOT / "configs" / "calibration" / "litearm_real.example.json").read_text())
    # 2 cm below the table is a deliberate penetration allowance.
    raw["table"]["z_floor_m"] = -0.02
    path = tmp_path / "penetration.json"
    path.write_text(json.dumps(raw))
    calibration = load_calibration(path)
    assert calibration.table.z_floor_m == pytest.approx(-0.02)
    # A floor more than 5 cm below the table is an implausible calibration.
    raw["table"]["z_floor_m"] = -0.06
    bad = tmp_path / "too_deep.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="penetration allowance"):
        load_calibration(bad)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_interpolation_from_limit_readback_never_commands_outside_limits(side):
    from pyrualean.hardware.planning import PlanningError, _followable

    limits = np.array([[-1.0, 1.0]] * 7)
    start = np.zeros(7)
    boundary = -1.0 if side == "lower" else 1.0
    start[5] = boundary + (-0.0000123 if side == "lower" else 0.0000123)
    unchanged_start = start.copy()
    target = start.copy()
    target[0] = 0.2  # forces subdivision while J6 stays at its boundary
    target[5] = boundary
    ways = np.asarray(_followable(start, [target], 0.05, joint_limits_rad=limits))
    assert len(ways) == 4
    assert np.all(ways >= limits[:, 0]) and np.all(ways <= limits[:, 1])
    assert ways[0, 5] == boundary
    assert np.array_equal(start, unchanged_start)
    assert np.array_equal(ways[-1], target)
    assert np.max(np.abs(np.diff(np.vstack([start, ways]), axis=0))) <= 0.05 + 1e-12
    with pytest.raises(PlanningError, match="trajectory target is outside"):
        _followable(start, [unchanged_start], 0.05, joint_limits_rad=limits)
    start[5] = boundary + (-0.002 if side == "lower" else 0.002)
    with pytest.raises(PlanningError, match="q_start is outside"):
        _followable(start, [target], 0.05, joint_limits_rad=limits)


def test_second_episode_joint_move_generates_strictly_bounded_waypoints():
    from pyrualean.hardware.planning import Planner

    station = load_station(ROOT / "tests/data/stations" / "litearm_single_right.safety-test.json")
    planner = Planner(station.calibration, station.execution, station.safety)
    # Original step 14 readback and move_pose call 19 from live-openai-full-102149.
    # Force joint interpolation to retain coverage when auto finds a valid line.
    q = [
        0.23220348358154297,
        0.7265214920043945,
        -0.3389406204223633,
        -0.7516975402832031,
        -1.210993766784668,
        -1.5535593032836914,
        1.429579734802246,
    ]
    plan = planner.plan_move(
        "right",
        q,
        [0.02589393819755676, 0.4639370391448221, 0.24],
        path="joint",
        pitch=83.0623,
        yaw=-2.6154,
        roll=-93.6603,
    )
    limits = np.asarray(station.calibration.arm("right").joint_limits_rad)
    assert plan.path == "joint"
    assert len(plan.waypoints) == 4
    assert plan.waypoints[0][5] == limits[5, 0]
    assert np.all(np.asarray(plan.waypoints) >= limits[:, 0])
    assert np.all(np.asarray(plan.waypoints) <= limits[:, 1])
    assert not planner.validate_waypoints("right", q, plan.waypoints)


def test_fingertip_floor_uses_the_lowest_finger_point(tmp_path):
    raw = json.loads((ROOT / "configs" / "calibration" / "litearm_real.example.json").read_text())
    raw["table"]["z_floor_m"] = 0.55
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(raw))
    calibration = load_calibration(path)
    kin = LitearmKinematics(calibration)
    collision = CollisionModel(calibration, kin)
    home = calibration.arm("left").safe_home_rad
    # The TCP is above the floor, but the finger points are below it: the
    # floor must be enforced on the fingertips, not on the grasp point.
    assert float(kin.fk("left", home)[2, 3]) > calibration.table.z_floor_m
    bad = collision.z_floor_violations("left", home)
    assert bad and "fingertip" in bad[0]


def test_table_surface_check_skips_the_fingers_but_the_floor_still_applies(tmp_path):
    raw = json.loads((ROOT / "configs" / "calibration" / "litearm_real.example.json").read_text())
    # LiteGrip-like tips: 4.6 cm beyond the grasp point.
    raw["arms"]["left"]["tool_collision"]["tip_offset_m"] = 0.046
    # A small table patch under the safe-home fingertips, 1 cm above the lowest
    # finger point: the modelled tips dip below the table surface there.  The
    # hard floor stays 1.9 cm below them.
    home = raw["arms"]["left"]["safe_home_rad"]
    raw["table"] = {
        "z_m": 0.53,
        "x_min_m": -0.24,
        "x_max_m": -0.19,
        "y_min_m": 0.36,
        "y_max_m": 0.41,
        "edge_margin_m": 0.0,
        "z_floor_m": 0.50,
    }
    path = tmp_path / "fingers-below-table.json"
    path.write_text(json.dumps(raw))
    calibration = load_calibration(path)
    kin = LitearmKinematics(calibration)
    collision = CollisionModel(calibration, kin)

    # The finger samples are below the table surface (this is what the old
    # table check refused), yet the table check now leaves them to the floor.
    finger_z = [
        float(point[2])
        for first, second, _radius, name in collision._link_segments("left", home)
        if "finger" in name
        for point in (first, second)
    ]
    assert min(finger_z) < calibration.table.z_m + 0.002
    assert collision.table_violations("left", home) == []
    assert collision.z_floor_violations("left", home) == []

    # Raising the hard floor above the fingertips must still refuse the pose.
    raw["table"]["z_floor_m"] = 0.55
    strict = tmp_path / "floor-above-fingers.json"
    strict.write_text(json.dumps(raw))
    strict_calibration = load_calibration(strict)
    strict_collision = CollisionModel(strict_calibration, LitearmKinematics(strict_calibration))
    bad = strict_collision.z_floor_violations("left", home)
    assert bad and "fingertip" in bad[0]


def test_line_path_is_collision_checked(station, kin):
    q0 = np.array(station.calibration.arm("left").safe_home_rad)
    pose = kin.fk("left", q0)
    target = pose.copy()
    target[2, 3] -= 0.10
    ways, why = kin.line_path("left", q0, target)
    assert why is None and ways
    assert np.linalg.norm(kin.tcp_point("left", ways[-1]) - target[:3, 3]) < 0.002
    below = pose.copy()
    below[2, 3] = -0.2
    _, why2 = kin.line_path("left", q0, below)
    assert why2 is not None
