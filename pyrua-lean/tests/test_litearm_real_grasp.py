"""Cube grasp geometry, feedback failures, and the gated offline RPC path."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hardware import grasp
from pyrualean.hardware.config import load_station
from pyrualean.hardware.daemon import HardwareDaemon
from pyrualean.hardware.kinematics import orientation
from pyrualean.hardware.rpc import HardwareClient
from pyrualean.hardware.toolkit import HardwareBackend, HardwareToolkit
from pyrualean.litearm_real import LitearmRealRobot, RealGrasp
from pyrualean.prompt import api_reference

ROOT = Path(__file__).resolve().parents[1]
ARGS = dict(
    center_xy=[-0.08, 0.48],
    top_z=0.1365,
    support_top=0.1165,
    side_m=0.020,
    yaw=1.5,
    pad_height_m=0.040,
)


def _pose(xyz, yaw=1.5):
    pose = np.eye(4)
    pose[:3, :3] = orientation(90, yaw, 0)
    pose[:3, 3] = xyz
    return pose


def _config(tmp_path, *, enabled=True, stack_enabled=False):
    raw = json.loads((ROOT / "configs/litearm_real.offline.example.json").read_text())
    cal = json.loads(
        (ROOT / "tests/data/stations/calibration/litearm_single_right.measured-fingertip.json").read_text()
    )
    # These are test-only copies, using fake devices. No station profile is edited.
    cal["table"]["y_min_m"] = 0.30
    cal["arms"]["right"]["gripper"]["force_n"] = 12.0  # FakeStation nominal hold
    cal["cameras"] = {
        k: v for k, v in cal["cameras"].items() if k in ("cam_head", "cam_right_wrist")
    }
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(json.dumps(cal))
    raw["station_id"] = cal["station_id"]
    raw["calibration"] = str(cal_path)
    raw["execution"]["active_arms"] = ["right"]
    raw["devices"] = {"right": raw["devices"]["right"]}
    raw["cameras"] = {k: v for k, v in raw["cameras"].items() if k in cal["cameras"]}
    raw["snapshot"]["camera_joint_sync_s"] = 0.20
    raw["fake"].update(time_scale=0.001, object_width_m=0.020)
    if enabled:
        raw["manipulation"] = {"enable_cube_grasp": True}
        if stack_enabled:
            raw["manipulation"]["enable_cube_stack"] = True
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        raw["service"]["port"] = sock.getsockname()[1]
    path = tmp_path / "station.json"
    path.write_text(json.dumps(raw))
    return load_station(path)


def test_geometry_uses_local_support_and_more_than_half_height():
    g = grasp.cube_geometry(**ARGS)
    assert g["target_xyz"][2] == pytest.approx(0.1225)
    check = grasp._pose_check(_pose(g["target_xyz"]), g, 0.032, engage=True)
    assert check["valid"]
    assert check["overlap_fraction"] == pytest.approx(0.70)
    shifted = {**ARGS, "top_z": 0.5365, "support_top": 0.5165}
    assert grasp.cube_geometry(**shifted)["target_xyz"][2] == pytest.approx(0.5225)


@pytest.mark.parametrize(
    "change",
    [
        {"overlap_fraction": 0.5},
        {"overlap_fraction": 1.1},
        {"pad_height_m": 0.008},
        {"top_z": 0.1205},
        {"center_xy": [0, float("nan")]},
        {"clearance_m": 0.001},
        {"lift_m": 0.1},
        {"side_m": 0},
        {"yaw": float("inf")},
    ],
)
def test_invalid_or_conflicting_geometry_is_refused(change):
    with pytest.raises(ValueError):
        grasp.cube_geometry(**(ARGS | change))


def test_face_parallelism_and_measured_engagement_are_required():
    g = grasp.cube_geometry(**ARGS)
    assert grasp._pose_check(_pose(g["target_xyz"], 181.5), g, 0.032, engage=True)["valid"]
    assert not grasp._pose_check(_pose(g["target_xyz"], 46.5), g, 0.032, engage=True)["valid"]
    assert not grasp._pose_check(_pose([-0.075, 0.48, 0.1225]), g, 0.032, engage=True)["valid"]
    assert not grasp._pose_check(_pose([-0.08, 0.48, 0.13]), g, 0.032, engage=True)["valid"]


class _ScriptedToolkit:
    def __init__(
        self,
        config,
        monkeypatch,
        *,
        widths=(0.022, 0.020, 0.020, 0.020),
        forces=None,
        lift_slip=False,
        descent_error=0.0,
        open_settled=True,
    ):
        self.config = config
        self.calibration = config.calibration
        self._closed = False
        self.device_steps = 0
        self.held_openings = {}
        self.pose = _pose([-0.08, 0.48, 0.1965])
        self.width = 0.050
        self.command = "open"
        self.calls = []
        self.widths = widths
        self.forces = forces or [12.0] * len(widths)
        self.sample_index = 0
        self.clock = 0.0
        self.kinematics = SimpleNamespace(fk=lambda arm, q: q)
        self.planner = SimpleNamespace(
            collision=SimpleNamespace(tool_table_constraint=lambda a, r: {"tcp_min_z_m": 0.1195}),
            plan_move=lambda a, q, xyz, **kw: SimpleNamespace(waypoints=[_pose(xyz)]),
        )
        monkeypatch.setattr(grasp.time, "monotonic", lambda: self.clock)
        monkeypatch.setattr(grasp.time, "sleep", lambda s: setattr(self, "clock", self.clock + s))
        monkeypatch.setattr(grasp, "grip", self.grip)
        monkeypatch.setattr(grasp, "move_pose", self.move)
        self.lift_slip = lift_slip
        self.descent_error = descent_error
        self.open_settled = open_settled
        self.baseline_at_lift = None

    def refresh_observation(self, **kw):
        i = min(self.sample_index, len(self.widths) - 1)
        if self.command == "closed":
            self.width = self.widths[i]
            self.sample_index += 1
        cal = self.calibration.arm("right").gripper
        arm = SimpleNamespace(
            tcp_pose=self.pose,
            q_model=self.pose,
            gripper_width_m=self.width,
            gripper_opening=cal.width_to_opening(self.width),
            gripper_force_n=self.forces[i],
            gripper_enabled=True,
            gripper_fresh=True,
            gripper_command=self.command,
            enabled=True,
            fault=False,
        )
        return SimpleNamespace(
            arm=lambda _: arm, arms=["right"], fault=False, step=self.sample_index
        )

    def current_observation(self, **kw):
        return self.refresh_observation()

    def grip(self, toolkit, arm, gripper, opening=None):
        self.device_steps += 1
        self.calls.append(("grip", gripper))
        cal = self.calibration.arm(arm).gripper
        self.command = "closed" if gripper == 1 else "open"
        self.width = self.widths[0] if gripper == 1 else cal.opening_to_width(opening or 1)
        if gripper == 1:
            self.held_openings[arm] = cal.width_to_opening(self.width)
        else:
            self.held_openings.pop(arm, None)
        return {
            "holding": gripper == 1,
            "settled": self.open_settled,
            "opening": cal.width_to_opening(self.width),
            "width_m": self.width,
        }

    def move(self, toolkit, xyz, arm, path, **kw):
        lifting = self.command == "closed"
        before = self.refresh_observation()
        pose = _pose(xyz)
        kw["_validate_plan"](SimpleNamespace(waypoints=[pose]), before)
        self.calls.append(("move", path, list(xyz)))
        self.device_steps += 1
        self.pose = pose
        if path == "line" and not lifting:
            self.pose[2, 3] += self.descent_error
        if lifting:
            self.baseline_at_lift = self.held_openings[arm]
        return {
            "reached": not (lifting and self.lift_slip),
            "blocked": False,
            "slip": lifting and self.lift_slip,
        }


def test_baseline_is_taken_after_force_ramp_not_at_close(tmp_path, monkeypatch):
    toolkit = _ScriptedToolkit(_config(tmp_path), monkeypatch)
    r = grasp.grasp_cube(toolkit, **ARGS)
    assert r["grasped"] and r["lift_reached"] and r["holding"]
    assert r["baseline_width_m"] == pytest.approx(0.020)
    assert toolkit.baseline_at_lift == pytest.approx(
        toolkit.calibration.arm("right").gripper.width_to_opening(0.020)
    )
    assert r["steps_used"] == 5
    assert toolkit.clock >= 1.0
    assert toolkit.calls[-1][1] == "line"
    assert not any(c[0] == "grip" and c[1] == -1 for c in toolkit.calls[2:])


@pytest.mark.parametrize(
    "settings,stage",
    [
        ({"widths": (0.020, 0.004)}, "settle"),
        ({"widths": (0.020, 0.021) * 30}, "settle"),
        ({"forces": [1.0] * 4}, "settle"),
        ({"descent_error": 0.004}, "descent"),
        ({"open_settled": False}, "open"),
        ({"lift_slip": True}, "lift"),
    ],
)
def test_failed_stage_never_reports_grasp_success(tmp_path, monkeypatch, settings, stage):
    toolkit = _ScriptedToolkit(_config(tmp_path), monkeypatch, **settings)
    r = grasp.grasp_cube(toolkit, **ARGS)
    assert not r["grasped"] and r["failure_stage"] == stage
    if stage == "settle":
        assert not r["lift_reached"] and toolkit.calls[-1] == ("grip", -1)
    elif stage in ("descent", "open"):
        assert ("grip", 1) not in toolkit.calls
    elif stage == "lift":
        assert toolkit.calls[-1][0] == "move"  # no opening a potentially held cube in the air


def test_default_disabled_has_no_device_or_snapshot_side_effect(tmp_path, monkeypatch):
    config = _config(tmp_path, enabled=False)
    toolkit = SimpleNamespace(config=config)
    with pytest.raises(ValueError, match="disabled"):
        grasp.grasp_cube(toolkit, **ARGS)
    # Even a direct toolkit dispatch cannot bypass the opt-in.
    fake_toolkit = HardwareToolkit.__new__(HardwareToolkit)
    fake_toolkit.config = config
    result = fake_toolkit.execute_tool("grasp_cube", ARGS)
    assert "disabled" in result["error"]


def test_existing_hardware_hold_is_not_opened_even_without_local_baseline(tmp_path, monkeypatch):
    toolkit = _ScriptedToolkit(_config(tmp_path), monkeypatch)
    toolkit.command = "closed"
    r = grasp.grasp_cube(toolkit, **ARGS)
    assert not r["grasped"] and r["failure_stage"] == "geometry"
    assert toolkit.calls == []


def test_cancel_during_settle_never_establishes_baseline_or_lifts(tmp_path, monkeypatch):
    toolkit = _ScriptedToolkit(_config(tmp_path), monkeypatch)
    original_sample = grasp._hold_sample

    def cancelled(t, arm):
        t._closed = True
        return original_sample(t, arm)

    monkeypatch.setattr(grasp, "_hold_sample", cancelled)
    r = grasp.grasp_cube(toolkit, **ARGS)
    assert not r["grasped"] and r["failure_stage"] == "settle"
    assert r["baseline_opening"] is None
    assert toolkit.calls[-1] == ("grip", 1)


def test_stale_feedback_refuses_baseline_and_lift(tmp_path, monkeypatch):
    toolkit = _ScriptedToolkit(_config(tmp_path), monkeypatch)
    original_refresh = toolkit.refresh_observation

    def stale(**kw):
        obs = original_refresh(**kw)
        if kw.get("tool") == "grasp_cube_hold":
            obs.arm("right").gripper_fresh = False
        return obs

    toolkit.refresh_observation = stale
    r = grasp.grasp_cube(toolkit, **ARGS)
    assert not r["grasped"] and r["failure_stage"] == "settle"
    assert r["baseline_opening"] is None and toolkit.calls[-1] == ("grip", -1)


@pytest.mark.parametrize(
    "change",
    [
        {"center_xy": [-0.08, 0.305]},
        {"side_m": 0.080},
    ],
)
def test_footprint_and_stroke_refusals_precede_motion(tmp_path, monkeypatch, change):
    toolkit = _ScriptedToolkit(_config(tmp_path), monkeypatch)
    r = grasp.grasp_cube(toolkit, **(ARGS | change))
    assert not r["grasped"] and r["failure_stage"] == "geometry" and toolkit.calls == []


def test_cube_grasp_public_api_over_offline_fake_rpc(tmp_path):
    config = _config(tmp_path)
    assert config.execution.mode.value == "offline-fake"
    daemon = HardwareDaemon(config)
    client = HardwareClient(config)
    toolkit = None
    try:
        daemon.start_service()
        daemon.listen("127.0.0.1", config.service.port)
        client.connect()
        client.open_session(client_id="cube-grasp-offline-test")
        toolkit = HardwareToolkit(
            config, client, task="grasp", instruction="offline cube grasp", out_dir=tmp_path / "run"
        )
        toolkit.refresh_observation(tool="boot")
        robot = LitearmRealRobot(HardwareBackend(toolkit))
        r = robot.grasp_cube(**(ARGS | {"center_xy": [-0.05, 0.45], "yaw": 0.0}))
        assert isinstance(r, RealGrasp)
        assert r.grasped, r
        assert r.overlap_fraction > 0.5
        assert r.lift_reached and not r.slip and r.holding
        assert r.baseline_width_m == pytest.approx(0.020)
        assert r.steps_used == toolkit.device_steps
        assert toolkit.solved() is False
        assert "grasp_cube" in LitearmRealRobot.STATEFUL_TOOLS
        assert "grasp_cube" in api_reference(LitearmRealRobot, primitives="no-vla")
    finally:
        if toolkit:
            toolkit.close()
        client.close(force=True)
        daemon.close()


def test_full_stack_sequence_over_offline_rpc(tmp_path, monkeypatch):
    from pyrualean.hardware import cube_stack

    config = _config(tmp_path, stack_enabled=True)
    assert config.execution.mode.value == "offline-fake"
    daemon, client, toolkit = HardwareDaemon(config), HardwareClient(config), None
    recorded_grasp = {}
    real_grasp = cube_stack.grasp_cube

    def record_grasp(*args, **kwargs):
        result = real_grasp(*args, **kwargs)
        recorded_grasp.update(result)
        return result

    monkeypatch.setattr(cube_stack, "grasp_cube", record_grasp)
    try:
        daemon.start_service()
        daemon.listen("127.0.0.1", config.service.port)
        client.connect()
        client.open_session(client_id="stack-cube-offline-test")
        toolkit = HardwareToolkit(
            config, client, task="stack", instruction="offline stacking", out_dir=tmp_path / "run"
        )
        toolkit.refresh_observation(tool="boot")

        def locate(kwargs):
            if kwargs["text_prompt"] == "support":
                xy, top = [0.02, 0.45], 0.1365
            elif recorded_grasp.get("grasped"):
                pose = np.asarray(
                    toolkit.current_observation(refresh=True).arm("right").tcp_pose
                ).reshape(4, 4)
                delta = np.array([-0.05, 0.45]) - recorded_grasp["grasp_xyz"][:2]
                xy = (pose[:2, 3] + delta).tolist()
                top = pose[2, 3] - recorded_grasp["tcp_to_bottom"] + 0.02
            else:
                xy, top = [-0.05, 0.45], 0.1365
            return {
                "found": True,
                "measurement_failed": False,
                "source": "depth",
                "center_xyz": [*xy, top - 0.01],
                "top_z": top,
                "height_m": 0.02,
                "length_m": 0.02,
                "width_m": 0.02,
                "grasp_yaw": 0,
            }

        toolkit._locate = locate
        r = LitearmRealRobot(HardwareBackend(toolkit)).stack_cube(
            "source", "support", pad_height_m=0.04
        )
        assert r.completed, r
        assert r.aligned and r.placed and r.released and r.safe_to_home
        assert r.steps_used == toolkit.device_steps
        assert toolkit.solved() is False
        assert [s["stage"] for s in r.stages] == [
            "frozen_geometry",
            "grasp",
            "clearance_lift",
            "horizontal_carry",
            "alignment",
            "place",
        ]
    finally:
        if toolkit:
            toolkit.close()
        client.close(force=True)
        daemon.close()
