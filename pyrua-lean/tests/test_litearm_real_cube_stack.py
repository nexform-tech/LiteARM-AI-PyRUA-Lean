"""Frozen stacking targets, bounded transport and verification failures."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hardware import cube_stack
from pyrualean.hardware.kinematics import orientation
from pyrualean.litearm_real import LitearmRealRobot, RealCubeStack
from pyrualean.prompt import api_reference


def _sight(xy, *, top=0.02, height=0.02):
    return {
        "found": True,
        "measurement_failed": False,
        "source": "depth",
        "center_xyz": [*xy, top - height / 2],
        "top_z": top,
        "height_m": height,
        "length_m": 0.02,
        "width_m": 0.02,
        "grasp_yaw": 0.0,
    }


class _Station:
    def __init__(
        self,
        monkeypatch,
        *,
        measurement_failure=False,
        support_changed=False,
        alignment_error=0,
        move_failed=False,
        release_failed=False,
    ):
        self.config = SimpleNamespace(
            raw={"manipulation": {"enable_cube_stack": True, "enable_cube_grasp": True}},
            execution=SimpleNamespace(active_arms=["right"]),
            arms={"right": SimpleNamespace(gripper=True)},
            calibration_sha256="frozen-calibration",
        )
        self.calibration = SimpleNamespace(
            table=SimpleNamespace(z_m=0, x_min_m=-1, x_max_m=1, y_min_m=0, y_max_m=1)
        )
        self.calls = []
        self.device_steps = 0
        self._closed = False
        self.held_openings = {}
        self.pose = np.eye(4)
        self.pose[:3, :3] = orientation(90, 0, 0)
        self.pose[:3, 3] = [-0.1, 0.4, 0.021]
        self.kinematics = SimpleNamespace(fk=lambda a, q: q)
        self.measurement_failure = measurement_failure
        self.support_changed = support_changed
        self.alignment_error = alignment_error
        self.move_failed = move_failed
        self.release_failed = release_failed
        monkeypatch.setattr(cube_stack, "grasp_cube", self.grasp)
        monkeypatch.setattr(cube_stack, "move_pose", self.move)
        monkeypatch.setattr(cube_stack, "place_on", self.place)
        monkeypatch.setattr(cube_stack, "_hold_sample", self.hold)

    def _locate(self, kwargs):
        prompt = kwargs["text_prompt"]
        self.calls.append(("locate", prompt))
        after_grasp = "right" in self.held_openings
        if self.measurement_failure:
            return {
                "found": True,
                "measurement_failed": True,
                "table_projection": {"center_xy": [0.1, 0.4]},
            }
        if prompt == "support":
            xy = [0.1, 0.4]
            if after_grasp and self.support_changed:
                xy[0] += 0.01
            return _sight(xy)
        if after_grasp:
            xy = self.pose[:2, 3] + [-0.001, 0]
            xy[0] += self.alignment_error
            return _sight(xy, top=self.pose[2, 3] - 0.006 + 0.02)
        return _sight([-0.1, 0.4])

    def refresh_observation(self, **kwargs):
        arm = SimpleNamespace(tcp_pose=self.pose)
        return SimpleNamespace(arm=lambda _: arm)

    def grasp(self, toolkit, *args, **kw):
        assert self.calls == [("locate", "source"), ("locate", "support")] * 2
        self.calls.append(("grasp",))
        self.device_steps += 5
        self.held_openings["right"] = 0.2
        # Deliberate 1 mm grasp-centre difference tests held-offset compensation.
        self.pose[:3, 3] = [-0.099, 0.4, 0.021]
        return {
            "grasped": True,
            "grasp_xyz": [-0.099, 0.4, 0.006],
            "tcp_to_bottom": 0.006,
            "baseline_width_m": 0.02,
        }

    def move(self, toolkit, xyz, *, arm, path, **kwargs):
        assert path == "line" and kwargs["_half_turn"] is False
        pose = self.pose.copy()
        pose[:3, 3] = xyz
        kwargs["_validate_plan"](SimpleNamespace(waypoints=[pose]), self.refresh_observation())
        self.calls.append(("move", list(xyz)))
        self.device_steps += 1
        self.pose = pose
        return {"reached": not self.move_failed, "blocked": False, "slip": False}

    def hold(self, toolkit, arm):
        return self.refresh_observation(), {"valid": True, "width_m": 0.02}

    def place(self, toolkit, x, y, top, offset, **kw):
        self.calls.append(("place", x, y, top, offset))
        self.device_steps += 4
        return {
            "placed": not self.release_failed,
            "released": not self.release_failed,
            "safe_to_home": not self.release_failed,
            "note": "open did not settle" if self.release_failed else None,
        }


def test_support_is_measured_before_grasp_and_target_is_frozen(monkeypatch):
    station = _Station(monkeypatch)
    r = cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)
    assert r["completed"] and r["aligned"] and r["placed"]
    assert r["support_xyz"][:2] == [0.1, 0.4]
    np.testing.assert_allclose(r["held_offset_xy"], [-0.001, 0])
    np.testing.assert_allclose(r["landing_tcp_xy"], [0.101, 0.4])
    assert r["carry_z"] == pytest.approx(0.056)
    moves = [c for c in station.calls if c[0] == "move"]
    assert len(moves) == 2  # vertical lift and one horizontal carry only
    assert moves[0][1][:2] == [-0.099, 0.4]
    assert moves[1][1][2] == moves[0][1][2]
    assert len([c for c in station.calls if c[0] == "locate"]) == 6
    assert station.calls[-1][0] == "place"
    assert r["steps_used"] == station.device_steps == 11


@pytest.mark.parametrize(
    "settings,stage",
    [
        ({"measurement_failure": True}, "measure"),
        ({"support_changed": True}, "alignment"),
        ({"alignment_error": 0.01}, "alignment"),
        ({"move_failed": True}, "clearance_lift"),
        ({"release_failed": True}, "place"),
    ],
)
def test_failures_do_not_search_for_new_targets(monkeypatch, settings, stage):
    station = _Station(monkeypatch, **settings)
    r = cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)
    assert not r["completed"] and r["failure_stage"] == stage
    if stage == "measure":
        assert station.device_steps == 0
    if stage == "alignment":
        assert not any(c[0] == "place" for c in station.calls)
        assert len([c for c in station.calls if c[0] == "move"]) == 2
        assert r["landing_tcp_xy"] == pytest.approx([0.101, 0.4])


def test_disabled_stack_refuses_before_any_measurement_or_device_action():
    toolkit = SimpleNamespace(config=SimpleNamespace(raw={}))
    with pytest.raises(ValueError, match="disabled"):
        cube_stack.stack_cube(toolkit, "source", "support", pad_height_m=0.04)


@pytest.mark.parametrize("component", ["height_reliable", "center_reliable", "footprint_reliable"])
def test_unreliable_geometry_component_refuses_before_motion(monkeypatch, component):
    station = _Station(monkeypatch)
    original = station._locate

    def unreliable(kwargs):
        result = original(kwargs)
        result["measurement_quality"] = {component: False}
        return result

    station._locate = unreliable
    result = cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)
    assert not result["completed"] and result["failure_stage"] == "measure"
    assert station.device_steps == 0


def test_inconsistent_pregrasp_measurements_issue_no_motion(monkeypatch):
    station = _Station(monkeypatch)
    locate = station._locate

    def changed(kwargs):
        result = locate(kwargs)
        if len(station.calls) == 4:
            result["center_xyz"][0] += 0.01
        return result

    station._locate = changed
    result = cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)
    assert result["failure_stage"] == "measure"
    assert station.device_steps == 0


def test_stack_requires_grasp_enable_before_reading_scene():
    station = SimpleNamespace(
        config=SimpleNamespace(raw={"manipulation": {"enable_cube_stack": True}})
    )
    with pytest.raises(ValueError, match="both"):
        cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)


def test_measured_outside_object_has_bounds_error_and_issues_no_motion(monkeypatch):
    station = _Station(monkeypatch)
    station.calibration.table.y_min_m = 0.43
    result = cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)
    assert result["error_code"] == "outside_calibrated_table"
    assert result["failure_stage"] == "measure"
    assert "not an IK" in result["note"]
    assert station.device_steps == 0
    assert result["stages"][-1]["stage"] == "table_bounds"


@pytest.mark.parametrize("outside", [False, True])
def test_stack_api_is_stateful_and_returns_stage_evidence(monkeypatch, outside):
    station = _Station(monkeypatch)
    if outside:
        station.calibration.table.y_min_m = 0.43
    payload = cube_stack.stack_cube(station, "source", "support", pad_height_m=0.04)

    class Backend:
        def execute_tool(self, name, kwargs):
            assert name == "stack_cube"
            return {"step": 1, "state": {}, "log": {"tool": name, "result": payload}}

        def env_steps(self):
            return payload["steps_used"]

        def solved(self):
            return False

    r = LitearmRealRobot(Backend()).stack_cube("source", "support", pad_height_m=0.04)
    assert isinstance(r, RealCubeStack) and r.completed is not outside
    assert r.error_code == ("outside_calibrated_table" if outside else None)
    assert "stack_cube" in LitearmRealRobot.STATEFUL_TOOLS
    assert "stack_cube" in api_reference(LitearmRealRobot, primitives="no-vla")
