"""Known-geometry regressions for identity, occlusion and suspended objects."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hardware.config import load_station
from pyrualean.hardware.perception import FrameData, Perception, Segment
from pyrualean.hardware.target_tracking import TargetExpectation


def _station():
    return load_station(Path(__file__).resolve().parents[1] / "configs/litearm_real.example.json")


class _ColourSegmenter:
    """Like a text detector, prefer the largest red region in the supplied image."""

    def __init__(self):
        self.calls = []

    def segment(self, image, *, text_prompt=None, min_score=0.2):
        import cv2

        self.calls.append((image.shape, min_score))
        red = ((image[..., 0] > 100) & (image[..., 1] < 50)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
        if count < 2:
            return Segment(False, reason="no target in supplied image")
        label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return Segment(True, 0.9, labels == label)


def _scene(capture=1, *, center=(0.0, 0.0), top=0.10, size=(0.06, 0.04)):
    h, w = 120, 160
    k = np.array([[300.0, 0.0, w / 2], [0.0, 300.0, h / 2], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[2, 3] = 1.0  # runtime camera looks along -z
    rows, cols = np.mgrid[:h, :w]
    x = (cols - k[0, 2]) / k[0, 0] * (1 - top)
    y = -(rows - k[1, 2]) / k[1, 1] * (1 - top)
    mask = (np.abs(x - center[0]) <= size[0] / 2) & (np.abs(y - center[1]) <= size[1] / 2)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask, 0] = 220
    depth = np.ones((h, w), dtype=np.float32)
    depth[mask] = 1 - top
    return FrameData("cam_head", rgb, depth, k, pose, capture), mask


def _perception():
    station = _station()
    return Perception(station.calibration, station.sam3, segmenter=_ColourSegmenter())


def _confirmed(perception):
    frame, _ = _scene()
    first = perception.locate({"cam_head": frame}, "red block", track=True)
    assert not first["measurement_failed"] and not first["geometry_confirmed"]
    frame.captured_at_wall_ns = 2
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["geometry_confirmed"]
    return result


def _hold_state(pose, width=0.036, fresh=True):
    arm = SimpleNamespace(
        tcp_pose=pose.ravel(), gripper_width_m=width, gripper_fresh=fresh, fault=False
    )
    return SimpleNamespace(arm=lambda _: arm)


def test_repeated_snapshot_is_not_independent_confirmation():
    perception = _perception()
    frame, _ = _scene()
    for _ in range(3):
        result = perception.locate({"cam_head": frame}, "red block", track=True)
        assert not result["geometry_confirmed"]
        assert result["measurement_quality"]["confirmations"] == 1


def test_roi_excludes_larger_red_distractor_and_preserves_full_image_coordinates():
    perception = _perception()
    reference = _confirmed(perception)
    frame, _ = _scene(3)
    frame.rgb[5:35, 5:35, 0] = 220  # would win unconstrained global text segmentation
    result = perception.locate({"cam_head": frame}, "red block", min_score=0.01, track=True)
    assert result["geometry_confirmed"] and not result["measurement_failed"]
    assert result["center_xyz"] == pytest.approx(reference["center_xyz"], abs=0.001)
    assert result["centroid_rc"] == [60, 80]
    assert result["search_roi_rc"] is not None
    assert perception._segmenter.calls[-1][0][0] < frame.shape[0]
    assert perception._segmenter.calls[-1][1] >= perception.sam3.min_score
    segmented = perception.segment({"cam_head": frame}, "red block", "cam_head", track=True)
    assert segmented["centroid_rc"] == [60, 80]


def test_missing_target_does_not_switch_to_distractor_or_refresh_anchor():
    perception = _perception()
    reference = _confirmed(perception)
    frame, _ = _scene(3)
    frame.rgb[:] = 0
    frame.rgb[5:35, 5:35, 0] = 220
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and result["error_code"] == "target_unconfirmed"
    assert result.get("center_xyz") is None
    assert perception.tracker.expectation("red block").center == pytest.approx(
        reference["center_xyz"]
    )


@pytest.mark.parametrize("change", ["position", "height", "size"])
def test_geometry_change_is_rejected_without_replacing_reference(change):
    perception = _perception()
    reference = _confirmed(perception)
    settings = {
        "position": {"center": (0.016, 0)},
        "height": {"top": 0.17},
        "size": {"size": (0.09, 0.04)},
    }[change]
    frame, _ = _scene(3, **settings)
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and result["error_code"] == "target_unconfirmed"
    assert perception.tracker.expectation("red block").center == pytest.approx(
        reference["center_xyz"]
    )


def test_foreground_depth_is_removed_without_shifting_the_top_or_center():
    perception = _perception()
    reference = _confirmed(perception)
    frame, mask = _scene(3)
    samples = np.argwhere(mask)[::6]
    frame.depth[samples[:, 0], samples[:, 1]] = 0.70
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert not result["measurement_failed"]
    assert result["top_z"] == pytest.approx(0.10, abs=0.001)
    assert result["center_xyz"] == pytest.approx(reference["center_xyz"], abs=0.002)
    quality = result["measurement_quality"]
    assert quality["top_plane"]["rms_m"] < 0.002
    assert quality["foreground_rejected_ratio"] > 0


def test_partial_top_reports_height_but_cannot_authorize_a_center():
    perception = _perception()
    _confirmed(perception)
    frame, mask = _scene(3)
    rows, cols = np.nonzero(mask)
    # Leave a visible, well-supported half-top; other depth is unavailable.
    frame.depth[rows[cols < 80], cols[cols < 80]] = np.nan
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and result.get("center_xyz") is None
    assert result["partial_measurement"]["top_z"] == pytest.approx(0.10, abs=0.001)
    assert result["measurement_quality"]["height_reliable"]
    assert not result["measurement_quality"]["center_reliable"]
    assert not result["partial_measurement"]["authorizes_target"]


def test_slanted_view_accepts_measured_side_depth_without_rgb_center_projection():
    h, w = 240, 320
    k = np.array([[400.0, 0, w / 2], [0, 400.0, h / 2], [0, 0, 1]])
    position = np.array([0.0, -0.5, 0.65])
    z = position - [0, 0, 0.02]
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = position
    rows, cols = np.mgrid[:h, :w]
    rays = np.stack(
        [(cols - k[0, 2]) / k[0, 0], -(rows - k[1, 2]) / k[1, 1], -np.ones((h, w))], axis=-1
    )
    direction = rays @ pose[:3, :3].T
    low, high = np.array([-0.03, -0.02, 0]), np.array([0.03, 0.02, 0.04])
    with np.errstate(divide="ignore", invalid="ignore"):
        near, far = (low - position) / direction, (high - position) / direction
    enter = np.minimum(near, far).max(axis=-1)
    leave = np.maximum(near, far).min(axis=-1)
    mask = (leave >= enter) & (enter > 0)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask, 0] = 220
    depth = np.full((h, w), np.nan, dtype=np.float32)
    depth[mask] = enter[mask]
    frame = FrameData("cam_head", rgb, depth, k, pose, 1)
    result = _perception().locate({"cam_head": frame}, "red block")
    assert not result["measurement_failed"]
    assert result["center_xyz"] == pytest.approx([0, 0, 0.02], abs=0.003)
    assert result["top_z"] == pytest.approx(0.04, abs=0.002)
    quality = result["measurement_quality"]
    assert quality["top_plane"]["coverage"] < 0.75  # visible side is not an occluder
    assert quality["object_depth_coverage"] >= 0.75
    assert quality["center_source"] == "depth_top_plane_footprint"


def test_invalid_capture_requires_two_new_valid_captures_to_confirm_again():
    perception = _perception()
    _confirmed(perception)
    bad, _ = _scene(3, top=0.17)
    assert perception.locate({"cam_head": bad}, "red block", track=True)["measurement_failed"]
    good, _ = _scene(4)
    assert not perception.locate({"cam_head": good}, "red block", track=True)["geometry_confirmed"]
    good.captured_at_wall_ns = 5
    assert perception.locate({"cam_head": good}, "red block", track=True)["geometry_confirmed"]


def test_missing_wrist_rgb_geometry_does_not_invalidate_confirmed_head_depth():
    perception = _perception()
    _confirmed(perception)
    frame, _ = _scene(3)
    wrist = replace(frame, name="cam_right_wrist", depth=None)
    result = perception.locate(
        {"cam_right_wrist": wrist}, "red block", "cam_right_wrist", track=True
    )
    assert result["measurement_failed"]
    assert perception.tracker.tracks["red block"].confirmations == 2
    frame.captured_at_wall_ns = 4
    assert perception.locate({"cam_head": frame}, "red block", track=True)["geometry_confirmed"]


def test_majority_foreground_depth_does_not_become_a_new_support_height():
    perception = _perception()
    _confirmed(perception)
    frame, mask = _scene(3)
    frame.depth[mask] = 0.78
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and result.get("top_z") is None
    assert not result["measurement_quality"]["height_reliable"]


def test_held_height_and_centre_are_independent_of_table_gap():
    frame, _ = _scene(top=0.30)
    perception = _perception()
    result = perception.locate(
        {"cam_head": frame}, "red block", geometry_mode="held", object_height_m=0.10
    )
    assert not result["measurement_failed"]
    assert result["height_m"] == pytest.approx(0.10)
    assert result["center_xyz"][2] == pytest.approx(0.25)
    assert result["measurement_quality"]["height_reference"] == "measured_pregrasp_height"
    perception.calibration = replace(
        perception.calibration, table=replace(perception.calibration.table, z_m=0.12)
    )
    changed = perception.locate(
        {"cam_head": frame}, "red block", geometry_mode="held", object_height_m=0.10
    )
    assert changed["center_xyz"] == pytest.approx(result["center_xyz"])
    unknown = perception.locate({"cam_head": frame}, "red block", geometry_mode="held")
    assert unknown["measurement_failed"] and unknown.get("height_m") is None


def test_settled_hold_moves_search_volume_with_actual_tcp_and_release_invalidates_it():
    perception = _perception()
    reference = _confirmed(perception)
    pose = np.eye(4)
    pose[2, 3] = 0.02
    width = reference["width_m"]
    perception.tracker.sync_holds(_hold_state(pose, width), {"right": 0.4})
    pose[:3, 3] += [0.12, 0.0, 0.20]
    perception.tracker.sync_holds(_hold_state(pose, width), {"right": 0.4})
    frame, _ = _scene(3, center=(0.12, 0), top=0.30)
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert not result["measurement_failed"] and result["geometry_mode"] == "held"
    assert result["height_m"] == pytest.approx(0.10)
    assert result["center_xyz"] == pytest.approx([0.12, 0, 0.25], abs=0.002)
    assert not result["geometry_confirmed"]  # hold requires independent visual confirmation
    frame.captured_at_wall_ns = 4
    assert perception.locate({"cam_head": frame}, "red block", track=True)["geometry_confirmed"]
    perception.tracker.sync_holds(_hold_state(pose, width), {})
    assert perception.tracker.expectation("red block") is None


def test_unconfirmed_grasp_identity_and_changed_hold_feedback_fail_closed():
    perception = _perception()
    frame, _ = _scene()
    perception.locate({"cam_head": frame}, "red block", track=True)
    pose = np.eye(4)
    pose[2, 3] = 0.02
    perception.tracker.sync_holds(_hold_state(pose), {"right": 0.4})
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and "identity" in result["note"]
    unknown = perception.locate({"cam_head": frame}, "another red object", track=True)
    assert unknown["measurement_failed"]
    perception = _perception()
    reference = _confirmed(perception)
    perception.tracker.sync_holds(_hold_state(pose, reference["width_m"]), {"right": 0.4})
    perception.tracker.sync_holds(_hold_state(pose, reference["width_m"] - 0.002), {"right": 0.4})
    assert perception.locate({"cam_head": frame}, "red block", track=True)["measurement_failed"]
    # The original width cannot silently restore an unverified offset.
    perception.tracker.sync_holds(_hold_state(pose, reference["width_m"]), {"right": 0.4})
    assert perception.locate({"cam_head": frame}, "red block", track=True)["measurement_failed"]


def test_full_tcp_rotation_updates_volume_and_tilt_is_not_measured_with_table_model():
    perception = _perception()
    reference = _confirmed(perception)
    pose = np.eye(4)
    pose[2, 3] = 0.02
    perception.tracker.sync_holds(_hold_state(pose, reference["width_m"]), {"right": 0.4})
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    pose[2, 3] = 0.20
    perception.tracker.sync_holds(_hold_state(pose, reference["width_m"]), {"right": 0.4})
    expected = perception.tracker.expectation("red block")
    assert expected.center == pytest.approx([0, -0.03, 0.20], abs=0.002)
    assert not expected.upright
    frame, _ = _scene(3, center=(0, -0.03), top=0.20)
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and "tilted" in result["note"]


def test_search_box_cannot_authorize_a_mask_filling_the_entire_crop():
    perception = _perception()
    _confirmed(perception)
    frame, _ = _scene(3)
    frame.rgb[..., 0] = 220
    result = perception.locate({"cam_head": frame}, "red block", track=True)
    assert result["measurement_failed"] and "oversized" in result["note"]


def test_negative_or_nan_threshold_is_rejected():
    frame, _ = _scene()
    for score in [-0.1, float("nan"), 1.1]:
        with pytest.raises(ValueError, match="min_score"):
            _perception().locate({"cam_head": frame}, "red block", min_score=score)


def test_oriented_search_bounds_include_only_points_near_the_object():
    expected = TargetExpectation(np.array([0.0, 0.0, 0.2]), np.array([0.02, 0.04, 0.02]), np.eye(3))
    assert expected.contains(np.array([[0, 0, 0.2], [0, 0, 0.3]])).tolist() == [True, False]


def test_toolkit_enables_tracking_and_uses_measured_hold_state_automatically():
    from pyrualean.hardware.toolkit import HardwareToolkit

    toolkit = HardwareToolkit.__new__(HardwareToolkit)
    toolkit.perception = _perception()
    toolkit.held_openings = {}
    pose = np.eye(4)
    pose[2, 3] = 0.02
    observation = _hold_state(pose)
    frame, _ = _scene()
    toolkit.current_observation = lambda **kwargs: observation
    toolkit.frames = lambda obs: {"cam_head": frame}
    kwargs = {"text_prompt": "red block", "camera": "cam_head"}
    assert not toolkit._locate(kwargs)["geometry_confirmed"]
    frame.captured_at_wall_ns = 2
    assert toolkit._locate(kwargs)["geometry_confirmed"]
    toolkit.held_openings = {"right": 0.4}
    toolkit.sync_target_holds(observation)
    pose[:3, 3] += [0.12, 0, 0.20]
    frame, _ = _scene(3, center=(0.12, 0), top=0.30)
    result = toolkit._locate(kwargs)
    assert result["geometry_mode"] == "held" and not result["measurement_failed"]
    assert result["center_xyz"] == pytest.approx([0.12, 0, 0.25], abs=0.002)
