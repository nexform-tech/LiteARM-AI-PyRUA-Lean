"""Replay saved camera artifacts through perception without connecting devices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pyrualean.hardware.config import load_station
from pyrualean.hardware.kinematics import LitearmKinematics
from pyrualean.hardware.perception import Perception, frame_from_observation
from pyrualean.hardware.state import build_observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--steps", required=True, nargs="+", type=int)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--camera", default="cam_head")
    parser.add_argument(
        "--hold-step", type=int, help="saved settled-close snapshot to bind before later steps"
    )
    parser.add_argument("--held-arm", default="right", choices=["left", "right"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing replay evidence")
    config = load_station(args.config)
    perception = Perception(config.calibration, config.sam3)
    source = Path(__file__).resolve().parents[1] / "src/pyrualean/hardware"
    report = {
        "episode": str(args.episode.resolve()),
        "config_sha256": config.sha256,
        "device_actions": 0,
        "perception_source_sha256": hashlib.sha256(
            (source / "perception.py").read_bytes() + (source / "target_tracking.py").read_bytes()
        ).hexdigest(),
        "method": "saved RGB/depth through configured SAM3; no device connection",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    held = {}
    kinematics = LitearmKinematics(config.calibration) if args.hold_step is not None else None

    def observation(payload):
        return build_observation(
            config.calibration,
            kinematics,
            payload,
            task="saved target perception replay",
            instruction=args.prompt,
            execution_mode=config.execution.mode.value,
            max_age_s=None,
            now_wall_ns=payload["captured_at_wall_ns"],
            active_arms=config.execution.active_arms,
        )

    for step in args.steps:
        folder = args.episode / "steps" / f"{step:06d}"
        payload = json.loads((folder / "snapshot.json").read_text())
        if args.hold_step is not None and step >= args.hold_step:
            if not held:
                closing = json.loads(
                    (args.episode / "steps" / f"{args.hold_step:06d}" / "snapshot.json").read_text()
                )
                closing_obs = observation(closing)
                held[args.held_arm] = closing_obs.arm(args.held_arm).gripper_opening
                perception.tracker.sync_holds(closing_obs, held)
            perception.tracker.sync_holds(observation(payload), held)
        cameras = payload["reading"]["cameras"]
        metadata = cameras[args.camera]
        names = [metadata["color_artifact"], metadata.get("depth_artifact")]
        artifacts = {name: (folder / "artifacts" / name).read_bytes() for name in names if name}
        frame = frame_from_observation(
            SimpleNamespace(camera_state=cameras), artifacts, args.camera
        )
        result = perception.locate({args.camera: frame}, args.prompt, track=True)
        report["results"].append({"step": step, "result": result})
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "step": step,
                    "failed": result["measurement_failed"],
                    "confirmed": result["geometry_confirmed"],
                    "center_xyz": result.get("center_xyz"),
                    "top_z": result.get("top_z"),
                    "partial": result.get("partial_measurement"),
                    "note": result.get("note"),
                }
            ),
            flush=True,
        )
    print("report:", args.output.resolve())


if __name__ == "__main__":
    main()
