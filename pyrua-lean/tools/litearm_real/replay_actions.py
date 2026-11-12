#!/usr/bin/env python3
"""Replay a recorded primitive sequence (action points) on the live station.

The sequence below is the 15 motion/gripper actions recorded in
``runs_hw/single-right-park-unified/live-stack-111750/export/action_points.json``
(the green-block-on-red-block success).  It executes the same commanded
targets through the same toolkit primitives; the service turns each
``move_pose(path="line")`` into IK waypoints and runs them as ``movej_sync``
joint segments.

Usage:
  PYTHONPATH=src python tools/litearm_real/replay_actions.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

from pyrualean.hardware.config import load_station  # noqa: E402
from pyrualean.hosts.litearm_hardware import Cell, boot  # noqa: E402

DEFAULT_CFG = REPO / "configs/litearm_real.json"


def boot_toolkit(cfg_path: Path, out_dir: Path):
    cfg = load_station(str(cfg_path))
    cell = Cell(
        hardware_config=str(cfg_path),
        instruction="deterministic action replay",
        station_id=cfg.station_id,
        task="replay",
        suite="hardware",
        seed=0,
        max_episode_steps=400,
    )
    toolkit, _ = boot(
        cell,
        out_dir,
        hardware_config=str(cfg_path),
        instruction="deterministic action replay",
    )
    toolkit.max_device_actions = 400
    return cfg, toolkit


def call(toolkit, name: str, args: dict) -> dict:
    out = toolkit.execute_tool(name, args)
    if "error" in out:
        return {"error": out["error"], "detail": out.get("detail")}
    return out.get("log", {}).get("result", out)


def save_frames(toolkit, out_dir: Path, tag: str) -> None:
    obs = toolkit.current_observation(refresh=False)
    img = out_dir / "images"
    img.mkdir(parents=True, exist_ok=True)
    for name, frame in toolkit.frames(obs).items():
        if frame.rgb is not None:
            Image.fromarray(frame.rgb).save(img / f"{tag}_{name}.png")


ACTIONS = [
    ("move", {"xyz": [0.055, 0.410, 0.240], "pitch": 90.0, "yaw": 20.0, "roll": 0.0, "path": "line"}),
    ("move", {"xyz": [0.054, 0.4134, 0.220], "pitch": 90.0, "yaw": 49.0, "roll": 0.0, "path": "line"}),
    ("move", {"xyz": [-0.0359, 0.4134, 0.185], "pitch": 90.0, "yaw": 49.0, "roll": 0.0, "path": "line"}),
    ("move", {"xyz": [-0.0359, 0.4134, 0.129], "pitch": 90.0, "yaw": 49.0, "roll": 0.0, "path": "line"}),
    ("grip", {"gripper": 1}),
    ("grip", {"gripper": 1}),
    ("move", {"xyz": [-0.0345, 0.4137, 0.1448], "path": "line"}),
    ("move", {"xyz": [-0.0346, 0.4134, 0.220], "path": "line"}),
    ("move", {"xyz": [-0.0585, 0.4336, 0.220], "path": "line"}),
    ("move", {"xyz": [-0.1471, 0.4339, 0.185], "pitch": 90.0, "yaw": 49.0, "roll": 0.0, "path": "line"}),
    ("place_on", {"x": -0.14707721196197707, "y": 0.43385911942952277, "support_top": 0.1421,
                  "tcp_to_bottom": 0.004792389080345141, "clearance": 0.002, "lift_m": 0.08,
                  "retreat_xy": [0.09, 0.0], "path": "line"}),
    ("grip", {"gripper": -1}),
    ("grip", {"gripper": -1, "opening": 0.38}),
    ("move", {"xyz": [-0.1454, 0.4348, 0.220], "path": "line"}),
    ("move", {"xyz": [-0.0585, 0.4336, 0.220], "path": "line"}),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "replay.log", "w")

    cfg, tk = boot_toolkit(args.cfg, args.out)
    try:
        for i, (kind, spec) in enumerate(ACTIONS, start=1):
            payload = dict(spec)
            payload.setdefault("arm", "right")
            payload.setdefault("gripper", 0) if kind == "move" else None
            record = {"index": i, "kind": kind, "args": payload}
            if kind == "move":
                result = call(tk, "move_pose", payload)
            elif kind == "grip":
                result = call(tk, "grip", payload)
            else:
                result = call(tk, "place_on", payload)
            record["result"] = result
            log.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            log.flush()
            print(json.dumps({k: record[k] for k in ("index", "kind")} | {"ok": "error" not in result},
                             ensure_ascii=False), flush=True)
            if kind == "move":
                save_frames(tk, args.out, f"{i:02d}_move")
                if result.get("error") or not result.get("reached") or result.get("blocked"):
                    print("STOP: move did not arrive at", i, flush=True)
                    return 2
            time.sleep(0.3)
        save_frames(tk, args.out, "final")
        print("replay complete", flush=True)
    finally:
        tk.close()
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
