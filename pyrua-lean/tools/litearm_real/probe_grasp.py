#!/usr/bin/env python3
"""Deterministic live probes for the LiteArm fingertip-TCP station.

Phases
------
dry    : boot, read state, run one check_move, no motion.
tcp    : empty gripper, step down at a clear table spot and save images.
grasp  : locate the red block, descend, close, wait, read twice, micro-lift,
         then place the block back and retreat.

Motion is only issued when --allow-motion is given.  The station service must
already be running with the measured-fingertip profile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pyrualean.hardware.config import load_station  # noqa: E402
from pyrualean.hosts.litearm_hardware import Cell, boot  # noqa: E402

DEFAULT_CFG = REPO / "configs/litearm_real.json"


def boot_toolkit(cfg_path: Path, out_dir: Path):
    cfg = load_station(str(cfg_path))
    cell = Cell(
        hardware_config=str(cfg_path),
        instruction="deterministic hardware probe",
        station_id=cfg.station_id,
        task="probe",
        suite="hardware",
        seed=0,
        max_episode_steps=400,
    )
    toolkit, _ = boot(
        cell,
        out_dir,
        hardware_config=str(cfg_path),
        instruction="deterministic hardware probe",
    )
    toolkit.max_device_actions = 400
    return cfg, toolkit


def state(toolkit) -> dict:
    obs = toolkit.refresh_observation(tool="probe_state")
    a = obs.arm("right")
    tcp = np.asarray(a.tcp_pose, dtype=float).reshape(4, 4)[:3, 3]
    return {
        "tcp": [round(float(v), 4) for v in tcp],
        "width_m": round(float(a.gripper_width_m), 5) if a.gripper_width_m is not None else None,
        "opening": round(float(a.gripper_opening), 5) if a.gripper_opening is not None else None,
        "force_n": round(float(a.gripper_force_n), 3) if a.gripper_force_n is not None else None,
        "enabled": bool(a.enabled),
        "fault": bool(a.fault),
        "fault_detail": a.fault_detail,
        "q": [round(float(v), 4) for v in a.q_model],
        "step": int(obs.step),
    }


def save_frames(toolkit, out_dir: Path, tag: str) -> list[str]:
    obs = toolkit.current_observation(refresh=False)
    frames = toolkit.frames(obs)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, frame in frames.items():
        if frame.rgb is None:
            continue
        path = img_dir / f"{tag}_{name}.png"
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(path)
        saved.append(str(path))
        if frame.depth is not None and name == "cam_head":
            np.save(img_dir / f"{tag}_{name}_depth.npy", np.asarray(frame.depth))
    return saved


def call(toolkit, name: str, args: dict) -> dict:
    result = toolkit.execute_tool(name, args)
    if "error" in result:
        return {"error": result["error"], "detail": result.get("detail")}
    return result.get("log", {}).get("result", {})


def check_move(toolkit, xyz, *, pitch, yaw, roll, path) -> dict:
    out = toolkit.execute_tool(
        "check_move",
        {"xyz": list(xyz), "pitch": pitch, "yaw": yaw, "roll": roll, "path": path, "arm": "right"},
    )
    if "error" in out:
        return {"feasible": False, "error": out["error"], "detail": out.get("detail")}
    # check_move returns the planner dict directly (not wrapped in log.result).
    if "log" in out:
        return out.get("log", {}).get("result", out)
    return out


def move(toolkit, xyz, *, pitch, yaw, roll, path, tag, out_dir, pause) -> dict:
    result = call(
        toolkit,
        "move_pose",
        {"xyz": list(xyz), "pitch": pitch, "yaw": yaw, "roll": roll, "path": path, "arm": "right"},
    )
    if "error" in result:
        print(json.dumps({"tag": tag, "move_error": result}, ensure_ascii=False), flush=True)
        return result
    images = save_frames(toolkit, out_dir, tag)
    summary = {
        "tag": tag,
        "target": [round(float(v), 4) for v in xyz],
        "final_xyz": result.get("final_xyz"),
        "dist_to_target_m": result.get("dist_to_target_m"),
        "reached": result.get("reached"),
        "blocked": result.get("blocked"),
        "slip": result.get("slip"),
        "note": result.get("note"),
        "ik_error": result.get("ik_error"),
        "images": images,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if pause:
        wait_for_continue(out_dir, tag)
    return result


def wait_for_continue(out_dir: Path, tag: str) -> None:
    marker = out_dir / "PAUSED"
    marker.write_text(tag)
    print(f"PAUSED {tag} (write {out_dir / 'continue'} to continue)", flush=True)
    while not (out_dir / "continue").exists():
        time.sleep(0.2)
    (out_dir / "continue").unlink()


def phase_dry(args) -> int:
    cfg, toolkit = boot_toolkit(args.cfg, args.out)
    try:
        print(json.dumps({"state": state(toolkit)}, ensure_ascii=False), flush=True)
        chk = check_move(
            toolkit,
            [args.xy[0], args.xy[1], args.z],
            pitch=90.0,
            yaw=0.0,
            roll=0.0,
            path="auto",
        )
        print(json.dumps({"check_move": chk}, ensure_ascii=False, default=str), flush=True)
    finally:
        toolkit.close()
    return 0


def phase_tcp(args) -> int:
    if not args.allow_motion:
        raise SystemExit("refusing motion without --allow-motion")
    cfg, toolkit = boot_toolkit(args.cfg, args.out)
    try:
        print(json.dumps({"state": state(toolkit)}, ensure_ascii=False), flush=True)
        pre = [args.xy[0], args.xy[1], args.pregrasp_z]
        chk = check_move(toolkit, pre, pitch=90.0, yaw=0.0, roll=0.0, path="auto")
        print(json.dumps({"check_pregrasp": chk}, ensure_ascii=False, default=str), flush=True)
        if not chk.get("feasible"):
            return 2
        res = move(
            toolkit,
            pre,
            pitch=90.0,
            yaw=0.0,
            roll=0.0,
            path="auto",
            tag="pregrasp",
            out_dir=args.out,
            pause=args.pause,
        )
        if not res.get("reached"):
            print("pregrasp not reached; stopping", flush=True)
            return 2
        for z in args.steps:
            target = [args.xy[0], args.xy[1], float(z)]
            chk = check_move(toolkit, target, pitch=90.0, yaw=0.0, roll=0.0, path="line")
            print(json.dumps({"check": target, "result": chk}, ensure_ascii=False, default=str), flush=True)
            if not chk.get("feasible"):
                print("not feasible; stopping descent", flush=True)
                break
            res = move(
                toolkit,
                target,
                pitch=90.0,
                yaw=0.0,
                roll=0.0,
                path="line",
                tag=f"z{z:.4f}",
                out_dir=args.out,
                pause=args.pause,
            )
            if res.get("error") or res.get("ik_error"):
                break
            if not res.get("reached") or res.get("blocked"):
                print("descent stopped by the arm; do not force lower", flush=True)
                break
            actual_z = float(res["final_xyz"][2])
            if actual_z - float(z) < -0.003:
                print(f"actual z {actual_z:.4f} is >3mm below the target; stopping", flush=True)
                break
        # retreat to the pregrasp height before closing the session
        move(
            toolkit,
            [args.xy[0], args.xy[1], args.pregrasp_z],
            pitch=90.0,
            yaw=0.0,
            roll=0.0,
            path="line",
            tag="retreat",
            out_dir=args.out,
            pause=False,
        )
        print(json.dumps({"final_state": state(toolkit)}, ensure_ascii=False), flush=True)
    finally:
        toolkit.close()
    return 0


def locate(toolkit, prompt: str, camera: str, min_score: float) -> dict:
    # locate returns a flat dict; move_pose/grip are wrapped in log.result.
    out = toolkit.execute_tool(
        "locate", {"text_prompt": prompt, "camera": camera, "min_score": min_score}
    )
    if "error" in out:
        return out
    if "log" in out:
        return out.get("log", {}).get("result", out)
    return out


def phase_grasp(args) -> int:
    if not args.allow_motion:
        raise SystemExit("refusing motion without --allow-motion")
    cfg, toolkit = boot_toolkit(args.cfg, args.out)
    try:
        print(json.dumps({"state": state(toolkit)}, ensure_ascii=False), flush=True)
        if args.start_lift is not None:
            now = state(toolkit)
            up = [now["tcp"][0], now["tcp"][1], float(args.start_lift)]
            chk = check_move(toolkit, up, pitch=None, yaw=None, roll=None, path="line")
            print(json.dumps({"check_start_lift": chk}, ensure_ascii=False, default=str), flush=True)
            if chk.get("feasible"):
                move(
                    toolkit,
                    up,
                    pitch=None,
                    yaw=None,
                    roll=None,
                    path="line",
                    tag="start_lift",
                    out_dir=args.out,
                    pause=False,
                )
        l1 = locate(toolkit, args.prompt, "cam_head", args.min_score)
        print(json.dumps({"locate1": l1}, ensure_ascii=False, default=str), flush=True)
        if l1.get("error") or l1.get("measurement_failed") or not l1.get("found", True):
            print("first locate failed; stopping", flush=True)
            return 2
        time.sleep(args.locate_gap_s)
        l2 = locate(toolkit, args.prompt, "cam_head", args.min_score)
        print(json.dumps({"locate2": l2}, ensure_ascii=False, default=str), flush=True)
        if l2.get("error") or l2.get("measurement_failed") or not l2.get("found", True):
            print("second locate failed; stopping", flush=True)
            return 2
        t1, t2 = float(l1["top_z"]), float(l2["top_z"])
        if abs(t1 - t2) > args.max_top_z_delta:
            print(f"top_z inconsistent ({t1} vs {t2}); stopping", flush=True)
            return 2
        # prefer the higher-quality measurement
        q1 = float((l1.get("measurement_quality") or {}).get("score", 0.0) or 0.0)
        q2 = float((l2.get("measurement_quality") or {}).get("score", 0.0) or 0.0)
        loc = l2 if q2 >= q1 else l1
        x, y = float(loc["center_xyz"][0]), float(loc["center_xyz"][1])
        top_z = float(loc["top_z"])
        yaw = float(loc["grasp_yaw"])
        # target depth
        if args.z_cmd is not None:
            z_cmd = float(args.z_cmd)
        else:
            z_cmd = max(args.z_min, min(args.z_max, args.z_touch + args.tip_clearance))
        print(
            json.dumps(
                {
                    "chosen_locate": loc,
                    "target_xy": [x, y],
                    "top_z": top_z,
                    "z_cmd": z_cmd,
                    "overlap_mm_at_cmd": round((top_z - z_cmd) * 1000, 1),
                    "z_touch": args.z_touch,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        pre = [x, y, args.pregrasp_z]
        chk = check_move(toolkit, pre, pitch=90.0, yaw=yaw, roll=0.0, path="auto")
        print(json.dumps({"check_pregrasp": chk}, ensure_ascii=False, default=str), flush=True)
        if not chk.get("feasible"):
            return 2
        res = move(
            toolkit,
            pre,
            pitch=90.0,
            yaw=yaw,
            roll=0.0,
            path="auto",
            tag="grasp_pregrasp",
            out_dir=args.out,
            pause=False,
        )
        if not res.get("reached"):
            return 2
        target = [x, y, z_cmd]
        chk = check_move(toolkit, target, pitch=90.0, yaw=yaw, roll=0.0, path="line")
        print(json.dumps({"check_descend": chk}, ensure_ascii=False, default=str), flush=True)
        if not chk.get("feasible"):
            print("descent not feasible; stopping", flush=True)
            return 2
        res = move(
            toolkit,
            target,
            pitch=90.0,
            yaw=yaw,
            roll=0.0,
            path="line",
            tag="grasp_descend",
            out_dir=args.out,
            pause=False,
        )
        if not res.get("reached"):
            print("descent not reached; stopping", flush=True)
            return 2
        actual_z = float(res["final_xyz"][2])
        print(
            json.dumps(
                {
                    "actual_descend_z": actual_z,
                    "overlap_mm_actual": round((top_z - actual_z) * 1000, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        # close
        close = call(toolkit, "grip", {"arm": "right", "gripper": 1})
        print(json.dumps({"grip_close": close}, ensure_ascii=False), flush=True)
        time.sleep(args.post_close_wait_s)
        s1 = state(toolkit)
        print(json.dumps({"post_close_state_1": s1}, ensure_ascii=False), flush=True)
        save_frames(toolkit, args.out, "post_close_1")
        time.sleep(args.post_close_gap_s)
        s2 = state(toolkit)
        print(json.dumps({"post_close_state_2": s2}, ensure_ascii=False), flush=True)
        save_frames(toolkit, args.out, "post_close_2")
        drift = None
        if s1.get("width_m") is not None and s2.get("width_m") is not None:
            drift = round((s2["width_m"] - s1["width_m"]) * 1000, 3)
        ok = (
            close.get("holding")
            and close.get("settled")
            and drift is not None
            and drift > -1.0
            and (s2.get("force_n") or 0.0) >= args.min_force_n
        )
        print(json.dumps({"post_close_drift_mm": drift, "post_close_ok": ok}, ensure_ascii=False), flush=True)
        if not ok:
            print("post-close check failed; opening and not lifting", flush=True)
            call(toolkit, "grip", {"arm": "right", "gripper": -1})
            return 3
        if args.reclose:
            # The first close reports the early width snapshot as the carry
            # baseline.  A second CLOSE keeps the existing hold thread and
            # re-reads the settled width, so the slip guard gets the real
            # baseline instead of the pre-force-ramp width.
            rc = call(toolkit, "grip", {"arm": "right", "gripper": 1})
            print(json.dumps({"reclose": rc}, ensure_ascii=False), flush=True)
            time.sleep(0.3)
            rs = state(toolkit)
            print(json.dumps({"reclose_state": rs}, ensure_ascii=False), flush=True)
            save_frames(toolkit, args.out, "reclose")
        # micro-lift
        lift_z = actual_z + args.micro_lift_m
        chk = check_move(toolkit, [x, y, lift_z], pitch=90.0, yaw=yaw, roll=0.0, path="line")
        print(json.dumps({"check_micro_lift": chk}, ensure_ascii=False, default=str), flush=True)
        if not chk.get("feasible"):
            print("micro-lift not feasible; opening", flush=True)
            call(toolkit, "grip", {"arm": "right", "gripper": -1})
            return 3
        res = move(
            toolkit,
            [x, y, lift_z],
            pitch=90.0,
            yaw=yaw,
            roll=0.0,
            path="line",
            tag="micro_lift",
            out_dir=args.out,
            pause=False,
        )
        print(json.dumps({"micro_lift_result": res}, ensure_ascii=False, default=str), flush=True)
        micro_ok = bool(res.get("reached")) and not bool(res.get("slip"))
        print(json.dumps({"micro_lift_ok": micro_ok}, ensure_ascii=False), flush=True)
        if not micro_ok:
            print("micro-lift failed/slipped; opening and stopping", flush=True)
            call(toolkit, "grip", {"arm": "right", "gripper": -1})
            return 3
        if args.complete_stack:
            # place the block back at its original x/y, then open.
            back = [x, y, actual_z]
            chk = check_move(toolkit, back, pitch=90.0, yaw=yaw, roll=0.0, path="line")
            print(json.dumps({"check_put_back": chk}, ensure_ascii=False, default=str), flush=True)
            if chk.get("feasible"):
                res = move(
                    toolkit,
                    back,
                    pitch=90.0,
                    yaw=yaw,
                    roll=0.0,
                    path="line",
                    tag="put_back_down",
                    out_dir=args.out,
                    pause=False,
                )
                print(json.dumps({"put_back_down": res}, ensure_ascii=False, default=str), flush=True)
                opened = call(toolkit, "grip", {"arm": "right", "gripper": -1})
                print(json.dumps({"put_back_open": opened}, ensure_ascii=False), flush=True)
                move(
                    toolkit,
                    [x, y, args.pregrasp_z],
                    pitch=90.0,
                    yaw=yaw,
                    roll=0.0,
                    path="line",
                    tag="retreat",
                    out_dir=args.out,
                    pause=False,
                )
            else:
                print("put-back not feasible; keeping the block held; retract only", flush=True)
        # final state
        print(json.dumps({"final_state": state(toolkit)}, ensure_ascii=False), flush=True)
    finally:
        toolkit.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dry", "tcp", "grasp"], required=True)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--pause", action="store_true", help="wait for an out/continue file after each move")
    parser.add_argument("--xy", type=float, nargs=2, default=[-0.16, 0.47])
    parser.add_argument("--z", type=float, default=0.16)
    parser.add_argument("--pregrasp-z", type=float, default=0.18)
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[0.160, 0.150, 0.142, 0.135, 0.129, 0.125, 0.122],
    )
    parser.add_argument("--prompt", default="the red block")
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--locate-gap-s", type=float, default=1.0)
    parser.add_argument("--max-top-z-delta", type=float, default=0.005)
    parser.add_argument("--z-touch", type=float, default=0.1195)
    parser.add_argument("--tip-clearance", type=float, default=0.003)
    parser.add_argument("--z-min", type=float, default=0.1195)
    parser.add_argument("--z-max", type=float, default=0.126)
    parser.add_argument("--z-cmd", type=float, default=None)
    parser.add_argument("--post-close-wait-s", type=float, default=1.0)
    parser.add_argument("--post-close-gap-s", type=float, default=0.5)
    parser.add_argument("--min-force-n", type=float, default=13.0)
    parser.add_argument("--micro-lift-m", type=float, default=0.015)
    parser.add_argument("--start-lift", type=float, default=None, help="lift to this TCP z before locate")
    parser.add_argument("--reclose", action="store_true", help="re-issue CLOSE after settling to refresh the slip baseline")
    parser.add_argument("--complete-stack", action="store_true", help="put the block back after the probe")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "dry":
        return phase_dry(args)
    if args.phase == "tcp":
        return phase_tcp(args)
    return phase_grasp(args)


if __name__ == "__main__":
    raise SystemExit(main())
