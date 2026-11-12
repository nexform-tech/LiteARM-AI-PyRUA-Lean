#!/usr/bin/env python3
"""Read-only D435/D435i check without Intel realsense-viewer.

This tool is deliberately independent of the hardware service.  It enumerates
RealSense devices, reports the negotiated USB link, captures one colour/depth
frame from the head camera and one colour frame from the wrist camera, and
writes JSON/PNG evidence.  It never opens the arm, enables anything or moves.

Examples::

    python tools/litearm_real/cameras.py \
      --head-serial HEAD-SERIAL-EXAMPLE \
      --wrist-device /dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.3-video-index0 \
      --out runs_hw/camera-check

The head camera should be on a USB3 link (5000M in ``lsusb -t``).  The wrist
camera may be RGB-only through OpenCV on USB2.  The tool does not install
udev rules; install the librealsense rules separately if access is denied.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_rgb(path: Path, rgb: Any) -> None:
    import cv2  # noqa: PLC0415 - optional diagnostic dependency

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write {path}")


def enumerate_realsense() -> tuple[list[dict[str, Any]], Any]:
    import pyrealsense2 as rs  # noqa: PLC0415

    devices = []
    for device in rs.context().query_devices():
        info = {}
        for key in (
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.usb_type_descriptor,
            rs.camera_info.physical_port,
        ):
            try:
                info[str(key)] = device.get_info(key)
            except Exception:  # noqa: BLE001 - older firmware may omit a field
                pass
        devices.append(info)
    return devices, rs


def check_head(serial: str, out: Path, width: int, height: int, fps: int) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    devices, rs = enumerate_realsense()
    match = next((d for d in devices if d.get("camera_info.serial_number") == serial), None)
    if match is None:
        raise RuntimeError(f"RealSense serial {serial!r} not found; connected: {devices}")
    link = str(match.get("camera_info.usb_type_descriptor", ""))
    if not link.startswith("3."):
        raise RuntimeError(
            f"head camera {serial} negotiated USB {link or 'unknown'}, not USB3; "
            "check the host port and cable before using colour+depth"
        )
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("head stream returned no colour/depth frame")
        bgr = np.asanyarray(color_frame.get_data()).copy()
        rgb = bgr[..., ::-1].copy()
        depth_mm = np.asanyarray(depth_frame.get_data()).copy()
        scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_valid = depth_mm > 0
        _save_rgb(out / "head_color.png", rgb)
        import cv2  # noqa: PLC0415

        cv2.imwrite(str(out / "head_depth_mm.png"), depth_mm)
        np.save(out / "head_depth_mm.npy", depth_mm)
        return {
            "serial": serial,
            "usb_type_descriptor": link,
            "physical_port": match.get("camera_info.physical_port"),
            "firmware": match.get("camera_info.firmware_version"),
            "color_shape": list(rgb.shape),
            "depth_shape": list(depth_mm.shape),
            "depth_scale_m": scale,
            "depth_valid_ratio": float(depth_valid.mean()),
            "depth_center_m": float(depth_mm[depth_mm.shape[0] // 2, depth_mm.shape[1] // 2])
            * scale,
            "intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "model": str(intrinsics.model),
                "coeffs": list(intrinsics.coeffs),
            },
            "color_path": str(out / "head_color.png"),
            "depth_path": str(out / "head_depth_mm.png"),
        }
    finally:
        pipeline.stop()


def check_wrist(device: str, out: Path, width: int, height: int, fps: int) -> dict[str, Any]:
    import cv2  # noqa: PLC0415

    target = os.path.realpath(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open wrist camera {device!r} (resolved {target})")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frames = []
        stamps = []
        for _ in range(10):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            stamps.append(time.monotonic())
        if not frames:
            raise RuntimeError(f"wrist camera {device!r} opened but returned no frame")
        bgr = frames[len(frames) // 2]
        rgb = bgr[..., ::-1].copy()
        _save_rgb(out / "wrist_color.png", rgb)
        interval = (
            sum(b - a for a, b in zip(stamps, stamps[1:], strict=False)) / (len(stamps) - 1)
            if len(stamps) > 1
            else None
        )
        return {
            "device": device,
            "resolved_device": target,
            "frames_read": len(frames),
            "shape": list(rgb.shape),
            "fps_from_interval": None if not interval else round(1.0 / interval, 2),
            "color_path": str(out / "wrist_color.png"),
        }
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-serial", default=None, help="head D435/D435i serial number")
    parser.add_argument("--wrist-device", default=None, help="wrist V4L2 path (OpenCV)")
    parser.add_argument("--out", default="runs_hw/camera-check", help="evidence directory")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "devices": [],
        "head": None,
        "wrist": None,
        "errors": [],
    }
    try:
        devices, _ = enumerate_realsense()
        report["devices"] = devices
    except Exception as exc:  # noqa: BLE001 - reported as evidence
        report["errors"].append(f"RealSense enumeration: {type(exc).__name__}: {exc}")
        devices = []
    if args.head_serial:
        try:
            report["head"] = check_head(args.head_serial, out, args.width, args.height, args.fps)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"head: {type(exc).__name__}: {exc}")
    if args.wrist_device:
        try:
            report["wrist"] = check_wrist(args.wrist_device, out, args.width, args.height, args.fps)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"wrist: {type(exc).__name__}: {exc}")
    if not args.head_serial and not args.wrist_device:
        report["errors"].append("no camera requested; pass --head-serial and/or --wrist-device")

    _write_json(out / "camera_check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
