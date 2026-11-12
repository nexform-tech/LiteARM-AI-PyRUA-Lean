"""Create a fingertip-TCP station profile without connecting to hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyrualean.hardware.calibrate import retarget_tcp
from pyrualean.hardware.config import load_calibration, load_station


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument(
        "--offset-mm",
        type=float,
        required=True,
        help="Measured distance from current TCP to finger-end midpoint along +x",
    )
    parser.add_argument("--table-z-m", type=float, required=True)
    parser.add_argument("--table-margin-mm", type=float, default=3.0)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument(
        "--measured",
        action="store_true",
        help="Offset is a physical measurement; otherwise output is read-only",
    )
    args = parser.parse_args()
    cfg = load_station(args.config)
    raw = retarget_tcp(cfg.calibration, args.arm, args.offset_mm / 1000, measured=args.measured)
    raw["table"]["z_m"] = raw["table"]["z_floor_m"] = args.table_z_m
    raw["arms"][args.arm]["tool_collision"].update(
        table_check_mode="tcp_plane", table_margin_m=args.table_margin_mm / 1000
    )
    output = args.output_config.resolve()
    calibration_path = output.parent / "calibration" / output.name
    if output.exists() or calibration_path.exists():
        parser.error("Output exists; use a new filename to retain the previous calibration")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    try:
        load_calibration(
            calibration_path,
            arms=tuple(cfg.calibration.arms),
            cameras=tuple(cfg.calibration.cameras),
        )
    except Exception:
        calibration_path.unlink()
        raise
    station = dict(cfg.raw)
    station["calibration"] = "calibration/" + output.name
    station["execution"] = dict(station["execution"])
    station["execution"]["settle_tolerance_rad"] = 0.005
    if not args.measured:
        station["execution"]["mode"] = "read-only"
    station["results"] = dict(station["results"])
    station["results"]["directory"] = str(output.parent.parent / "runs_hw" / output.stem)
    output.write_text(json.dumps(station, indent=2, allow_nan=False) + "\n")
    checked = load_station(output)
    print(
        json.dumps(
            {
                "config": str(output),
                "calibration": str(calibration_path),
                "mode": checked.execution.mode.value,
                "device_actions": 0,
                "flange_to_tcp_xyz": [
                    raw["arms"][args.arm]["flange_to_tcp"][i][3] for i in range(3)
                ],
                "table_tcp_min_z_m": args.table_z_m + args.table_margin_mm / 1000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
