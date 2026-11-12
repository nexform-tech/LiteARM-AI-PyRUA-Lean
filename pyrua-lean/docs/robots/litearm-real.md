English | [简体中文](litearm-real.zh-CN.md)

# LiteArm Real hardware path

This page documents the real-hardware adapter for a LiteArm / LiteGrip
workstation: one or two arms, with or without a gripper on each active arm,
and any configured subset of the head and wrist cameras.  It is separate from
the RoboDojo LiteArm simulation adapter:
the real path does not import or start RoboDojo, does not read its tasks,
layouts, assets or `.cache`, and does not write `eval_result`.

## Status at a glance

| Area | Status |
| --- | --- |
| Offline end-to-end: runner → `robo` → toolkit → localhost RPC → fake station → recording | Implemented and tested |
| Adapter, config/calibration validation, FK/IK/TCP, planning, collision primitives | Implemented and tested offline |
| RPC de-duplication, exclusive session, cancel, timeout, disconnect, fault state | Implemented and tested over a real localhost socket |
| RGB-D and multi-view perception | Implemented; tested with synthetic calibrated geometry; SAM3 HTTP endpoint `http://sam3-host.example:8790` verified on 2026-11-05 |
| Vendor SDK wrapper | Tested with `litearm.testing.FakeTransport` and a mock gripper in the device environment |
| Real hardware motion, emergency-stop hold/disable, firmware preemption, calibrated accuracy | **Not run here; field verification required** |
| Real camera capture (OpenCV / RealSense) | Implemented; not connected in this validation |

## Depth and motion failure recovery

The 2026-11-10 `live-openai-full-094649` episode exposed aligned-depth
contamination at the can's silhouette boundary and an IK solver that diverged
at wrist limits. Its saved step 35 now replays with a can top of 0.1801 m
(height 0.0513 m), versus the original top of 0.3821 m. These are perception
estimates from the saved images, not an independent physical measurement.

- `locate` now fits a measured top plane and uses its depth points for the
  centre and footprint. RGB silhouette projections remain separate estimates.
  `measurement_quality` reports component reliability and foreground rejection.
- `table_projection` exposes depth-free x/y and yaw on the configured table,
  even when object height fails. Check `within_table_bounds`; perspective can
  shift elevated objects outside the table projection. Do not descend using
  this projection as an object height.
- Explicit table queries return `source="calibrated_table"` and the configured
  support z. Missing-depth `back_project` raises instead of matching a generic
  object across views or inventing a table/top-plane surface.
- For empty top grasps, explicitly set `pitch=90`, `yaw=sight.grasp_yaw`,
  `roll=0`. Omitted angles preserve the measured posture. Lift first, then
  reorient in clear space with an explicit `path="auto"` if needed. `path="line"`
  refuses without automatically executing a joint swing.
- IK uses bounded steps and a decreasing residual near joint limits. Joint,
  collision and acceptance tolerances remain in force. The saved 8 cm lift
  produces a checked line plan with a reported 1.9 cm / 7 degree near miss.
  This replay executed zero device actions and does not validate a physical grasp.

Reproduce from `pyrua-lean` using the cached SAM3 masks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/miniconda3/envs/pyrualean-hw-agent/bin/python \
  -m pytest -q tests/test_litearm_real*.py
~/miniconda3/envs/pyrualean-hw-agent/bin/python tools/replay_episode_geometry.py \
  runs_hw/single-right-safety-test/live-openai-full-094649 \
  --config configs/litearm_real.json --step 35 \
  --mask-dir runs_hw/single-right-safety-test/diagnosis-20261110 \
  --prompt 'the red can' --prompt 'the white tabletop' \
  --output runs_hw/single-right-safety-test/diagnosis-20261110/replay.json
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated ROS pytest plugins from
the host's Python path. The replay tool reads saved files and never connects
to the hardware service.

Validation on 2026-11-10: 116 tests passed, 1 skipped (hardware tests plus
prompt/robot/CLI tests); Ruff passed. After restarting the service, a fresh
read-only snapshot returned can height 0.0537 m and yaw 9.3 degrees. The arm
remained enabled and fault-free; the closed session left the service READY,
with zero motion commands. Evidence is in
`runs_hw/single-right-safety-test/diagnosis-20261110/live-readonly/`.

The fresh snapshot also exposed J6 readback 0.0000123 rad below its limit.
FK and the measured trajectory start accept up to 0.001 rad of limit
readback noise while retaining the measured value. IK and commanded waypoints
remain bounded by the exact calibrated limits; readings beyond that observation
tolerance are refused. No calibration limit or table floor was changed.

The second episode, `live-openai-full-102149`, revealed that subdivision from
this measured start copied the small J6 undershoot into waypoint 0. Command
interpolation now projects each sample into the exact joint limits, while
preserving measured `q_start`. IK/home endpoints outside the limits remain
refused. The full adjusted path still passes collision, step, duration and
speed checks on the runner and is validated again by the service.

Replaying saved call 19 with explicit `path="joint"` produces four bounded
joint waypoints, with the first
J6 exactly at -1.5535470247268677 rad, and passes the fake-device RPC execution
and readback test. The saved position-only lift to z=0.29 remains unreachable;
this fix does not relax reach tolerances. Failure guidance also recognizes
explicitly copied state angles, which retain the same wrist posture. Use
`pitch=90`, `yaw=sight.grasp_yaw`, `roll=0` for an empty top-grasp approach;
reorient with `path="auto"` only where a joint swing has clearance.

Validation after this fix: 121 tests passed, 1 skipped; Ruff passed. Regression
tests cover both limit sides, subdivisions from noisy readback, invalid
endpoints, the recorded failed request and fake RPC execution. Evidence is in
`runs_hw/single-right-safety-test/diagnosis-20261110/waypoint-fix/replay.json`.
No real motion was executed. New agent processes load the waypoint-generation
fix; the service's strict waypoint validator does not need to be relaxed.

## SDK planning comparison and low grasps

The 2026-11-10 audit compares the current planner with
`~/Code/NEXTFORM/litearm-python` at commit `5643e47`. Its `arm.py`
matches the installed device SDK byte for byte. No firmware planner replacement
is required by these results. Local numerical IK remains a separate solver;
this audit does not establish numerical equivalence with firmware IK.

| Component | Result |
| --- | --- |
| Joint transport | Each waypoint uses SDK `movej_sync`, with synchronous joint interpolation |
| Flange and TCP | SDK `get_tcp()` is the flange; the calibrated offset matches `get_tool_tcp()` semantics and is not applied twice to joint commands |
| Units and orientation | SDK offset uses metres/radians and ZYX rotation; `robo` converts its degrees and 90-degree yaw convention explicitly |
| Official URDF | Origins and axes for joint1–7 and fixed `ee_link` match the vendor SE archive exactly |
| Rotation interpolation | Replaced matrix blending with shortest-arc, constant-speed SO(3) interpolation; 30 random poses match SDK transform/interpolation helpers within floating-point error |
| Line continuation | Refines coarse samples when IK fails or a joint step is too large; retains collision, limit, step, duration and speed checks |

The recorded pregrasp in `live-openai-full-103151`, step 44, used a 2 cm TCP
sample spacing and a 0.05 rad joint-step limit. A large joint change at that
spacing was reported as wrist singularity. Finer continuation finds a valid
branch. The planner now halves a failed advance, up to eight attempts, instead
of treating coarse sampling as proof of a singularity. It still refuses a line
that cannot be solved and validated. `path="line"` does not execute a joint
fallback; `path="auto"` may select a checked joint path.

For `pitch=90`, the current body capsule requires TCP z ≥ 0.1658 m over the
table: table z 0.1288 m + margin 0.002 m + body radius 0.035 m. This is a local
collision-model constraint, not an SDK limit. It has not been relaxed. The
configured fingertip offset is 0.046 m; the physical short fingers' contact
depth remains unverified. Feasibility does not prove a successful grasp.

| Recorded target `[-0.036, 0.432, z]`, yaw 14.4° | Updated replay |
| --- | --- |
| z = 0.162 m | `line` and `auto` refuse `right_tool_body` collision |
| z = 0.168 / 0.170 / 0.171 m | Both requests find validated line paths with 8–10 waypoints |
| z = 0.196 m | Feasible, but the recorded real grasp at this height was empty |

These paths use the half-turn-equivalent empty-gripper orientation. This
symmetry remains disabled while holding. SDK execution interpolates each
segment in joint space, so a checked line is a Cartesian approximation rather
than a firmware Cartesian command.

Use the read-only `robo.check_move()` to compare nearby heights before moving:

```python
checks = [
    robo.check_move([-0.036, 0.432, z], pitch=90, yaw=14.4, roll=0, path="auto")
    for z in (0.171, 0.170, 0.168)
]
```

It returns feasibility, selected path, planned position, waypoint count,
duration, notes and the body-table bound, or an error with details. It reads
fresh state and executes zero device actions. `move_to` replans before
execution. Use scene-specific coordinates and yaw for a new episode.
Read `planned_xyz` and `note`; an accepted near miss can be too high for
contact even when the requested height was appropriate.
**Do not** jump to a much higher grasp after a low collision refusal: compare
heights a few millimetres apart above the reported bound. If `line` fails,
check the same target with `auto` before repeating it. Movable objects are not
in the fixed collision model; inspect their clearance before a joint swing.

Reproduce the offline audit from `pyrua-lean`:

```bash
~/miniconda3/envs/litearm-hw-device/bin/python tools/audit_sdk_planning.py \
  --sdk-root ~/Code/NEXTFORM/litearm-python \
  --official-description ~/Code/NEXTFORM/assets/litearm/litearm_description_SE.zip \
  --config configs/litearm_real.json \
  --episode runs_hw/single-right-safety-test/live-openai-full-103151 --step 44 \
  --output runs_hw/single-right-safety-test/diagnosis-20261110/sdk-planning/audit.json
```

Validation: 126 tests passed, 1 skipped in the agent environment; 4 SDK tests
passed in the device environment. A single historical firmware flange pose
matches local FK within 3.04 µm and 0.000284°. That is one recorded pose, not
a new firmware IK or real-motion comparison. The audit executes zero device
actions. New agent processes load the planning and guidance changes; the
service's trajectory validator remains strict.
The final read-only service check returned READY, matching configuration
hashes, no active session or command, and no fault. Its evidence is saved in
`runs_hw/single-right-safety-test/diagnosis-20261110/sdk-planning/service-status.json`.

## TCP table-plane profile

For the operator's approximately 12.5 cm platform, use the separate station
`configs/litearm_real.json`. Its matching calibration is
`configs/calibration/litearm_real.json`. The profile keeps
the measured base, TCP and camera transforms and the joint/speed settings.
The table height is an operator-supplied value, not a new measurement.

| Parameter | Value and effect |
| --- | --- |
| `table.z_m` | `0.125` m, table surface in the arm-base world frame |
| `table.z_floor_m` | `0.125` m, minimum TCP height anywhere in this mode |
| `arms.right.tool_collision.table_check_mode` | `"tcp_plane"`, skips tool-body/finger geometry for table contact |
| `arms.right.tool_collision.table_margin_m` | `0.002` m, giving TCP z ≥ `0.127` m over the table; `0.0` is supported for a plane with no extra margin |
| `arms.right.tool_collision.body_radius_m` | Retains `0.035` m for self/obstacle/other-arm checks; it does not raise the TCP/table threshold in this mode |

Arm-link/table collisions, joint limits, self/other-arm collisions and speed
checks remain active. The profile checks the TCP against the table and floor,
so it does not establish finger/table clearance. `geometry` is the default
when `table_check_mode` is omitted and retains the existing capsule/finger
checks. `table_margin_m` affects the tool check; arm links retain their 2 mm
table margin.

Offline replay of step 44 admits z = 0.130 / 0.150 / 0.162 / 0.170 m and
refuses z = 0.125 / 0.126 m. No real motion was executed. This separate
profile is not active in the existing service. Start the service and agent
with the same station file to satisfy their configuration-hash handshake.

The task card now reports the configured table z and world frame, the tool
table mode and the correct translation from measured TCP transforms. It no
longer reports a constant table z = 0 or rotation entries as TCP position.

Validation after this change: 128 tests passed, 1 skipped; Ruff passed.

## Fingertip TCP and measured arrival

The CAD asset generator at `~/Code/PyRUA/tools/litearm/build_litearm_assets.py`
defines the pad centre at flange +z = 0.1306 m and the mesh finger end at
0.0641 + 0.1125 = 0.1766 m. The finger end is 46 mm beyond the pad centre.
These are CAD dimensions; the current physical short fingers remain unverified.
The suspected table stop near old TCP z = 0.1575 m would imply about 29 mm
of extension if it were table contact, but following error alone does not
establish that contact or measure a finger length.

`tools/retarget_litearm_tcp.py` composes the new flange-to-TCP transform with
a local +x translation. It also re-expresses wrist-camera extrinsics and the
tool collision envelope. `body_end_offset_m` retains the original body end
instead of extending the capsule to the new finger-end control point. FK,
SDK offset setup, state, planning and the task card use the resulting TCP.

From `pyrua-lean`, reproduce the CAD candidate with a new
output filename if the listed one already exists:

```bash
~/miniconda3/envs/pyrualean-hw-agent/bin/python tools/retarget_litearm_tcp.py \
  --config configs/litearm_real.json \
  --offset-mm 46 --table-z-m 0.1288 --table-margin-mm 3 \
  --output-config configs/litearm_real.candidate.json
```

The candidate is read-only and `executable=false`. Its fingertip TCP/table
minimum is 0.1318 m; its global TCP floor is 0.1288 m. Only use `--measured`
with a physical current-TCP-to-fingertip measurement to create a live-capable
profile. The supplied 130.6 mm pad-centre setting does not supply that missing
measurement. Neither this conversion nor the tests move hardware.

SDK `Arm.q_tol` defaults to 0.03 rad. The station requires 0.01 rad, so the
SDK could return while the service still reports an arrival failure. The
wrapper now sets SDK `q_tol` to `execution.settle_tolerance_rad` and uses the
same bound for segment results. The fingertip candidate uses 0.005 rad.
This changes when the SDK reports arrival; it does not change motor gains or
remove following-error, timeout or fault checks. A real contact remains a
stop condition. Physical improvement is unverified until a hardware retry.

With the fingertip reference, choose a tip target above the table and below
the object top; do not add the old 46 mm offset to a world target again.
`place_on` needs the actual configured-TCP-to-held-object-bottom distance.

Validation: 129 agent tests passed, 1 skipped; 5 tests passed against the
installed SDK in the device environment; Ruff passed. Camera world transforms
and the original collision envelope remain invariant after the TCP shift.
Offline fingertip targets at 0.140 and 0.133 m pass; 0.130 m is refused by
the table bound. Evidence:
`runs_hw/single-right-table-plane/diagnosis-20261110/fingertip-tcp-audit.json`.
The existing service has not been restarted with these changes.

## Architecture

Two processes:

1. **Runner / agent** (`pyrualean-hw-agent` environment).  It runs the model's
   Python through the normal `robo` API.  `LitearmRealRobot` talks to
   `HardwareToolkit`; the toolkit plans, perceives, records evidence and is the
   RPC client.
2. **Hardware service** (`litearm-hw-device` environment).  It owns the
   configured arms, LiteGrips and cameras.  It validates every trajectory again,
   executes synchronous joint segments, captures measured state, and owns the
   session, stop and fault state machine.

The service listens on `127.0.0.1` by default.  The wire protocol is
newline-delimited JSON over TCP; it can be used across machines later, but the
first version has no TLS/authentication and must be protected by a host firewall
or a secure tunnel if it is not local.  A runner session is exclusive.

```
policy code -> LitearmRealRobot -> HardwareToolkit -> HardwareClient
                                                          |
                                             localhost TCP JSON
                                                          |
HardwareDaemon -> Planner validation -> LiteArmStation -> LiteArm SDK
                                                       -> LiteGrip SDK
                                                       -> CameraManager
```

All hardware SDK and camera imports happen inside the device creation
functions.  Importing `pyrualean.hardware` or `pyrualean.litearm_real` does not
connect a device, start a thread, load a GPU model or require a hardware SDK.

## Verified SDK interfaces and versions

Local sources checked:

- `litearm-python` **2.1.0**, `~/Code/NEXTFORM/litearm-python`, commit
  `5643e47` (git describe `v1.3.0-3-g5643e47`; the package's `__version__` and
  pyproject say 2.1.0).  It is not the same version as the PyPI 1.3.0.
- `litegrip-python` **2.2.0**, local `~/Code/NEXTFORM/litegrip-python`,
  commit `6b5fdce` (git describe `v1.2.1-2-g6b5fdce`; the source tree carries
  the semantic-release placeholder `0.0.0`, while the verified install reports
  `2.2.0`).  It is not the PyPI 1.2.1.

Used LiteArm API:

- `Arm.connect()`, `get_state(refresh=True)` (`Msg.value`), `get_tcp()`,
  `get_tool_tcp()`, `set_tcp_offset(xyz, rpy)`;
- `movej_sync(q, speed)` is blocking and uses `speed ∈ [0,1]`; arrival is read
  from measured state.  `movej` is not used for segments because its per-joint
  S-curves make the intermediate Cartesian path unpredictable.
- `emergency_stop()` is a firmware command; `disable()` and `park()` are
  separate and are not treated as controlled stops.
- `move_js` is deliberately not exposed: it needs ≥10 Hz resend and has a
  0.1 s watchdog; the first version has no fixed-rate servo scheduler.
- `get_tcp()` returns the flange pose; `get_tool_tcp()` applies the host-side
  TCP offset.  The station computes TCP from measured joints and the
  calibration, and uses the firmware flange pose as a cross-check.

Used LiteGrip API:

- `LiteGrip(channel, can_id)`, `connect()`, `load_calibration(path=...)`,
  `get_state(wait=False)` with `position_mm`, `force_n`, `fresh`,
  `data_age_s`, `error_code`;
- `grasp(force_n, hold_s=0)` blocks while holding and is ended by
  `release()` from another thread; one persistent grasp thread per gripper
  implements `KEEP` without restarting the squeeze;
- `move_at_speed(target_mm, speed_mm_s)` for open/partial open; `request_release(abort=True)`
  for the emergency path; `stop()` is one zero-torque frame and is not a
  safety-rated stop.

The vendor SDKs are installed from fixed, non-editable wheels.  Build them
with `scripts/build_litearm_device_wheels.sh`; the LiteGrip repository's
semantic-release placeholder is replaced with the verified distribution
version in a temporary build copy.  The inputs are not modified.

## Installation

### Agent environment

```bash
conda env create -f environments/litearm-agent.yml
conda activate pyrualean-hw-agent
python -m pip install -e .          # in this worktree
```

The agent environment has no hardware SDK.  SAM3 runs on the Jetson Thor; the
agent only needs the stdlib HTTP client and the endpoint in the station file.
The agent tests use a scripted/HTTP segmenter and do not need `torch` or the
`sam3` package.

### SAM3 on the Jetson Thor

SAM3 is deployed as an HTTP service on the Jetson Thor.  On Thor (where the
`sam3` package/torch/checkpoint live):

```bash
python src/pyrualean/hosts/sam3_http.py \
  --host 0.0.0.0 --port 8790 \
  --checkpoint ~/PyRUA/PyRUA-Lean/integrations/xpolicylab/PyRUA_Lean/checkpoints/sam3.pt
```

Check it from the agent machine (the device network is direct, so bypass any
HTTP proxy):

```bash
curl --noproxy '*' http://sam3-host.example:8790/health
# {"ok": true, "served": 0}
```

The agent station file then points at it:

```json
"perception": {
  "sam3": {
    "mode": "http",
    "endpoint": "http://sam3-host.example:8790",
    "timeout_s": 60.0
  }
}
```

`sam3-host.example` stands for the SAM3 service host on your network; update
the field when the address changes. Connectivity was checked on 2026-11-05
with `/health` and one point-prompt segmentation over this address. The
service has no authentication in this version, so restrict it to the trusted
robot network or a tunnel. A station file with
`mode="local"` and a checkpoint still works on a machine with a local GPU, but
the default hardware example uses the Thor endpoint.

### Device environment

```bash
conda env create -f environments/litearm-device.yml
conda activate litearm-hw-device
LITEARM_SDK_SRC=/path/to/litearm-python \
LITEGRIP_SDK_SRC=~/Code/NEXTFORM/litegrip-python \
scripts/build_litearm_device_wheels.sh
python -m pip install --no-index --find-links wheelhouse \
    litearm-python==2.1.0 litegrip-python==2.2.0
python -m pip install --no-deps -e .
```

Do not use an editable install of a vendor SDK on the driving machine.

The device environment already pins `pyrealsense2` (the Python wrapper, which
bundles librealsense) and OpenCV.  The Intel `realsense-viewer` is **not** a
runtime dependency; install it only as an optional diagnostic tool and close it
before starting the service, because it takes exclusive control of the cameras.
On Linux, install the librealsense udev rules as well; the pip wheel does not
necessarily install them.

## Station configuration

Copy `configs/litearm_real.example.json` to a local, git-ignored file (for
example `configs/litearm_real.json`) and copy the example calibration to a real
calibration.  Relative paths in the station file are resolved relative to the
station file.

Key sections:

- `service`: host, port, connect/request timeouts, heartbeat period and
  timeout, maximum clock skew.  Defaults to localhost.
- `execution`: `mode` (`read-only`, `offline-fake`, `live`), optional
  `active_arms` (default `["left", "right"]`; set `["left"]` or `["right"]`
  when only one arm is physically present — the other arm's `devices` entry
  and wrist camera may then be omitted, and it is never opened or commanded),
  speed fraction,
  configured joint/TCP ceilings, segment length, settle tolerance and command
  timeout.  The SDK exposes only a speed fraction; acceleration is owned by the
  firmware and is **not** claimed to be enforced by this layer.
- `devices`: per-arm serial port (prefer the stable
  `/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_*` symlink
  over `/dev/ttyACM*`) and minimum firmware, per-gripper CAN channel,
  CAN ID, optional per-gripper calibration path, force, grasp/open speeds.  A
  `gripper` may be `null` or omitted when that active arm has no LiteGrip: the
  station then opens, reads and stops only the arm, `state()` reports
  `gripper`/`width_m` as `None`, and every OPEN/CLOSE or `place_on` request is
  refused instead of driving a made-up gripper.  If `calibration_path` is
  omitted, the SDK's shipped factory calibration is used; a `can1` gripper
  needs its own file because the factory file declares `can0`.
- `cameras`: driver (`opencv`, `realsense`, or `fake` only in offline-fake),
  device index/serial, resolution, FPS, depth flag.  The `opencv` driver also
  accepts `device` (an explicit path such as `/dev/video4`, opened with
  `cv2.CAP_V4L2`; it takes precedence over `index`) and `fourcc` (for example
  `"YUYV"`, set before the resolution/FPS negotiation).  An integer index
  cannot address every V4L2 node.  `device`/`fourcc` are rejected for
  non-opencv drivers, and each opencv camera needs an index or a device path.
  Recommended split for the current D435/D435i station: the head camera on
  USB3 uses `driver="realsense"` with its serial and `depth=true`; a wrist
  camera on USB2 uses `driver="opencv"` with a stable `device` path and
  `depth=false`.  Check the negotiated link with `lsusb -t` (5000M vs 480M);
  a USB2 link is not enough for the head colour+depth stream.  Prefer the
  RealSense serial for the head and a by-path/udev symlink for the wrist over
  `/dev/videoN`, whose number changes across reconnects.
- `perception`: SAM3 mode (`http` for the Jetson Thor, or `local`), endpoint
  and timeout, optional local checkpoint/GPU, voxel size, maximum object
  height, minimum valid-depth ratio, and `primary_camera` (default
  `"cam_head"`; must name a configured camera).  The runner's `locate`,
  `segment`, `back_project`, `image` and `camera_meta` use it when the caller
  omits `camera`, so a wrist-only station can call `locate(prompt)`.
- `snapshot`: maximum age, camera/joint synchronization limit, refresh interval,
  retained history.
- `results`: output root, device-action budget, wall-clock budget.
- `safety`: stop timeout, stationary confirmation threshold, home lift height,
  operator-reset policy.
- `fake`: offline-only deterministic settings; ignored by live mode.

`execution.mode=live` is refused unless the calibration has
`"executable": true` and a measured (non-example) quality status for the
selected arms and cameras.  The example is intentionally `executable: false`.

## Calibration

The calibration file defines the station:

- world origin at the centre of the near table edge, +x right, +y into the
  table, +z up, table `z=0`;
- each arm's full base pose, flange-to-TCP transform, joint map
  (`model_joint`, `sdk_index`, sign, zero offset), measured joint limits and
  safe home;
- measured LiteGrip closed/open widths (normalised opening is
  `(width - closed)/(open - closed)`), force, minimum holding width;
- each camera's camera-to-world (head) or camera-to-tool (wrist) transform,
  intrinsic matrix, distortion, resolution and depth flag;
- table extent and fixed obstacles;
- source, timestamp, version and quality.

All matrices are validated (shape, finite values, orthonormal rotation with
determinant +1).  Joint maps must use model joints 0..6 and SDK indices 0..6
exactly once with sign ±1.  Limits must be within the packaged URDF.  The safe
home must be inside the limits.

The calibration must describe **exactly** the station's devices: its `arms`
keys must equal `execution.active_arms`, and its `cameras` keys must equal the
configured `cameras`.  A single-arm, wrist-only station therefore uses a
trimmed calibration file with only the entries it actually has; the loader
neither requires nor fabricates entries for absent devices.  The shipped
two-arm/three-camera example remains valid for the default configuration.
`load_calibration` accepts explicit `arms=`/`cameras=` selectors (default:
all) and enforces the same exact-set rule.

### Measurement procedure (manual)

`tools/litearm_real/calibrate.py` validates and stamps a measurement file.  It
does **not** move an arm; passing `--allow-motion` makes that refusal explicit.
Measure each item by hand and record how:

1. **Base pose.**  With the arm at a known joint configuration, measure the
   base frame origin and axes in the world frame (fiducials or a metrology
   arm).  Write the 4x4 base-to-world transform.  Verify by comparing
   `get_tcp()` (flange) with the computed FK for several poses; disagreement
   over 20 mm is rejected by the observation builder.
2. **TCP.**  Measure the grasp-point position/orientation relative to the
   flange, including the physical LiteGrip mount.  Write `flange_to_tcp`.  Set
   the same offset in the SDK and compare `get_tool_tcp()` with the computed
   TCP at several joint configurations.
3. **Joints.**  For each arm, verify the SDK joint order, direction and zero
   offset against the URDF.  Move one joint at a time by hand (controller
   disabled) or with the operator's low-speed tool, and record the mapping and
   actual soft limits.  Never guess a sign.
4. **Gripper.**  The LiteGrip SDK ships `src/litegrip/factory_calibration.json`
   (`calibrated: true`, `channel: can0`), so a `can0` gripper needs no extra
   file: a no-argument `load_calibration()` uses it.  The SDK deliberately
   refuses to reuse that file implicitly on `can1`; run the vendor
   `zero()`/`calibrate_guided()`/`calibrate_manual()` for that unit and point
   `devices.<arm>.gripper.calibration_path` at its
   `~/.litegrip/<channel>_calibration.json`.  Measure the closed and fully open
   jaw width with a caliper and verify that a commanded `goto`/`move_at_speed`
   stops within the calibrated travel.  Force is a nominal SDK scaling, not a
   calibrated force; do not use it for safety decisions.  `calibrate.py` below
   stamps the **station** calibration (base/TCP/camera), not this vendor
   gripper file.
5. **Cameras.**  Measure intrinsics with a target and distortion model; measure
   camera-to-world for the fixed head and camera-to-tool for each wrist camera.
   Confirm the image resolution matches the calibration.  For RealSense, the
   driver rejects a device intrinsic matrix that differs from the calibration
   by more than 2% (or 4 px for the principal point).  The RealSense colour
   stream is requested as `rs.format.bgr8` and converted to RGB before it is
   returned: D435i firmware 5.17.0.9 timed out on an `rgb8` colour stream at
   640x480@30 while `bgr8` delivered colour and depth reliably.
   For an OpenCV camera whose V4L2 node cannot be opened by integer index, set
   `cameras.<name>.device` to the path (for example `/dev/video4`) and, when
   the node needs it, `fourcc` (for example `"YUYV"`).  The driver opens the
   path with `cv2.CAP_V4L2` and sets the FourCC before width/height/FPS.
6. **Home.**  Choose a collision-free joint configuration, record it as
   `safe_home_rad`, and verify the lift-and-return path with the planner.  This
   is not the SDK's zero pose.

Replace every example value in a working copy, set its `source` to the actual
measurement record and `quality.status` to a measured status, then stamp it:

```bash
python tools/litearm_real/calibrate.py \
  --from /abs/path/to/measured_calibration.json \
  --out configs/calibration/litearm_real.json \
  --measurement-quality measured \
  --mark-executable
```

The tool refuses to mark a file executable while its input still carries an
`example`, `template` or `unverified` marker.  Only mark `executable=true`
after every section is measured and checked.  The loader refuses a live station
with an unverified/example calibration.

## Starting the service

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_MODE=live \
LITEARM_HW_LOG=/abs/path/service.log \
scripts/litearm_hardware_service.sh
```

For offline verification use `configs/litearm_real.offline.example.json` (or a
copy) and `LITEARM_HW_MODE=offline-fake`.
The service:
- never enables, resets, clears faults, or homes an arm/gripper automatically;
- refuses to start live if an arm/gripper is disabled or faulted (enable and
  clear faults outside the service after inspection);
- owns one runner session at a time;
- keeps read-only snapshots available, but read-only mode refuses motion.

## Running an episode

```bash
LITEARM_HW_CONFIG=/abs/path/station.json \
LITEARM_HW_INSTRUCTION='put the red block in the tray' \
LITEARM_HW_MODE=live \
LITEARM_HW_BUDGET_S=1200 \
scripts/litearm_hardware_play.sh
```

Equivalent direct command:

```bash
pyrualean play \
  --robot litearm_real \
  --arm cells \
  --primitives no-vla \
  --hardware-config /abs/path/station.json \
  --task pick_place \
  --instruction "put the red block in the tray" \
  --out /abs/path/results_hw/run_id
```

The adapter is selected only by `--robot litearm_real`; no environment variable
changes the default LiteArm simulation backend.  `read-only` still connects and
perceives but refuses all motion.  `offline-fake` uses the synthetic station and
cameras; it is never selected because a real driver failed.  `live` requires
an executable calibration with measured quality for the selected active arms
and configured cameras.

## Stop semantics

- **Cancel**: confirmed at a synchronous segment boundary with a fresh state
  read; the arm holds its measured pose and the service enters `stopped`.  A
  new trajectory requires restarting the service.
- **Timeout / heartbeat loss / explicit emergency**: the service calls the
  firmware emergency stop and enters `fault`.  The result records whether the
  stop could be confirmed; an unconfirmed stop is reported as
  `stop_unconfirmed`, not success.
- The service does not call `disable()` or `park()` as a controlled stop.
- A fault never clears itself.  Reconnecting does not resume motion.
- Real emergency-stop hold/disable behaviour and preemption of a blocking
  segment are **field-verification items**.

## Results and diagnostics

An episode directory contains:

- `prompt.txt`, `calls.jsonl`, `tool_turns.json`, `result.json`;
- `steps/NNNNNN/` snapshots: `snapshot.json`, `observation.json`,
  `camera_meta.json`, `device.json`, and `artifacts/...` PNG/NumPy files;
- `episode.json` with the stop evidence and success semantics;
- `hardware_host.json`, `card/` (images and `card.json`).

`State.step` is the snapshot number.  `steps_used`/`env_steps` count executed
trajectory segments or gripper actions.  `environment_success` is false and
`environment_success_known` is false; `task_success` is `not_measured`.  A
model's `finish(status="success")` is recorded separately and is not a task
verdict.

## Diagnostics

- Use `tools/litearm_real/cameras.py` for a Viewer-free camera check.  It
  enumerates RealSense serials/USB links and captures head colour+depth and
  wrist colour to JSON/PNG.  The head serial must negotiate a USB3 link:
  ```bash
  python tools/litearm_real/cameras.py \
    --head-serial HEAD-SERIAL-EXAMPLE \
    --wrist-device /dev/v4l/by-path/... \
    --out runs_hw/camera-check
  ```
- Use `tools/litearm_real/smoke.py` for read-only checks of the handshake,
  hashes, clock, snapshots, TCP and stop evidence.  With `--allow-motion` it
  can run a small motion and a single-object pick/place; the live-motion path
  was **not** run in this repository's validation.
- Use the service log for state read failures, fault reasons and stop
  confirmation.
- Check `steps/*/camera_meta.json` and `device.json` for consistent capture
  metadata.
- `camera_meta` for `step=-1` and `image(..., step=-1)` come from the same
  pinned snapshot; a later state refresh creates a new snapshot.

## Tests

Agent environment (no hardware SDK; no local SAM3/torch):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_litearm.py tests/test_litearm_real.py \
  tests/test_litearm_real_config.py tests/test_litearm_real_geometry.py \
  tests/test_litearm_real_perception.py tests/test_litearm_real_execution.py \
  tests/test_litearm_real_cameras.py tests/test_cli.py -q
```

Device environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_litearm_real_devices.py -q
```

The device test uses `litearm.testing.FakeTransport` and a mock LiteGrip; it is
not physical evidence.  The execution tests use a real localhost TCP socket and
the synthetic station, not mocked client methods.

## Known limits / field verification

- All physical motion, gripper force, emergency stop, firmware behavior, cable
  timing and calibrated accuracy remain to be verified on the workstation.
- Collision geometry is primitive-based and conservative; it is not mesh-exact
  or certified.
- The SDK exposes a speed fraction, not acceleration; the configured segment
  length and timeouts are runtime guards, not a proof of dynamic limits.
- The service has no TLS/authentication in this version.  Keep it on localhost
  or a protected network.
- SAM3 is expected on the Jetson Thor; the examples point at the verified
  address above, but the HTTP service must be running.  If it is unreachable,
  perception calls fail explicitly.  A local `sam3`/checkpoint remains an
  optional fallback.
- RealSense support requires `pyrealsense2` and matching firmware; no camera
  was connected here.
- The example config/calibration are not executable.  Do not set
  `executable=true` without measurements.

## Cube grasp development primitive (disabled)

`robo.grasp_cube` performs one centred, vertical pinch of an upright cube.
It is **disabled by default** and requires the explicit station setting
`manipulation.enable_cube_grasp=true`. This version has only offline tests;
the active station configuration and service have not been changed. Enable
it for a live trial only after the operator approves this version.

Supply a fresh measured `center_xy`, object `top_z`, local `support_top`,
closing-axis cross-section `side_m`, and closing-axis heading `yaw`
(`locate(...).grasp_yaw`). Also supply `pad_height_m`: the verified usable
contact length measured up from the finger tips. Collision capsule dimensions
are not usable pad measurements. The primitive requires a
`fingertip_midpoint` TCP with zero extra tip offset.

The sequence is approach, vertical descent, close, hold stabilization,
settled baseline, and a 15 mm vertical test lift. Default engagement is 65%
of the measured height, with 1 mm of arrival allowance:
`tip_z = top_z - 0.65 * (top_z - support_top) - 0.001`.
It checks measured horizontal centring within 2 mm, vertical and finger-face
alignment within 3 degrees, and actual geometric engagement above half the
height before closing. Both finger tips must clear the support by at least
3 mm. Insufficient pad height, table bounds, gripper stroke, configured
tool/floor bounds, or a shallow/unaligned measured arrival refuses the grasp.

After CLOSE, the primitive waits at least 1 s for at least 0.5 s of fresh,
stable width and nominal-force feedback. Width must match the supplied
cross-section within 3 mm and vary by no more than 0.5 mm over that window.
The reported nominal hold must reach 95% of the configured value and stabilize;
it is not calibrated contact-force evidence. Only then is the settled opening
saved as the existing slip guard's carry baseline. A failed test lift preserves
the squeeze for recovery; a pre-lift failure releases on the support when the
service is healthy. There are no automatic retries or home movements.

`RealGrasp` reports `grasped`, `failure_stage`, actual `overlap_fraction`,
`baseline_width_m`, `baseline_opening`, `lift_reached`, `slip`,
`tcp_to_bottom`, action count and per-stage evidence. `grasped` describes a
measured geometric/hold test, not independent object identity or task success.
Supplied geometry and physical TCP/pad accuracy still require verification.

Run its offline tests from `pyrua-lean`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/miniconda3/envs/pyrualean/bin/python \
  -m pytest -q tests/test_litearm_real_grasp.py
```

Only temporary fake-station test configurations opt in; tests do not connect
to the active hardware service.

## PARK camera unification and fixed stacking (review only)

The operator approved enabling the next trial on 2026-11-11. The review
profile below remains read-only; the selected enabled profile and launch
procedure stay in the local station configuration.

The wrist calibration source for the new review profile is a local PARK
eye-in-hand export; create `configs/calibration/litearm_real.json` from the
published example. The importer rebuilds the wrist
extrinsic from `inv(flange_to_tcp) @ T_flange_cam @ diag(1,-1,-1,1)`.
Re-importing rebuilds from PARK; it never flips a previously imported matrix.
Runtime camera metadata and parsed calibration use `camera_frame="opengl"`
(+x right, +y up, viewing along -z). Explicit OpenCV inputs convert once at
the boundary; unknown frame labels are refused. Depth back-projection,
RGB projection and ray/plane intersection share `hardware/camera_geometry.py`.
Policy code must not try another camera convention.

Wrist intrinsics also come from the PARK export. The 1280x720 intrinsics are
mapped to 640x480 by uniform height scaling and a centred horizontal crop;
the mapping is recorded in `handeye_source`. This matches the previous 640x480
intrinsic values to rounding precision, but the physical stream crop and
calibration accuracy still require field verification.

Prepared files:

- Station: `configs/litearm_real.json`.
- Calibration: `configs/calibration/litearm_real.json`.
- Both cube primitives are disabled; calibration is `executable=false` and
  execution mode is `read-only`. The active service/profile was not replaced.

`robo.stack_cube(source_prompt, support_prompt, pad_height_m=...)` wraps the
whole sequence when the operator later enables both cube primitives:

1. Measure both objects twice before any movement. Reject failed or inconsistent
   3-D measurements and objects outside the calibrated table bounds.
2. Freeze the support centre/top; grasp the source and save its measured TCP
   position and stable hold baseline. Fix the gripper-to-object horizontal
   offset and TCP-to-bottom height for this attempt.
3. Lift vertically until the held object's bottom clears the taller object by
   at least 30 mm. Preserve the held orientation, then move horizontally once
   to `support_center_xy - held_offset_xy` at that height.
4. Check measured TCP/hold and take one verification measurement of each object.
   Support and held-object centres must agree within 3 mm; height must agree
   within 3 mm. Missing or contradictory measurements stop the attempt without
   changing the landing target or making exploratory corrections.
5. Descend, release, lift and retreat using `place_on`. An unsettled open or
   failed retreat reports an incomplete operation. No automatic home is issued.

`RealCubeStack` exposes the frozen support, held offset, landing TCP, carry
height, alignment result, per-stage evidence and action count. `completed`
describes measured execution; independent task success remains unknown.

Offline regression includes independent OpenCV projection/depth round trips,
the actual PARK import and repeated import, contradictory/missing geometry,
disabled gates, and a full runner-to-fake-device RPC stacking sequence. No
live movement or speed/waypoint optimization was performed.

For old recordings made with the wrong wrist axes, the replay script can use
`--use-calibrated-camera-poses` with the review profile. It reconstructs the
wrist pose from recorded capture joints and selected calibration, labels the
output as a calibration replay, and issues zero device actions. Historical
recordings are never rewritten.

## Target tracking and top-plane measurements

The hardware runner tracks each text target within one episode so operators
can distinguish the intended cube from another object of the same colour.
The grasp and stack enable switches remain disabled for the next trial.
These changes have offline regression coverage; physical accuracy is unverified.

Call `locate` on the same target in two distinct captures before grasping and
require `geometry_confirmed=True`. A repeated camera timestamp does not count.
The first geometry becomes a fixed reference. Invalid observations in the
reference camera reset confirmation without replacing that reference. Missing
RGB geometry in a secondary camera does not erase confirmed depth in the
reference camera; a contradictory measured 3-D result still resets it.
The default position tolerance
is 8 mm, top tolerance is 4 mm, and dimension tolerance is the larger of 4 mm
or 30% of the reference dimension. These are software rejection bounds.

Subsequent `locate` and `segment` calls crop SAM3 input to the reference's
projected 3-D search region. Oversized, truncated or out-of-region detections
are rejected. The configured SAM3 score floor applies even if the caller
requests a lower `min_score`. The raw result records `search_roi_rc` and the
camera capture timestamp. Use the same text prompt for later checks.

A settled close associates a held target only when exactly one nearby,
confirmed reference agrees with the gripper width. Its full 3-D offset and
orientation are propagated using measured TCP poses. This is a search
prediction, not measured object geometry. Ambiguous grasps or changed width
feedback return `target_unconfirmed`. Opening clears the held reference.

RGB-D geometry fits a top plane with bounded RANSAC, rejecting foreground
depth outside the expected target volume. The plane RMS bound is 2 mm;
centre and footprint confirmation require at least 75% valid depth coverage
consistent with the fitted upright object's top and sides, including at least
20% top-depth coverage of the detected mask. This allows slanted views without
treating legitimate side depth as an occluder. `measurement_quality` reports
`height_reliable`, `center_reliable` and `footprint_reliable`. A visible top
with an occluded centre is diagnostic `partial_measurement` only: the main
geometry is unavailable and `measurement_failed=True`. Invalid observations
return `error_code="target_unconfirmed"`. RGB-only visual-hull reliability is
evaluated from voxel resolution, viewpoint parallax, hull occupancy, support
contact and agreement between the fitted box and both complete silhouettes.
It is not always false. Valid fits can establish a reference after two captures;
ambiguous or inconsistent fits remain unconfirmed. Quantization uncertainty is
reported separately and does not certify physical calibration accuracy.

Held height comes from the measured pregrasp object, never the table gap.
Tracked grasps select `geometry_mode="held"` automatically. Direct geometric
queries can provide `geometry_mode="held", object_height_m=...` with a measured
height, or `support_z_m=...` for a supported object on another known plane.
They still require measured depth or matched calibrated views. Held tilts
over 5° from upright remain unconfirmed by the horizontal top-plane model.
`rgb_footprint` and `table_projection` are separate estimates and never replace
the measured centre or a failed target.

**Do not** lower the score, change the prompt or use a prediction to bypass
failed identity or geometry checks. Confirmed tracking can be reacquired with
two new valid captures of the original target.

Offline replay of `live-stack-143454` accepts the clean green-block snapshots
70 and 71, with top z 0.1349 and 0.1367 m. It confirms the second capture and
rejects contaminated snapshots 145–147 instead of updating the support height.
The hidden red-block replay remains unconfirmed; it does not select the red
cup. These results validate rejection logic, not physical placement accuracy.

The HORAUD head-camera scene recorded before `live-stack-172530` exposed a
refinement regression: RANSAC selected a plane within its 10° slope bound,
but unconstrained least squares then left that bound and rejected otherwise
valid depth. Green and black interior depth were both 100% valid with about
1 mm residuals. Refinement now solves within the same bound instead of
rejecting a fit that escapes during polishing. The recorded black and green
frames pass depth geometry and two-capture confirmation; the incompatible
black RGB visual hull still fails full-silhouette agreement. The compact
depth/mask fixture is `tests/data/park_horaud_cube_depth.npz`.

The agent guidance also limits failed geometry acquisition to two new planned
views before stopping. Missing wrist depth is not a reason to keep moving when
head geometry is confirmed. This change does not modify SDK arrival thresholds
or activate either cube primitive. A fresh agent process loads the changes.

After a cooperative cancellation, heartbeats keep the stopped session alive
for command-result and stop-evidence readback. Further motion remains refused.
This removes a cancellation/readback race; the service loads it on restart.

Replay saved RGB/depth through the configured SAM3 endpoint from
`pyrua-lean`. The command connects no robot device:

```bash
PYTHONPATH=src ~/miniconda3/envs/pyrualean-hw-agent/bin/python \
  tools/replay_target_perception.py --config configs/litearm_real.json \
  --episode runs_hw/single-right-park-unified/live-stack-143454 \
  --steps 70 71 145 146 147 --prompt 'the green block' \
  --output runs_hw/perception-green-replay.json
```

The tool refuses an existing output file. `--hold-step 86 --held-arm right`
can reconstruct a settled hold from a saved snapshot for later-frame checks.
The report records configuration and perception-source hashes and zero device
actions.

## API reference

Object measurement and the permitted table footprint are separate. Valid
depth points are no longer discarded by the table's x/y bounds; multi-view
carving searches the camera silhouette rays with a bounded voxel count.
`locate` returns measured coordinates and separate `within_table_bounds`,
`footprint_within_table_bounds` and `bounds_error` fields. Outside-region
height still references the configured table plane and is an extrapolation.
`stack_cube` preserves its table refusal and reports the structured
`error_code="outside_calibrated_table"` with the measured location and bounds,
before any grip or motion. Perception success does not authorize extending
the operating region without measured physical support geometry.

<!-- api-reference: litearm_real primitives=no-vla -->
