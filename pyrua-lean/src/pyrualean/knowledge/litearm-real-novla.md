# Operating knowledge for a calibrated, real dual LiteArm / LiteGrip station (no-VLA)

This run uses `primitives=no-vla`: the VLA primitives are not available.  The
geometric primitives below are the complete plan surface.

## What this robot is (and is not)
- The station is calibrated by the operator; the values below are conventions,
  not numbers to assume.  Read the task card for the devices actually present:
  it may have one or both 7-DoF LiteArm arms (`"left"`, `"right"`), a LiteGrip
  gripper on each arm or none, a fixed head camera, and one wrist camera per
  arm.  `state().active_arms` lists the arms; a camera appears in the card's
  available cameras only when it is configured.
- `perception.primary_camera` (shown as "primary camera" on the task card) is
  the camera used when a perception or image call omits `camera`.  On a
  wrist-only station it is the wrist camera, so `locate(prompt)` works without
  a head camera.
- The instruction is `robo.task`.  Read it before planning and follow it.
- There is **no independent task-success evaluator** on this path.  `robo.done`
  raises, no result carries a task-success flag, and
  `finish(status="success")` is your statement only.  The host records
  `environment_success_known = false` and `task_success = not_measured`.
  Do not claim that the task is proven complete; stop when the measured scene
  and your verification show that it is.

## Frames, units, coordinates
- World origin: the centre of the table edge nearest the robots.  `+x` points
  right, `+y` points into the table, `+z` points up.  Read the table top's `z` from the task card; it need not be zero.
  Positions are metres; joint angles and orientation internals are radians.
- The arms' base poses, reachable limits, TCP (grasp-point) transforms, camera
  poses, table extent and obstacles are in the station calibration.  The task
  card reports the measured table extent and current wrist-camera calibration;
  use those, never a remembered height or base coordinate.
- `move_to` drives the **grasp point**, the middle of the LiteGrip finger
  contact region.  The measured pose is `state().left.tcp_pos`; the flange pose
  is also in the observation, but it is not the policy-facing end effector.
- Orientations are `pitch`, `yaw`, `roll` in degrees: pitch 90 points straight
  down, 0 level, -90 straight up; yaw is the heading of the pointing direction
  on the table; roll turns the fingers about the pointing direction.  An angle
  left as `None` keeps its measured value, so a position-only move does not
  rotate the wrist.
- Quaternions in the observation are `[w, x, y, z]`.  Camera extrinsic
  matrices are camera-to-world and use the camera convention: looking down its
  own `-z`, `+y` up.
- Runtime metadata labels this as `camera_frame="opengl"`. PARK's OpenCV
  optical frame is converted once when imported. Do not apply another axis
  flip, try a second projection convention, or write pixel-to-world helpers.
  Use calibrated `locate`/`back_project`; a failed measurement is not a target.
- Perception can measure beyond the calibrated table rectangle. Check
  `footprint_within_table_bounds` and `bounds_error`: an
  `outside_calibrated_table` result is a support/operating-region issue, not
  an IK failure. Height there extrapolates the configured support plane.
  Do not use manual moves to bypass the grasp's table check; obtain the
  measured support boundary/height or move the object into the known region.

## Fixed cube stacking (disabled unless the task card enables it)

- Use `robo.stack_cube(source_prompt, support_prompt, pad_height_m=...)` only
  when the task card enables both cube primitives and the operator has approved
  this version. The default is disabled; do not switch it on inside an episode.
- When both flags are enabled for a cube stacking task, use this complete
  primitive instead of assembling a new grasp/carry/place program from
  individual moves. Supply the measured usable pad height; do not invent it
  from a collision capsule or an unverified CAD offset.
- The primitive measures source and support twice before grasping, freezes the
  support centre/top and held-object offset, lifts vertically to safe clearance,
  carries horizontally once, verifies alignment, then descends/releases/retreats.
  Verification never changes the landing target. A failed 3-D measurement or
  alignment check stops the attempt; do not substitute hand-picked pixels or
  exploratory 2 cm corrections while holding the block.

## Measured state versus command
- Every pose, joint angle and gripper width in `state()` is measured.  A
  commanded target is never reported as measured.  `measured=True` is present
  in the observation.
- `state.step` is an observation snapshot number.  `steps_used` in a primitive
  result counts executed synchronous trajectory segments or gripper actions;
  `robo`'s host `env_steps` is the same device-action count, not simulator
  frames.  They are separate counters.
- A `Move.reached` / `Gripper.settled` result is decided after reading the
  hardware back.  An ACK or a queued command is not arrival.
- The device-action budget is shown in the task card and enforced by the
  toolkit; exceeding it refuses the call.  The episode also has a wall-clock
  and tool-call budget from the host.

## Motion

- Read the task card's TCP reference. With `fingertip_midpoint`, `move_to`
  positions the midpoint of the finger ends, not the finger-pad centre.
  Use a tip target above the physical table and below the object's top for
  a side pinch. Do not add the previous TCP-to-tip offset again.
  After grasping, `place_on(tcp_to_bottom=...)` must use the distance from this
  configured TCP to the held object's bottom, not an assumed half-height.
- `blocked` reports failed measured joint arrival; it does not prove contact.
  Check `note`, actual TCP and scene clearance. Do not repeatedly command
  lower heights after an unexplained following failure.
- The first version executes **synchronous joint segments** (`movej_sync` on
  the SDK).  A straight TCP move is generated as IK waypoints and checked
  before execution; a `path="joint"` move swings the joints.  There is no
  continuous-servo (`move_js`) mode, no raw motor command and no SDK handle in
  the `robo` API.
- `move_to`'s `gripper=robo.OPEN` is applied before the motion.  A
  `gripper=robo.CLOSE` on an empty gripper is not applied during the move
  (the result carries a note); call `set_gripper` after arriving.  `KEEP`
  changes nothing.
- `move_to` returns `reached` only when the measured TCP is within 0.01 m and
  the measured orientation within 5 degrees of the plan, after the final
  segment settles.  `blocked` means the measured arm stopped short of a
  feasible plan; do not push through it.
- The planner checks the calibrated joint limits, the table, configured
  fixed obstacles, self-collision and the other arm's **measured** pose.
  Collision geometry is conservative primitive-based, not mesh-exact; it is
  not a certification of clearance.  Move one arm at a time and keep the idle
  arm out of the way.
- For an empty top-grasp approach, explicitly request `pitch=90`,
  `yaw=locate(...).grasp_yaw`, `roll=0`. A position-only lift preserves the
  safe-home wrist posture, which may sit at a joint limit. If a lift and turn
  together fail, lift first, then reorient with `path="auto"` only in clear
  space above the scene, before using straight descents. Setting pitch/yaw
  alone leaves the measured roll unchanged.
- Before committing to a descent, use `check_move([x, y, z], pitch=90,
  yaw=sight.grasp_yaw, roll=0, path="line")`. It plans without device actions
  and reports the configured `tcp_min_z_m`. The task card identifies the tool
  table mode: `geometry` uses the body capsule and lowest finger floor;
  `tcp_plane` uses TCP height only for tool/table and floor checks. It does not
  check finger/table clearance. Compare heights a few mm apart
  above that bound; do not jump far above the object after one low refusal.
  Read `planned_xyz` and `note`, not just `feasible`: an accepted near miss
  may be too high for contact even when the requested height was appropriate.
  A feasible TCP height does not prove finger-pad contact. The configured
  fingertip offset is a collision envelope, not a verified short-finger length.
- If the straight path is refused, compare the same target with
  `check_move(..., path="auto")` before repeating a move. Auto can swing joints
  and movable objects are not in the fixed collision model: inspect clearance,
  or establish a different pregrasp posture in clear space. A coarse sample's
  joint jump is refined before the planner declares no continuous IK branch.
- `path="line"` refuses when the straight TCP line is not achievable; use it
  for approaches, descents and lifts near objects.  `path="auto"` falls back to
  a joint swing.  A near miss (up to 2 cm / 8 degrees) may be taken with a
  `note`; read it.
- The measured start is authoritative.  If the plan is refused because the
  state changed or is old, take a fresh `state()` and re-plan; do not retry a
  stale trajectory.
- The service validates every trajectory again against the current measured
  start and the other arm.  A mismatch is a refusal, not a silent adjustment.

## Gripper
- `robo.CLOSE` (1) closes, `robo.OPEN` (-1) opens, `robo.KEEP` (0) keeps the
  last command.  MEASUREMENTS are openings: 1.0 is the calibrated open limit,
  0.0 the calibrated closed limit.
- An arm with no configured gripper reports its `gripper` and `width_m` fields
  as `None`; `set_gripper`/`place_on`, and `move_to`
  with OPEN/CLOSE, raise "this arm has no gripper configured".  That arm can
  still move; never pretend it has a gripper.
- With `OPEN`, `opening=` in `(0, 1]` opens to a fraction of the measured
  closed-to-open span.  The width is the station's calibrated
  `closed_m + (open_m - closed_m) * opening`, not a fixed 2 mm + 85 mm rule.
- A close starts a persistent calibrated-force squeeze.  Asking to close again
  while it holds does not restart the grasp.  `holding` means the measured
  fingers stopped apart; it is a **candidate**, not proof that the intended
  object is held.
- After a lift, verify the object moved with the arm: look at the wrist image,
  or `locate` the object again and compare its position.  A close that stops
  near the closed limit is probably empty; a close whose width is far larger
  than the object may have closed on the gripper body or another object.
- `width_m` is measured across the fingers.  Objects can be compliant; do not
  treat a width as a material-independent size.

## Perception
- `segment` returns a SAM3 mask and centroid.  `back_project` returns a
  measured world point.  `locate` measures an object's centre, top, height and
  footprint.  Omitting `camera` uses `perception.primary_camera`; pass an
  explicit configured camera to override it.
- With depth, `locate` fits a horizontal top plane and obtains x/y, footprint
  and yaw from measured points on that plane. RGB silhouette projections are
  separate estimates (`rgb_footprint`, `table_projection`), never the centre.
  Inspect `measurement_quality.height_reliable`, `center_reliable` and
  `footprint_reliable`. An occluded top patch may be reported only in
  `partial_measurement`; it does not authorize a target.
- Measure the same text target in two distinct captures before grasping.
  Require `geometry_confirmed=True`; repeated reads of one frame do not count.
  The host remembers that target's location and dimensions for this episode.
  Position, size or top jumps return `error_code="target_unconfirmed"` without
  replacing the reference. Use the same text prompt for subsequent checks.
- After a uniquely associated settled close, searches use the actual measured
  TCP and full 3-D grasp offset to crop the target region. Gripper width alone
  does not prove identity. Unknown/ambiguous grasps remain unconfirmed.
  Lowering `min_score` cannot lower the configured floor or select another
  object outside that region. Do not change prompts to evade failed tracking.
- Held targets use measured pregrasp height, not their distance above the
  table. A tilted held target or missing depth/matched views can remain
  unconfirmed. Do not substitute table projections or predicted geometry.
  Opening the gripper clears the held target's old location.
- Confirm geometry using a valid calibrated depth view or a resolved
  multi-view fit. Missing wrist RGB geometry does not invalidate confirmed
  head depth. Do not move repeatedly merely to make every camera succeed.
  If source or support is still unconfirmed, try at most two new preplanned
  observation poses, then stop before grasping. Do not search while carrying.
- `table_projection` provides a separate depth-free x/y and yaw estimate on
  the calibrated table, even when height measurement fails. Perspective can
  shift it outside the table; inspect `within_table_bounds`. It is an estimate,
  not a measured object top or permission to descend. With no reliable depth,
  object height still requires matched masks in two calibrated views.
- If `locate` reports `measurement_failed` for a block, its
  `table_projection` still gives x/y and yaw.  Move the wrist above that point
  and `locate` again to get a second calibrated view and a height; do not
  invent a height from the mask.
- An explicit table query such as `locate("the white tabletop")` returns the
  configured support plane with `source="calibrated_table"`, not a depth fit.
- `views` says how many calibrated views agreed.  A `note` or `measurement_quality`
  says when the fit is rough, hidden, too flat or too far.  A mask is the best
  text match, not proof of identity; check the image.
- Occlusion removes points: the measured centre and footprint can be short or
  off-centre.  Do not infer a full object from a partial mask.
- Do not locate from a pose directly above a small block.  With the arm over
  the target the head view is partly occluded by the arm and the wrist mask is
  truncated at close range, so an otherwise good measurement is rejected.
  The head camera sits on the +y side of the table looking toward -y, so move
  the arm sideways **along x** by about 8-10 cm while keeping its y close to
  the block's y (height roughly 0.20-0.25 m).  Offsetting the arm toward
  smaller y puts it on the head camera's line of sight and hides the block
  from the depth stream.  Move directly above the block only for the final
  vertical descent.
- `back_project` at a pixel with invalid depth raises. A generic `"object"`
  mask in another view does not establish correspondence to that pixel. Use
  `locate` with a specific prompt or its labelled table projection instead.


## Short-object grasp (measured-fingertip station)

- Read object top and support geometry from the current measurement and task
  card. The operator-updated trial profile has table/floor z=0.125 m,
  y_min=0.35 m and tcp_min_z_m=0.128 m. Do not reuse the old 0.1215 m grasp target.
- Choose the tip height from object height, usable pad geometry and local
  support clearance. The enabled cube primitive computes the requested
  engagement and verifies the measured arrival; neither a fixed station z
  nor a universal "top_z minus 5 mm" establishes enough contact.
- A correct side grasp closes at about the object's width (20-24 mm here),
  not near 16 mm.  A close near 16 mm means the pads are on the top chamfer:
  open, descend lower and retry.
- `holding` at the instant CLOSE returns is a snapshot.  Wait 1 s, read
  `state`, wait 0.5 s and read again.  A width that keeps falling toward the
  closed limit, or `force_n` below about 13 N, means the object is being
  extruded: OPEN, re-locate, retry; do not lift.
- The force ramp continues after CLOSE returns, so the carry baseline is the
  early close width, not the settled one.  Wait until the width and force
  settle, then issue CLOSE once more (it keeps the existing hold) and read
  `state`; the next move then uses the settled width as its baseline.  Without
  this re-close the first lift can report a false `slip` while the block is
  actually held.
- Lift 10-20 mm first; confirm with the wrist image that the block moved with
  the fingers and that `slip` is false before lifting further.

## Pick and place (first field task: one object into a loose target area)
1. Read `robo.task`, capture `state()` and the current images.
2. `locate` the object; check `measurement_failed`, `views`, `top_z` and the
   footprint.  If height is uncertain, move to a higher observation pose and
   locate again before committing.
3. Choose a grasp point from the **measured** object geometry (top/rim and
   footprint), not from a task constant.  Approach above it with `pitch=90`, `yaw=sight.grasp_yaw`, `roll=0`.
   Establish the pointing-down posture in clear space before a line descent.
4. Descend with `path="line"`; do not close after `blocked` or a descent that
   did not arrive.
5. Close; check `settled`, `holding` and `width_m`.  Lift a few cm with the
   same gripper command.  Verify visually or by re-locating that the intended
   object moved.
6. Move in short segments above the scene.  A carry baseline is compared with
   the measured opening; a large drop is reported as `slip`; stop and recover.
7. At the target, use `place_on` with **measured** `support_top` and the
   measured grasp-point-to-object-bottom distance.  `place_on` opens only
   after a safe descent; `safe_to_home` is true only after the lift and retreat
   both arrived.  Do not use a universal 5 cm offset.
8. `home()` returns to the calibrated safe pose, not the SDK zero pose.  It
   first plans a lift and checks the return path.  Test `reached`; do not claim
   it from a queued/ACK result.

## Stop, faults and recovery
- A normal cancel stops between synchronous segments.  A timeout, lost
  heartbeat or explicit emergency stop calls the firmware emergency stop and
  puts the service into a fault state.  The service never calls `disable`,
  `park` or `home` as a substitute for a controlled stop.
- After a fault, new trajectories are refused and reconnecting does not clear
  the fault or resume motion.  Stop planning, call `finish` if appropriate,
  and let the operator inspect the hardware and restart the service.
- If a tool result says `stop_unconfirmed`, do not treat the arm as stopped;
  do not home or approach.  Report the condition.
- The real emergency-stop hold/disable behaviour and the ability to preempt a
  blocking segment are field-verification items; do not assume them.

## Budgets and reporting
- Keep the device-action budget in mind: each synchronous segment and each
  gripper action counts.  Long lines and repeated approaches spend it
  quickly.  Prefer short, verified moves.
- Record progress with `print`.  `finish(status="success")` is a model claim;
  the host records it separately from execution completion and task success,
  and task success remains unmeasured.
