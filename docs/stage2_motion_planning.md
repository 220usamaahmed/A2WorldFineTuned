# Stage 2: cuRobo motion planning and sampled grasps

How the arm is moved in the stage-2 scenes (drawers now, shelves and table later), what was built, and what was measured.
Everything here is simulation (Isaac Sim 5.1 / Isaac Lab 2.3.2, UR5e + Robotiq 2F-85).

## Before / after

| | before (sidebins_test.py) | now (planning/, sidebins_curobo.py, table_grasps.py) |
|---|---|---|
| trajectory | straight Cartesian lines between hand-picked waypoints, fixed transit height | cuRobo MotionGen (collision-aware, time-parameterised), straight lines only for approach / lift / pull / place |
| low-level control | differential IK (DLS) on a TCP pose; PD drives with zero velocity target (lag up to 3 cm) | joint targets interpolated over the physics substeps + velocity feed-forward (joint_traj_action.py) |
| collision avoidance | none (waypoints chosen to stay clear) | table, drawer housings, crates at their current slide, objects (YCB meshes) at their current poses; carried object attached to the robot |
| grasp pose | fixed rule: top-down, object centre, fingers along world x | antipodal sampler on the object mesh, any object pose; the planner picks a reachable one |
| gripper physics | Robotiq four-bar linkage with mimic constraints, squeeze ~ gain x remaining angle, friction 4 needed | two sliding pads, 20 N force-limited and coupled; the linkage is visual only (follows the pads); friction 1 |

## Pieces

* `ur5e_sim/planning/make_robot_cfg.py` -> `ur5e_2f85.yml`: cuRobo robot model from our URDF. TCP = `ur5e_tool0` + 0.1388 m
  (same as the sim); checked against the sim's ready pose (0.3 mm). Arm collision spheres: cuRobo's UR5e template.
  Gripper spheres: own cover of the 2F-85 collision meshes (cuRobo's fitter covered < 25 % of these links' surfaces);
  full coverage, sticks out <= 13 mm. `finger_joint` locked open. An `attached_object` link carries grasped objects.
* `ur5e_sim/planning/planner.py` (`ArmPlanner`): free-space plans, joint-space plans, straight lines, batched IK,
  attach / detach, obstacle sync, and `plan_grasp` (below). Output: joint trajectories at the 10 Hz control step.
* `ur5e_sim/joint_traj_action.py` + `ur5e_env.Ur5eTableJointEnvCfg`: the arm action is the joint configuration to reach at
  the end of the control step.
* `ur5e_sim/planning/grasps.py`: antipodal grasp sampler (+ disk cache per object).
* `ur5e_sim/scene/layout.py`: scene geometry as numbers (drives both the Isaac scene and the cuRobo world).
* `ur5e_sim/scene/export_ycb_meshes.py`, `make_ycb_physics.py`: YCB meshes as .obj (object root frame) and
  physics-enabled USD wrappers for the 17 YCB objects Isaac ships without physics (official masses, convex colliders).

### Simplified gripper physics
`ur5e_sim/make_simple_gripper.py` writes `assets/ur5e_gripper_simple.urdf` (+ `simple_gripper.json`); `convert_arm.py
--simple` makes the USD; `ur5e_env.use_simple_gripper(cfg)` switches an env to it (default in `sidebins_curobo.py` and
`table_grasps.py`, `--gripper linkage` for the old model). The Robotiq finger links keep their meshes but lose their
colliders and mimic joints. Two invisible pads on prismatic joints from the gripper base do the gripping: each slides on
the straight line from the real pad's open to its closed position (44.5 mm, 13.9 mm of it along the approach axis), at
most 4.5 mm from the real pad's arc. Gripping face = the real pad (22 x 37.5 mm); the collider is 15 mm thick, extending
outwards. Drive: 4000 N/m, force limit 20 N (the real 2F-85's minimum), 0.1 m/s. `SimpleGripperAction` keeps the action
meaning (finger angle 0..0.8 rad -> the pad gap the real linkage has at that angle), couples the pads like the real
mechanism (neither target may lead the other pad by more than 5 mm) and drives the visual linkage, per side, to the angle
matching that pad's measured position (rendered pads within 4.7 mm of the physics pads).
Checked (`scene/gripper_test.py`): free air, gap = command at every angle; 5 cm cube floating between the pads: 5-30 N hold
it centred with <= 1.5 mm penetration. Found on the way: independent pads (a neutral tie) let one pad shove the cube 27 mm
and push the other back to its stop; 6.35 mm thick pads at 40 N, and 15 mm ones at 40 N, are pushed through the cube.

### plan_grasp
Given candidate grasp poses, it works backwards from the grasp: collision-aware IK at the grasps (the object itself
ignored), several IK branches each and both 2*pi-equivalents where a joint is ~pi away; (grasp, branch) pairs tried
nearest-first in joint space (or in the caller's order, for variety); for each, a straight Cartesian IK chain from the grasp
back to the pregrasp (10 cm along the approach axis), then a free-space plan to that pregrasp configuration, falling back
to going via a high home pose, or via a point straight above the pregrasp at 0.35 m. Reasons, all found the hard way:
planning to the pregrasp *pose* lets cuRobo pick an IK branch from which the straight descent is impossible (the wrist ends
up under a housing roof); cuRobo's hold-pose trajopt fails when an arm link is within a few cm of an obstacle; its
joint-space trajopt fails on wrist reconfigurations even where the straight joint line is free; the old ready pose is only
3 cm above the table, too low to flip the wrist.

### Grasp sampler
Surface points p1 with normals, a ray into the object finds the opposite surface p2; keep pairs narrower than the open gap
minus 8 mm whose normals lie in the friction cone (mu 0.5); each pair gives 16 approach directions around the grasp axis.
A grasp is rejected if the open gripper (42 boxes from the URDF at finger_joint 0, slanted links cut into slices), swept
back 10 cm along the approach, overlaps the object's surface points. Score: antipodality, distance of the grasp axis from
the object's centre, width margin. Per scene, grasps are moved to the object's simulated pose, filtered so the open gripper
stays above the table, and a score-weighted random subset is passed to `plan_grasp` in that order.

## Pick-and-place episodes (shelf + drawers -> container)
`ur5e_sim/scene/pick_place.py --episodes N --seed S`. Layout (`scene/layout.py`): drawers on the -y side as before; a
two-level wooden shelf on the +y side (0.9 m wide, 0.30 m deep, boards 15 cm and 47 cm above the table, open top, front at
y = 0.32); Isaac's shallow grey tray (`Props/PackingTable/props/container_h20`, 0.61 x 0.41 x 0.09 m) on the table at the
+x end, between a pulled-out crate (reaches y = -0.17) and the shelf. Only the near half of the tray is within reach.
Objects per run: 3 short ones in the crates (from soup can, pudding box, gelatin box, potted meat can, foam brick, banana;
any resting pose in which the 2F-85 can grasp them) and the 4 tall ones on the shelf (mustard 19 cm, sugar box 18 cm,
cracker box 21 cm, bleach 25 cm; upright). The master chef can and the wood block are wider than the 85 mm opening.

Each episode: reset (crates closed, all objects in new random poses), source alternating drawer / shelf, a random object
there; drawer: hook and pull the crate (the longest pull whose unhook ends collision-free); grasp: sampled grasps
(crates: within 40 deg of vertical) in score-weighted random order, accepted only if the object can also be got out
(`planning/tasks.plan_extraction`: lift 3 cm, attach, up out of the crate or straight out of the shelf); place: drop poses
in the tray in the orientation the object rested in (3 positions x 8 yaws), raised so the gripper clears the tray walls
(tall objects held from the side are released a few cm up), reached like a grasp with a vertical descent; release, back
off, home. If an object has no feasible plan, the next object of the same source is tried. Grip force per object:
250 N/kg x mass, 3-60 N (PhysX pushes the pads into light objects above ~300 N/kg).

Shelf reachability with the planner alone (`planning/test_shelf_container.py`, each tall object upright at 4 spots per
level, random yaw): grasp + extraction + placement planned for 19/32 (lower level 8/16, upper 11/16; cracker box 2/8,
the others 5-6/8). With the lower board at table height it was 3/16 for that level.

Episode runs (6-8 episodes each, alternating drawer / shelf): seed 0, first version: 1/4 (foam brick pushed through at
20 N, bleach slipped, side-on crate grasps unplannable, mustard grasped by its cap); seed 0 after per-object grip force,
crate grasp filter and centre-of-mass rescoring: 4/6; seed 1 (after raising drop poses above the tray walls): 2 of the
first 3 episodes, then a crash (fixed: unhook end state in collision for bin 1, float64 reset tensor); **seed 2: 8/8**
(4 drawer: soup can, pudding box x2, potted meat can; 4 shelf: mustard x2, sugar box x2; 1-31 mm from the target pose,
<= 3 deg tilt; two shelf episodes needed a second / third object). Not yet validated: the final camera poses and the
hard-exit excepthook (see HANDOFF.md).

## Results (2026-11-04)

### With the simplified gripper (current default)
**Sampled grasps on the table**, same scenes as the four-bar runs below (4 objects x 5 trials, seed 2): **20/20 lifted** at
friction 4 and **20/20 at friction 1** (soup can, pudding box, potted meat can, gelatin box: 5/5 each; approach angles
0-75 deg from vertical: 7 / 6 / 2 / 0 / 5 in 15-deg bins). The four-bar gripper lifted 5/20 and 7/20 in these scenes.
Foam brick alone (8 trials, seed 5): 6/8. The two failures are deterministic (same result with 255-vertex hulls): the pads
close 3-5 mm inside the sampled width (72.7 mm on a 75.9 mm grasp; 44.6 on 50.1) and the brick stays on the table, while
in the six successes the pads stop within 1 mm of its surface. Cause not found yet.

**Drawer task, mixed objects, sampled grasps** (seed 0, friction 1): gelatin box placed 2 mm / 2 deg from the target pose,
banana 3 mm / 0 deg; the foam brick's grasp (76 mm, at the gripper's limit) did not lift it (the brick problem above).
**Drawer task, cans** (friction 1): bins 0 and 1 opened, picked (TCP error 1.0-1.1 mm) and placed (0-2 mm, 0 deg). Bin 2's
can was lifted, but the straight put-down failed at 95 %: the carried can's spheres come inside the 1 cm collision
activation distance of the table when set down 4 mm above it. Fixed by putting down to 2 cm with full checking and the
last 2 cm with only the table ignored. On the rerun the final 2 cm still failed for bin 2 (with the can held lower in the
new pads, the wrist met bin 2's pulled-out crate wall), so bin 2's place spot moved from (0.40, -0.08) to (0.42, -0.04):
bin 2 then opened (0.151 m), picked (1.0 mm) and placed (4 mm / 4 deg). So all three bins work with the new gripper, bins
0-1 from the full run and bin 2 from a single-bin rerun. `ArmPlanner.plan_pose` also gained a fallback (IK + joint-space
plan), after one free-space plan to bin 2's rim bar failed in cuRobo's fine-tuning step.

### With the four-bar gripper (the earlier runs)

**Planner alone** (`planning/test_planner.py`, no simulator): top-down grasp plans at 3 positions in each of the 3 open
crates, from both start poses: 18/18, straight descent within 0.1 mm, planning median 2.1 s (max 3.5 s) on a shared
RTX A6000. Free-space plans to table poses: 0.4-0.6 s. Motion durations 1.3-4.5 s at half the time-optimal speed.

**Drawer task, cans, top-down rule grasp** (`scene/sidebins_curobo.py --can-y=-0.04,0,-0.04`, one run):

| bin | crate pulled | TCP error at grasp | can lifted | placed upright | TCP error at place |
|---|---|---|---|---|---|
| 0 | 0.149 m | 0.9 mm | 9.1 cm | yes | 1.0 mm |
| 1 | 0.119 m | 1.0 mm | 9.1 cm | yes | 1.4 mm |
| 2 | 0.149 m | 0.9 mm | 9.3 cm | yes | 0.9 mm |

Joint tracking error stayed below 0.8 deg in every motion (the old IK-pose control lagged up to 3 cm). Caveats:
bin 1's pull is shortened to 0.12 m automatically (at 0.15 m the folded arm collides with itself in the sphere model: the
template shoulder sphere is ~4 cm too fat); bin 2's can had to move from crate y = +0.04 (the old test's spot) to -0.04:
at +0.04 every IK solution puts the forearm under the housing roof or between the housing walls with 0-2 mm, below the
1 cm activation distance (the old script's run probably clipped the walls; PhysX allowed it). Place spots were moved 5 cm
away from the pulled-out crates (the old -0.12 left 14 mm between the can and bin 2's crate).

**Sampled grasps on the table** (`scene/table_grasps.py`, 4 objects x 5 trials, random graspable resting pose and yaw,
seed 2, 40 score-weighted random candidates per pick):

| gripper stiffness | planned | lifted | soup can | pudding box | potted meat can | gelatin box |
|---|---|---|---|---|---|---|
| 6 (as tuned) | 17/20 | 5/20 | 1/5 | 2/5 | 0/5 | 2/5 |
| 20 | 20/20 | 7/20 | 3/5 | 1/5 | 0/5 | 3/5 |

Approach angles of the executed grasps spread from 0 to 75 deg from vertical (k = 20: 3 / 4 / 7 / 3 / 3 in 15-deg bins).
An earlier run with the foam brick (28 g): 6/6 lifted. The failures are at the gripper, not the plan: the arm reaches
the grasp within 0.1 mm, the object does not move during the approach, and after closing the pads sit on the object
(e.g. 69 mm apart on the 72.5 mm gelatin box), yet the object stays on the table. The finger joint angles do not match the
pad positions (left finger 0.55 rad with its pad where 0.12 rad would put it): the simulated four-bar linkage gives way
under contact instead of squeezing. Outcomes for the same grasp flip between stiffness 6 and 20 in both directions, so
stiffness is not the lever. This is the gripper model the previous stage tuned on a 5 cm cube.

**Drawer task, mixed short objects, sampled grasps** (`--objects mixed --grasps sampled --seed 0`, one run): all three
crates opened (0.148 / 0.118 / 0.148 m) and all three grasps were planned and reached (TCP error 0.0 mm).
Foam brick: 30 deg approach, lifted, placed 5 mm / 4 deg from the target pose. Gelatin box: 11 deg approach across its
27 mm side, lifted and carried, but it ended 86 mm / 122 deg off the target (tipped over at or after the release).
Banana: 9 deg approach, not lifted (gripper, as above).

Not graspable by the 2F-85 in any resting pose (grasp sampler + floor filter): tuna can (86 mm across, 34 mm tall),
large marker, scissors (too flat). Lying flat, the pudding box and potted meat can are not graspable either (their
horizontal sides are wider than the 85 mm opening); scene sampling now only uses resting poses with >= 20 valid grasps.

## Setup (in the Isaac Lab env `env_isaaclab`; on the RTX 4080 machine a clone of it, see HANDOFF.md)
cuRobo at the commit Isaac Lab 2.3 tests against, built against the env's torch 2.7 / cu128 with the CUDA 12.9 toolkit
(`/usr/local/cuda-12.8` here has no nvcc; torch only refuses a different CUDA *major* version):
```bash
git clone https://github.com/NVlabs/curobo.git /home/data/siddiquieu1/src/curobo && cd /home/data/siddiquieu1/src/curobo
git checkout ebb71702f3f70e767f40fd8e050674af0288abe8 && git apply <repo>/docs/env_patches/curobo_warp118.patch   # warp 1.18 dropped wp.torch.*
# dependencies, without moving anything Isaac pins (packaging==23.0 in particular; setuptools_scm>=9 would pull 26.x)
printf "torch==2.7.0+cu128\nnumpy==1.26.0\nwarp-lang==1.18.0\nscipy==1.15.3\ntrimesh==4.5.1\nnetworkx==3.3\nsetuptools==80.10.2\npackaging==23.0\n" > /tmp/c.txt
pip install -c /tmp/c.txt pybind11 numpy-quaternion "setuptools_scm>=6.2,<9" yourdfpy importlib_resources scikit-image ninja
unset PYTHONPATH PYTHONHOME; export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6
pip install -e . --no-build-isolation --no-deps        # ~20 min of CUDA compilation
```
Then, from `ur5e_sim/planning/`: `python make_robot_cfg.py` (regenerates `ur5e_2f85.yml`, checks it against the sim) and
`CUDA_VISIBLE_DEVICES=0 python test_planner.py`. Asset prerequisites for the scene scripts (all under /home/data):
`scene/export_ycb_meshes.py` (meshes) and `scene/make_ycb_physics.py` (physics wrappers), each run once with `--headless`.
Grasp caches build themselves on first use (`/home/data/siddiquieu1/assets/grasp_cache/`, ~1 min per object).

Commands used for the results above:
```bash
cd ur5e_sim/scene
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python sidebins_curobo.py --headless --can-y=-0.04,0.0,-0.04
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python table_grasps.py --headless --trials 5 --seed 2 \
    --objects 005_tomato_soup_can,008_pudding_box,010_potted_meat_can,009_gelatin_box --grip-k 20
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python sidebins_curobo.py --headless --objects mixed --grasps sampled --seed 0
```
Videos and json results go to `/home/data/siddiquieu1/scene_previews/{curobo,table_grasps}/`.

## Known limits
* Collision model is conservative: the template arm spheres stick out ~2 cm (the shoulder sphere 4 cm), plus a 1 cm
  activation distance. Some placements the physical arm could reach are refused (examples in Results).
* Thin-walled YCB meshes (mug, bowl) give no grasps: the Isaac meshes are open shells 1.5-2.5 mm thick, and their simulated
  colliders are convex anyway.
* Grasp success is judged by lifting in the simulator only; no shaking, no repeated trials per grasp.
* Simplified gripper: the physics pads move on a straight line, the rendered Robotiq fingers on the real arc, up to 4.7 mm
  apart; the squeeze is a constant 20 N (PhysX pushed the pads through a 5 cm cube at 40 N); the foam brick problem above.
