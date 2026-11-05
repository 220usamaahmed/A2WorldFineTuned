# Handoff (2026-11-05)

Where the work stands, how to get it running on another machine, and what to do next. Read with `README.md` (stage 1:
A2World fine-tuning, done) and `docs/stage2_motion_planning.md` (stage 2 details and all measured results).

## Goal of stage 2
A simulated pick-and-place scene for the UR5e + Robotiq 2F-85 (Isaac Sim 5.1 / Isaac Lab 2.3.2): table with the arm,
drawers on one side, a two-level shelf on the other, a container on the table. Each episode picks one object from a drawer
or the shelf and puts it into the container. Scripted data collection for A2World comes after that (not started).

## State: what works
| Piece | File(s) | Status |
|---|---|---|
| cuRobo motion planning | `ur5e_sim/planning/planner.py`, `make_robot_cfg.py` -> `ur5e_2f85.yml` | works; plans in ~1-4 s |
| Joint-space arm control | `ur5e_sim/joint_traj_action.py`, `ur5e_env.Ur5eTableJointEnvCfg` | tracking error < 1 deg |
| Antipodal grasp sampler | `ur5e_sim/planning/grasps.py` (disk cache per object) | works, varied approach angles 0-110 deg |
| Simplified gripper physics | `ur5e_sim/make_simple_gripper.py`, `gripper_action.SimpleGripperAction`, `ur5e_env.use_simple_gripper` | sliding force-limited pads; linkage visual only |
| Drawers (3 pull-out crates) | `scene/build_pullbin.py`, `scene/layout.py` | hook-and-pull works for all 3 |
| Shelf + container | `scene/layout.py` | geometry fixed (shelf boards 15 / 47 cm above the table, tray at +x end) |
| Pick-and-place episodes | `scene/pick_place.py` | **seed 2: 8/8 episodes** put an object in the container |
| Older tests | `scene/sidebins_curobo.py` (drawer task), `scene/table_grasps.py` (sampled grasps on the table), `scene/gripper_test.py`, `planning/test_planner.py`, `planning/test_shelf_container.py` | kept as regression tests |

Last full result (`pick_place.py --episodes 8 --seed 2`, commit cb553f6 plus the 30 N cap): 4 drawer episodes (soup can,
pudding box x2, potted meat can) and 4 shelf episodes (mustard x2, sugar box x2), placed 1-31 mm from the target pose,
<= 3 deg tilt. Two shelf episodes needed a second/third object because the first had no feasible plan or slipped.

## Changes not yet validated by an episode run
Made after the seed-2 run and committed with this handoff (1e4893f):
1. New overview camera poses in `pick_place.py` (`EYE_A/LOOK_A`, `EYE_B/LOOK_B`): one from beyond the drawers looking at
   the shelf front, one from the +x end. Checked on single frames only. (The earlier ones looked through the shelf walls.)
2. `sys.excepthook` that hard-exits (`os._exit`) in `pick_place.py`, `sidebins_curobo.py`, `table_grasps.py`: a crashed
   Isaac run otherwise hangs in shutdown and keeps its GPU memory (one sat on 12 GB for hours).
3. Grip-force cap back to 30 N (it was briefly 60 N: a 0.35 kg soup can was then not lifted).
**First thing to do on the new machine:** `pick_place.py --episodes 8 --seed 3` to validate these. (The files
`ep00_s3.mp4` / `episodes_s3.json` on the old machine are from the aborted 60 N run; ignore them.)

## Open problems / next steps (in order)
1. Validate the above (seed 3, then a few more seeds); look at the videos for the camera framing.
2. Shelf reachability: only ~60 % of random upright shelf placements have a feasible grasp + extraction + placement plan
   (planner-only test: 19/32; cracker box worst, 2/8). Episodes cope by trying another object. Options: move shelf spots
   closer to the front, a lower shelf height, or let the scene sampler reject unplannable placements up front.
3. The 1.13 kg bleach bottle can slip out at 30 N (lost twice); forces above 30 N let the pads sink into objects in PhysX.
   Options: higher pad friction for heavy objects only, more solver iterations, or drop the bleach from the set.
4. Bin 2: objects on the crate's inner side (crate-local y > 0) are often unreachable under the conservative arm collision
   model (forearm between housings / under the roof). The template UR5e arm spheres overshoot ~2 cm (shoulder 4 cm);
   a tighter arm sphere model would help (see `make_robot_cfg.py`).
5. Then: scripted data collection for A2World (recording cameras, TCP pose, joints, gripper per step as in stage 1's
   `collect.py`), camera choice for the world model, episode variety.

## This machine (RTX 4080 SUPER, set up 2026-11-05)
Set up and smoke-tested; the seed-3 validation (next steps, 1.) has not been run yet.
* **Env: `/home/data/siddiquieu1/envs/env_a2world`** (`conda activate /home/data/siddiquieu1/envs/env_a2world`), a clone
  of `env_isaaclab` with the pinned cuRobo. Don't install this project's cuRobo into `env_isaaclab`: steve_il uses that
  env with the new cuRobo v0.8 (`~/curobo`, different API, same package name). Isaac Lab is shared (editable `~/IsaacLab`).
* `/home/user` (shared, ~92 % full) holds only the code; everything big is under `/home/data/siddiquieu1/` (other disk,
  1.1 TB free), so the hard-coded paths work as they are.
* cuRobo `ebb7170` + warp patch in `/home/data/siddiquieu1/src/curobo`, built with `TORCH_CUDA_ARCH_LIST=8.9` against a
  CUDA 12.8 toolkit in its own env `/home/data/siddiquieu1/envs/cuda128` (conda-forge `cuda-nvcc`, `cuda-cudart-dev`,
  `cuda-libraries-dev`; the system only has CUDA 13.3, a major version off torch's cu128):
  `CUDA_HOME=$C PATH=$C/bin:$PATH CPATH=$C/targets/x86_64-linux/include LIBRARY_PATH=$C/targets/x86_64-linux/lib`.
* conda here refuses the `defaults` channels (Anaconda ToS not accepted); every conda package we use is from conda-forge,
  so pass `--override-channels -c conda-forge`.
* Smoke tests (logs in `/home/data/siddiquieu1/setup_logs/`): `make_robot_cfg.py` OK; `test_planner.py` ALL OK
  (grasp plans median 1.8 s); `gripper_test.py` with `GRIP_FORCE=30` holds the cube at 0.6 mm penetration (the script's
  default is 40 N, which pushes through, as documented); `test_shelf_container.py` (no `--headless`, it has no Isaac app)
  19/32, cracker box 2/8, the same numbers as the old machine; `pick_place.py --episodes 2` 2/2.

## Setting up on a new machine
1. Clone with the submodule: `git clone --recurse-submodules git@github.com:220usamaahmed/A2WorldFineTuned.git`.
2. Isaac Sim / Isaac Lab env `env_isaaclab`: `README.md`, "Setup from scratch", section B.
3. cuRobo into that env: `docs/stage2_motion_planning.md`, "Setup" (pinned commit ebb7170, our warp patch
   `docs/env_patches/curobo_warp118.patch`, deps with a constraints file so `packaging==23.0` etc. stay, build with a
   CUDA toolkit whose major version matches torch's, `TORCH_CUDA_ARCH_LIST` for your GPU).
4. **Paths.** Big files live outside git under `/home/data/siddiquieu1/` and 15 scripts hard-code that prefix
   (`grep -rln /home/data/siddiquieu1 ur5e_sim`). Either create that path (symlink) or replace the prefix.
5. Generated files (not in git), in this order, all with `OMNI_KIT_ACCEPT_EULA=YES python <script> --headless` from the
   folder the script is in:
   * `ur5e_sim/convert_arm.py` and `convert_arm.py --simple` -> `ur5e_sim/assets/usd/ur5e_gripper{,_simple}.usd`
     (`make_simple_gripper.py` regenerates the simple URDF + json if needed; those are committed)
   * `ur5e_sim/scene/fetch_assets.py --list assets.txt` -> Isaac asset mirror (~0.9 GB) in `<data>/assets/Isaac/5.1/Isaac/`
   * `ur5e_sim/scene/build_pullbin.py` -> `ur5e_sim/assets/usd/pullbin.usd` (drawer housing + crate)
   * `ur5e_sim/scene/export_ycb_meshes.py` -> `<data>/assets/ycb_meshes/*.obj` (+ summary.json)
   * `ur5e_sim/scene/make_ycb_physics.py` -> `<data>/assets/ycb_physics/*.usd`
   * `cd ur5e_sim/planning && python make_robot_cfg.py` -> **regenerate `ur5e_2f85.yml`**: it stores the URDF's absolute
     path. The script also checks cuRobo's kinematics against the sim (must print "OK").
   * grasp caches build themselves on first use (`<data>/assets/grasp_cache/`, ~1 min per object).
6. Smoke tests, one at a time: `planning/test_planner.py` (must end "ALL OK"), `scene/gripper_test.py`,
   `planning/test_shelf_container.py`, then `scene/pick_place.py --episodes 2`.

## Gotchas (learned the hard way)
* Run Isaac / cuRobo jobs **one at a time** (on the old machine: GPU 1 only, shared with other users). Isaac can hang in
  `app.close()`: the scene scripts now `os._exit`, but check `nvidia-smi --query-compute-apps=pid,used_memory` + `ps` for
  leftovers and `kill -9` them (SIGTERM is ignored). Don't `pkill -f <pattern>` from a shell whose own command line
  contains the pattern: it kills that shell.
* Isaac Sim does not flush stdout: print with `flush=True`. Isaac Lab logs go to `ur5e_sim/logs/` (on the old machine
  `/tmp/isaaclab` belonged to another user).
* cuRobo: set `CUROBO_TORCH_CUDA_GRAPH_RESET=1` (planner.py does); IK returns arbitrary 2*pi-equivalents (UR joints are
  +-2pi), wrap them; `update_obstacle_pose(..., update_cpu_reference=True)` or attach uses stale poses; the minimum
  trajopt dt equals `interpolation_dt` (keep 0.02, resample to 10 Hz). Details in `planner.py` docstrings.
* PhysX gripper: >~300 N per kg of object, or > ~30 N absolute, and the pads push into / through objects. Keep grasps on
  the simulated collider shape (convex hull for box/can objects), not the visual mesh.
* The YCB objects Isaac ships without physics need the wrappers from `make_ycb_physics.py`; several YCB objects cannot be
  grasped by an 85 mm gripper at all (master chef can, wood block, tuna can lying, marker, scissors).

## Outputs on the old machine (not in git)
`/home/data/siddiquieu1/scene_previews/` (`pick_place/`, `curobo/`, `table_grasps/`, logs), the asset folders above, and
stage 1's data / checkpoints (see README, "Where the big files are").
