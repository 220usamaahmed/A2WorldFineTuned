# A2World experiments: world-model prediction for a simulated UR5e + Robotiq

Question: how well does the **A2World** action-conditioned video world model
([paper](https://arxiv.org/abs/2606.29501), [code](https://github.com/LogosIntelligenceLab/A2World),
[weights](https://huggingface.co/Fleurrr/A2World-World-Model)) predict the future for a **UR5e arm with a Robotiq 2F-85 gripper**
moving a cube on a table (including failures where the cube falls off), and what does fine-tuning on simulated data from that robot buy?

## What was done

| Step | Result |
|---|---|
| 1. Ran the released **LIBERO** checkpoint on 10 real LIBERO demos (`eval/eval_libero.py`) | Beats "copy the first frame": agent view 23.9 vs 22.3 dB, wrist 20.8 vs 13.0 dB over 80 steps. ~27 s per 20-step chunk on an RTX A6000. |
| 2. Built a **UR5e + Robotiq tabletop in Isaac Sim** and recorded scripted episodes (`ur5e_sim/`) | 3 RGB cameras (2 exterior, 1 wrist, 256x256, 10 Hz), TCP pose, joints, gripper, cube pose per step. 5 episode kinds: `success`, `drop_on_table`, `release_off_edge` (cube falls to the floor), `push_off_edge` (falls), `miss_grasp`. |
| 3. Ran the **pretrained** A2World checkpoint on 15 held-out episodes (`eval/eval_ur5e.py`) | Poor: mean PSNR 18.7 / 14.0 / 20.0 dB (ext1 / wrist / ext2) vs 21.9 / 16.4 / 23.5 for copying frame 0. Arm and gripper smear after ~2 chunks, the cube is mistaken for other objects and never falls. |
| 4. **Fine-tuned** from the pretrained checkpoint on 400 simulated episodes, 2 GPUs, 3,600 iterations (`ur5e_sim/run_main_finetune.sh`) | Same 15 held-out episodes (`eval/eval_ur5e_ft.py`): see below. |

### Fine-tuning result (held-out episodes, open-loop rollout from frame 0, 60-160 steps)

| view | copy frame 0 | pretrained (3-view) | fine-tuned @1500 | fine-tuned @3600 |
|---|---|---|---|---|
| agent view (ext1) PSNR | 21.9 dB | 18.7 | 27.8 | **28.1** |
| wrist PSNR | 16.4 dB | 14.0 | 23.2 | **24.6** |
| agent view SSIM | | 0.76 | 0.94 | 0.94 |

- Over the last 20 steps (episodes of 80+ steps): agent view 25.7 dB (copy frame 0: 20.0, pretrained: 16.6), wrist 20.5 (13.2, 10.8).
- Cube falling: in the 3 `push_off_edge` episodes the predicted cube leaves the scene (pretrained: 1 of 3). For `release_off_edge` the red-pixel count says "still visible", but the montages show the cube dropping past the edge; the real agent-view camera hides it behind the table, so that metric undercounts here.
- Iteration 1500 (~12 h of training) already reaches almost the same agent-view score; the wrist view kept improving.
- The fine-tuned model is the **2-view, history-aware variant** (agent view = ext1, wrist), not the 3-view pretrained model, so the comparison is not strictly like for like.

Per-episode metrics, PSNR-vs-step plots, logs, videos and montages are in `results/` (local only: the folder is git-ignored; the git repo holds code).

### Limitations (read before building on this)
- Held-out episodes use the **same simulated scene, cameras and cube range** as the training data: this is in-distribution adaptation only. No test of new camera poses, scenes, longer horizons (>160 steps), other tasks, or real robots.
- The scene is a bare block on a checkered floor (large visual gap to real data). The cube sometimes slides a little in the grasp; 2 of 400 training `success` episodes ended with the cube dropped (kept, labelled `unexpected` in `summary_*.json`).
- The DROID-example control run for the pretrained model was not done. Input conventions used for the pretrained evaluation (DROID-style per-step deltas in the previous end-effector frame, gripper 0 = open, view ids 2 0 3) are my reading of the repo.
- Training loss is noisy and not a quality measure; the repo's "validation" reads the training folder (not held out). Only the held-out rollouts are evidence.

## Repository layout

```
README.md
docs/
  UR5e_finetuning_plan.md        first, general plan (data, cameras, format, estimates)
  UR5e_finetune_2gpu_plan.md     the executed one-day / 2-GPU plan, pilot results (section 10) and outcome note
  env_patches/                   patches needed to train on this machine (see "Environment")
eval/
  eval_libero.py                 LIBERO checkpoint on LIBERO demos (step 1)
  eval_ur5e.py                   pretrained checkpoint on UR5e episodes, 3 views (step 3)
  eval_ur5e_ft.py                fine-tuned checkpoint on UR5e episodes, 2 views (step 4)
ur5e_sim/
  assets/ur5e_gripper.urdf       UR5e + Robotiq 2F-85 only (no mobile base, no cabinet), mesh paths relative to assets/meshes/
  convert_arm.py                 URDF -> USD (Isaac Lab importer, fixed base, mimic joints)
  robot_defs.py                  joint/link names, ready pose, TCP offset (from the Steve project)
  gripper_action.py              tuned Robotiq action term: finger angle -> all six gripper joints (from the Steve project)
  ur5e_env.py                    Isaac Lab environment: table, cube, 3 cameras, differential-IK TCP control
  collect.py                     scripted episodes + recording (data.npz and mp4 per episode)
  preview.py                     render the three cameras once (to check placement)
  make_check_video.py            frames + state plots video for recorded episodes
  convert_to_train.py            episodes -> A2World's LIBERO training layout (own converter, see below)
  run_main_finetune.sh           the exact fine-tuning command that was run
tools/nccl_check.py              2-process NCCL all-reduce test (diagnoses the GPU 0/1 hang)
results/                         (local only, git-ignored)
  libero_eval/  libero_demo/     step 1
  ur5e_pretrained/               step 3
  ur5e_finetuned/iter1500|iter3600/   step 4
  ur5e_pilot_check/              check videos of the 15 held-out episodes
A2World/                         git submodule: upstream A2World, pinned at commit 077e10a, unmodified (holds the weights and the training venv)
```
Not in git (kept on disk): `results/`, `ur5e_sim/data/` (the 15 held-out episodes), `ur5e_sim/assets/meshes/` and `ur5e_sim/assets/usd/` (meshes and generated USD). Only code, docs and the small URDF are committed.

## Where the big files are
| Path | Contents |
|---|---|
| `ur5e_sim/data/pilot/` | **held-out test set**: 15 episodes (seed 0), never used for training |
| `/home/data/siddiquieu1/a2world_ur5e/train/` | 400 training episodes (6.6 GB; seeds 1 and 2) |
| `/home/data/siddiquieu1/a2world_ur5e/train_processed/` | the same, in A2World's training layout (1,582 clips, 153 MB) |
| `/home/data/siddiquieu1/a2world_runs/runs/.../ur5e_main/` | fine-tuning run: the final training checkpoint (iteration 3600, 24 GB with optimizer state; the intermediate ones were deleted) and logs |
| `/home/data/siddiquieu1/a2world_runs/exports/` | release-format weights: `ur5e_iter1500_ema.pt`, `ur5e_iter3600_ema.pt` (5 GB each; use these for inference) |
| `/home/data/siddiquieu1/a2world_runs/pretrained_bundle/` | symlink bundle of `a2world-pretrained.pt` used as the starting point |
| `A2World/world_model/a2world_ckpt/` | released `a2world-libero.pt` and `a2world-pretrained.pt` (from Hugging Face) |

## Environment
- **GPUs:** only GPU 0 and GPU 1 may be used on this machine (2 and 3 belong to others). `/home/user` is almost full: put large outputs on `/home/data/siddiquieu1/`.
- **Isaac Sim 5.1 + Isaac Lab 2.3.2**, conda env `env_isaaclab` (Python 3.11, torch 2.7.0+cu128), Isaac Lab cloned at `~/IsaacLab` (tag v2.3.2, `./isaaclab.sh --install none`).
  Needed extras: `setuptools<81` and `flatdict==4.0.1 --no-build-isolation` before `pip install -e source/isaaclab` (flatdict's build needs `pkg_resources`); `packaging==23.0`.
  Isaac Lab wants to write logs to `/tmp/isaaclab` (owned by another user here): `ur5e_env.py` redirects them to `ur5e_sim/logs/`. Isaac Sim exits without flushing stdout, so print with `flush=True`.
  Run Isaac scripts as `OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python <script> --headless`.
- **A2World**: `A2World/world_model/.venv` (Python 3.10, torch 2.6.0+cu126, built with `uv sync --extra cu126` from the repo's lock file), `pip install -e . --no-deps`. Run training and evaluation **from `A2World/world_model/`** (the base-model assets are found relative to it).
- **Patches needed to train** (`docs/env_patches/`, all already applied in the venv):
  1. apex `FusedAdam` needs compiled CUDA extensions. Build apex tag `24.04.01` with only `amp_C` and `apex_C` (`apex_setup.patch`: relaxes the nvcc-vs-torch minor-version check and skips extensions that do not compile with torch 2.6), `TORCH_CUDA_ARCH_LIST=8.6`, nvcc 12.9.
  2. Transformer Engine 1.13 cannot load the stripped release checkpoint through torch's checkpoint loader (`te_ops_op.patch`, `te_module_base.patch`: ignore the `_extra_state` placeholder; it is FP8 metadata, unused in bf16).
  3. **`NCCL_P2P_DISABLE=1`**: peer-to-peer between GPU 0 and GPU 1 hangs (an all-reduce never returns; check with `tools/nccl_check.py`). `run_main_finetune.sh` sets it.
- **Robot assets** come from the user's Steve project (github.com/220usamaahmed/steve_il, commit `e7592bb`): the UR5e + Robotiq URDF with its meshes, the joint names and the tuned gripper controller were copied into `ur5e_sim/` (`assets/`, `robot_defs.py`, `gripper_action.py`), so this repo does not need that project. The meshes (`ur5e_sim/assets/meshes/`, 13 MB, UR and Robotiq) are git-ignored because their licences are unchecked; to restore them copy `assets/steve/meshes/{ur5e,robotiq_2f_85}` from that project.

## Setup from scratch
`A2World/` is a git submodule pinned to upstream commit `077e10a` (the version used here; we changed none of its files; upstream has since moved to `LogosIntelligenceLab/A2World` with one docs-only commit). The patches below are applied to *installed packages*, not to the A2World source.
```bash
git clone --recurse-submodules <this repo> && cd A2WorldExperiments      # or: git submodule update --init
```
**A. A2World training/inference environment** (Python 3.10, torch 2.6.0+cu126, RTX A6000 = sm_86)
```bash
cd A2World/world_model
uv sync --extra cu126                      # creates world_model/.venv (the commands in this README use it); then: source .venv/bin/activate
uv pip install -e . --no-deps
# apex with CUDA extensions (FusedAdam, needed for training): tag 24.04.01 + our patch, only amp_C and apex_C are built
git clone --depth 1 --branch 24.04.01 https://github.com/NVIDIA/apex /path/to/apex && git -C /path/to/apex apply $PWD/../../docs/env_patches/apex_setup.patch
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=6 uv pip install -v --no-deps --no-build-isolation \
    --config-setting="--build-option=--cpp_ext" --config-setting="--build-option=--cuda_ext" /path/to/apex
# Transformer Engine: ignore the missing _extra_state placeholder when loading the release checkpoint
(cd .venv/lib/python3.10/site-packages && for p in te_ops_op te_module_base; do patch -p1 < ../../../../../../docs/env_patches/$p.patch; done)
# weights (Hugging Face) and the NVIDIA base assets (needs `hf auth login` and accepting the NVIDIA model terms)
hf download Fleurrr/A2World-World-Model a2world-pretrained.pt a2world-libero.pt weights.json --local-dir a2world_ckpt
python scripts/download_base_models.py --checkpoint-dir checkpoints
a2world-env-check --require-cuda
```
The patches were verified: the two TE patches match the installed files, and `apex_setup.patch` applies cleanly to the pristine tag. The apex patch relaxes the nvcc-vs-torch minor-version check (nvcc 12.9 here, torch built with 12.6) and skips the extensions that do not compile with torch 2.6.

**B. Isaac Sim environment** (Python 3.11; order used here, with the `flatdict` workaround moved first, which avoids the failure we hit)
```bash
conda create -n env_isaaclab python=3.11 -y && conda activate env_isaaclab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab && cd ~/IsaacLab
pip install "setuptools<81" wheel && pip install flatdict==4.0.1 --no-build-isolation    # flatdict's build needs pkg_resources
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install none && pip install "packaging==23.0"
```
**C. Robot meshes** (not in git, licences unchecked): copy `assets/steve/meshes/ur5e` and `assets/steve/meshes/robotiq_2f_85` from the Steve project (github.com/220usamaahmed/steve_il, commit `e7592bb`) into `ur5e_sim/assets/meshes/`.

**D. Paths**: `ur5e_sim/run_main_finetune.sh` has the data and run directories (`/home/data/siddiquieu1/...`) hard-coded at the top; edit them for your machine.

*This repository has no remote yet; to let others clone it, push it to a host (the submodule URL is public).*

## How to reproduce
```bash
# 0. robot USD (once; needs ur5e_sim/assets/meshes, see Environment)
cd ur5e_sim && OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python convert_arm.py --headless

# 1. record episodes (held-out set: seed 0; training: seeds >= 1, ~33 s per episode)
python collect.py --headless --seed 0 --plan success:4,drop_on_table:3,release_off_edge:3,push_off_edge:3,miss_grasp:2 --out data/pilot
python collect.py --headless --seed 1 --first-index 100 --plan success:50,drop_on_table:40,release_off_edge:40,push_off_edge:40,miss_grasp:30 --out <data_root>/train   # GPU 0
python collect.py --headless --seed 2 --first-index 300 --plan success:50,drop_on_table:40,release_off_edge:40,push_off_edge:40,miss_grasp:30 --out <data_root>/train   # GPU 1
python make_check_video.py data/pilot/ep_* --out ../results/ur5e_pilot_check           # look at them

# 2. convert to the training layout (CPU), then validate with the repo's checker
python convert_to_train.py --episodes <data_root>/train --out <data_root>/train_processed
a2world-validate-data <data_root>/train_processed

# 3. fine-tune (GPU 0 + 1, ~29 h); bundle the pretrained weights first
cd ../A2World/world_model && python scripts/create_checkpoint_bundle.py --checkpoint a2world_ckpt/a2world-pretrained.pt --output <runs>/pretrained_bundle --mode symlink
../../ur5e_sim/run_main_finetune.sh        # edit the /home/data paths inside first

# 4. export a checkpoint to release format and evaluate (from A2World/world_model)
python -m scripts.export_release_checkpoint --checkpoint <run>/checkpoints/model/iter_000003600.pt --output <exports>/ur5e_iter3600_ema.pt --source ema --dtype bfloat16 --trust-checkpoint
CUDA_VISIBLE_DEVICES=0 python ../../eval/eval_ur5e_ft.py --checkpoint <exports>/ur5e_iter3600_ema.pt --episodes ../../ur5e_sim/data/pilot --out ../../results/ur5e_finetuned/iter3600
CUDA_VISIBLE_DEVICES=0 python ../../eval/eval_ur5e.py --episodes ../../ur5e_sim/data/pilot            # pretrained baseline (needs a2world-pretrained.pt)
```
The LIBERO experiment (`eval/eval_libero.py --hdf5 <file>`) needs a LIBERO demo file, e.g. `libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5` from the Hugging Face dataset `yifengzhu-hf/LIBERO-datasets` (deleted from this folder to save space).

## Things learned that are easy to get wrong
- **Temporal render ghost:** teleporting the cube after `env.reset()` leaves a faint ghost of it at its default position in the first frames, including frame 0, the conditioning frame. Fix: set the cube's pose in `default_root_state` before the reset (done in `collect.py`) and warm up 40 steps.
- **Cube settling:** a cube placed exactly on the table slides up to 2 cm in the first physics steps; the scripted grasp is planned from the *settled* pose, and the cube is dropped from 3 mm.
- **Orientation:** the repo's `a2world-prepare-libero` writes clips that are not upright (robot at the bottom, mirrored) while the model expects the robot at the top. `convert_to_train.py` writes upright frames.
- **Action format** (training): the loader multiplies raw `action[:6]` by 0.05 and maps the gripper `(1 - g) / 2` (raw -1 = open, +1 = closed). `convert_to_train.py` uses measured base-frame TCP deltas, `raw = delta / 0.018 m` (translation) and `delta / 0.1 rad` (rotation), chosen from histograms against LIBERO's.
- **Speed:** with history, EMA and self-forcing on, 2 x A6000 reach ~0.5 samples/s (32 s per iteration at global batch 16). Self-forcing starts at iteration 700, so the first 700 iterations looked twice as fast (18 s). Peak memory at batch 2 per GPU: 31 GB allocated.
- **Checkpoints** are 21 GB each (model + optimizer); the exported EMA weights are 4.7 GiB.
- The repo's `val` mode takes the last 1/16 of the *training* folder, so validation loss is not held out; the run used `trainer.run_validation=false`.

## Open items for the next stage
Test generalisation (new camera poses / lighting / table, longer horizons, other tasks), the 3-view variant (no released training path; would need a new dataset class), a counterfactual check that predictions follow the *actions* (same start, different actions), the pretrained-model control on the repo's DROID example, and deciding what to keep of the 24 GB final training checkpoint (resume state) next to the two 5 GB exports.
