# Plan: fine-tuning A2World for a UR5e + Robotiq parallel gripper (Isaac Sim)

Status: plan only, nothing here has been run. Items marked **[measured]** come from our LIBERO experiments, **[repo]** from reading the A2World code/docs, and **[estimate]** are my guesses and should be checked with the pilot steps below.

## 1. Goal and success criteria

Fine-tune A2World so that, given the first camera frames of a UR5e pick-and-place scene and a sequence of end-effector actions, it predicts the future video, including **failed** attempts (missed grasp, slipped or dropped cube).

Success looks like:
- Predicted arm motion follows the commanded actions (right direction and magnitude).
- Predicted video beats the "copy first frame" baseline by a clear margin, as the LIBERO checkpoint does on LIBERO [measured: agent view 23.9 vs 22.3 dB, wrist view 20.8 vs 13.0 dB over 80 steps].
- On held-out **failed** episodes, the cube ends up where the real one did (not lifted, dropped, knocked over), not where a successful episode would have put it.

## 2. Starting point and what the repo gives us

- Start from `a2world-pretrained.pt` (multi-robot real-robot pretraining), not `a2world-libero.pt`, which is adapted to the LIBERO Panda only. We have only the LIBERO checkpoint locally; the pretrained one (5 GB) still has to be downloaded from `Fleurrr/A2World-World-Model`.
- The repo's fine-tune recipe is `scripts/finetune_libero.sh` -> experiment `a2world_finetune_libero` [repo]. It is a **full-parameter** fine-tune with FSDP, producing the history-aware A2World-sim variant that we already know how to run and evaluate (`--variant libero`).
- Its data settings [repo]: 2 views (agent view + wrist view), 256x256, 21-frame windows conditioned on 20 actions, a 20-frame history, batch size 4 per GPU, 30,000 iterations by default, checkpoint every 2,000.
- The training data loader reads the "processed LIBERO layout" (section 5). The cheapest path is to produce exactly that layout from Isaac Sim, so no model code changes are needed.
- `a2world-prepare-libero` is not usable as-is for our data. It reads LIBERO HDF5 and, in my test, its output clips are **not upright** (robot at the bottom of the image, mirrored), whereas the repo's bundled example frames and the frames I fed the model in evaluation have the robot at the **top** [measured]. The loader does not flip images [repo]. So: write our own converter, and check the orientation of the output by eye against `examples/assets/libero/agentview.png` before training.

## 3. Data collection (Isaac Sim)

### Scene
- UR5e with a Robotiq 2F-85 (or whichever parallel gripper you use) mounted on a table, with 2-4 small cubes in distinct colours (distinct colours let us track cubes automatically in generated video for evaluation).
- Randomise per episode: cube positions and yaw, which cube is the target, and the place target (a marked spot or second cube). Add mild lighting variation. Keep the **camera poses fixed** at first (see section 4).

### Policy used to collect
- A scripted pick-and-place controller using Cartesian waypoints and IK (differential IK or RMPflow in Isaac Sim), with small noise on waypoints so trajectories are not identical. Check the Isaac Sim asset browser for the UR5e and Robotiq 2F-85 assets.
- Record the commanded **end-effector delta** at every step (see section 5), plus the absolute end-effector pose and gripper state.

### Successful trajectories
- Approach, grasp, lift, move, place, release. About 100 steps each (LIBERO episodes average 101 steps [measured]).

### Failed trajectories
The model is only conditioned on actions and images, so a failure has to be visible in the pixels and caused by the actions. Generate failures by injecting the cause in the scripted policy, not by labelling:
- grasp offset too large (gripper closes beside or over the cube),
- gripper closes too late or too early,
- gripper opens during transport (drop),
- lift too fast or cube clipped on the way (cube slips or is knocked),
- place position wrong (cube lands off target).

Keep roughly 25-30% of episodes as failures, with a mix of failure types. Store a `success` flag and `failure_type` per episode in a side file (not used for training, used for evaluation splits).

### Motion scale
Per-step end-effector motion in LIBERO is about 9 mm on average (95th percentile 14 mm) [measured]. Pick your control rate so UR5e per-step motion is in a similar range. If the arm moves much more per frame than that, the model will see motion unlike its training data.

## 4. Camera setup

Match the two views the LIBERO-style checkpoint uses:

| View | Camera embedding ID | Placement |
|---|---|---|
| `agentview` | 2 | Fixed third-person camera across the table looking toward the robot, tilted down ~30-45 deg so the base, the arm and the whole workspace are visible. The robot base should appear at the **top** of the image, as in the bundled example. |
| `eye_in_hand` (wrist) | 0 | Mounted on the tool flange looking along the approach direction, with the gripper fingers visible at the bottom of the frame (as in LIBERO). |

- Render at **256x256** directly (the model works at 256x256; LIBERO's 128x128 had to be upscaled).
- RGB only, no depth needed.
- One rendered frame per action step. Encode as H.264 mp4 (`yuv420p`).
- Keep camera intrinsics and extrinsics constant across the dataset. Varying them is a later experiment.
- Orientation check: view frame 0 of an episode next to `examples/assets/libero/agentview.png`. Both should have the robot at the top.

## 5. Data format

Target: the repo's processed layout [repo, `docs/DATA.md`]:

```text
ur5e_processed/
├── clips/
│   ├── agentview_<id>.mp4
│   └── eye_in_hand_<id>.mp4        # same <id> as the agentview clip
├── metadata/
│   └── agentview_<id>.npz
└── dataset.json                    # [{"caption": "...", "media_path": "clips/agentview_<id>.mp4"}, ...]
```

Keep the file names `agentview_` and `eye_in_hand_`: the loader finds the wrist clip by that naming [repo].

Each metadata `.npz` has the keys `a2world-prepare-libero` writes [measured on a converted clip]:

| Key | Shape | Meaning |
|---|---|---|
| `actions` | `[T, 2, 7]` | raw actions, arm 0 filled, arm 1 zeros (see below) |
| `end_position` | `[T, 2, 3]` | absolute end-effector position, arm 1 zeros |
| `end_orientation` | `[T, 2, 4]` | quaternion, SciPy `xyzw` order; arm 1 set to identity `[0,0,0,1]` |
| `effector_position` | `[T, 2, 2]` | gripper state (LIBERO stores two finger positions; for the Robotiq store opening width as a consistent 2-vector) |
| `start_frame`, `clip_len`, `fps` | scalars | clip bookkeeping (`clip_len` must equal the video frame count; `a2world-validate-data` checks this) |

Clip settings used for LIBERO: 64-frame clips, stride 16, 10 fps label [repo defaults]. The training window is 21 frames (20 actions).

### Action convention
The loader applies the LIBERO "servo" convention [repo `docs/DATA.md`]:
- `actions[:, 0, :6]` = end-effector delta (x, y, z, roll, pitch, yaw) in the **robot base frame**, normalised, then multiplied by 0.05 inside the loader.
- `actions[:, 0, 6]` = gripper command **-1 = open, +1 = close**, remapped by the loader to 1 = open, 0 = closed.
- Action `t` moves the robot from observation `t` to `t+1`.

Open point: LIBERO's raw actions are controller commands in [-1, 1] (translation up to ~0.94, rotations mostly below 0.2, 95th percentile ~0.1-0.12 [measured]). For the UR5e you have to define an equivalent normalised command: `raw = clip(delta_ee / scale, -1, 1)`. Choose `scale` so the **distribution of raw values resembles LIBERO's** (translation spread over most of [-1, 1] on fast moves, rotations small). I can compute and compare the histograms once you have a pilot batch. A wrong scale is the most likely silent failure: the video will look plausible but not follow the commands.

### Split
Hold out ~10% of **episodes** (not clips) for validation and test, stratified by success/failure and failure type. Never split clips of one episode across train and test.

## 6. How many trajectories

[estimate], not from the paper:

| Stage | Episodes | Purpose |
|---|---|---|
| Pilot | 50 (35 success + 15 failure) | Test the pipeline end to end: orientation, action scale, converter, `a2world-validate-data`, a 100-iteration training run |
| First real dataset | ~500 (350-400 success, 100-150 failure) | First fine-tune. ~100 steps per episode -> ~50k frames, ~13k training windows (train windows start every 3 frames) |
| If results are weak | ~1000-1500 | Add data where it fails (failure types, cube positions), not just more of the same |

A single narrow task (cubes on a table, fixed cameras) needs far less data than a general model. The risk is overfitting to fixed positions, so keep cube positions and target choices well spread. Failure examples are the scarce, valuable part: if the model sees only a few failures of each type it will tend to predict success.

## 7. Fine-tuning compute

### Requirements
- Full-parameter training of a 2B model with an EMA copy, using FSDP. Weights, gradients, Adam state and EMA in fp32 are roughly 8 + 8 + 16 + 8 GB ~ 40 GB before activations [estimate]. That **will not fit on one 48 GB A6000**, so plan for **at least 2, preferably all 4, A6000s**. This needs permission to use GPUs 1-3 (you asked me to use GPU 0 only for the evaluation).
- The repo's README example uses 8 GPUs (`GPUS=8`). We have 4. Use a smaller per-GPU batch if memory is tight (`dataloader_train.batch_size=1` or `2`, optionally `trainer.grad_accum_iter`).

### Ballpark time [estimate, +/-2x]
Inference on one A6000 took ~0.7 s per forward pass of a 2-view chunk (27 s for 35 steps) [measured]. A training step costs about 3x a forward pass, so roughly **2 s per training sample per GPU**, or ~2 samples/s on 4 GPUs.

| Run | Iterations x global batch | Samples | Wall-clock on 4 x A6000 |
|---|---|---|---|
| Pilot | 100 x 4 | 400 | ~5-10 min (mainly to measure real speed and memory) |
| Small fine-tune | 3,000 x 16 | 48k | ~7 h |
| Main fine-tune | 5,000-10,000 x 16 | 80-160k | ~11-22 h |
| Repo default | 30,000 x 32 (8 GPUs, batch 4) | 960k | ~5-6 days (not planned) |

A narrow dataset should converge in far fewer iterations than the repo default, which was set for the whole LIBERO suite. Use the validation loss and the open-loop metrics below to decide when to stop, and checkpoint every 1,000-2,000 iterations. Plan on **about a day of GPU time** for the first full fine-tune.

The very first thing to do is the 100-iteration pilot (the README shows `trainer.max_iter=100 dataloader_train.batch_size=1 checkpoint.save_iter=0`). It gives the real seconds per iteration and peak memory, and replaces the estimates above.

## 8. Evaluation

Reuse `eval_libero.py` (open-loop, autoregressive, same checkpoint variant) with an Isaac Sim data reader:

1. **Pixel metrics vs ground truth**: PSNR, SSIM, MAE per step on held-out episodes, plus the copy-first-frame baseline, for both views. Report success and failure episodes separately.
2. **Action following**: track the gripper (or arm) position in generated vs real video and compare the trajectories.
3. **Cube outcome**: segment the coloured cubes by colour in generated and real frames and compare cube position and lift height over time. This is the quantitative check for "did the model learn failures". Report, for failed episodes, how often the predicted final cube state matches the real one.
4. **Counterfactual test**: from one initial scene, run the successful action sequence and a failed one. The two predicted videos should end differently.
5. **Zero-shot baseline** (before fine-tuning): run `a2world-pretrained.pt` with `--variant pretrained` on 3-5 recorded episodes, to show what fine-tuning buys. The pretrained pipeline takes 3 views in the repo examples, so how it handles two views is untested; try it.
6. **Timing**: seconds per chunk and per frame, as measured for LIBERO (27 s per 20-step chunk, ~0.68 s per single-view frame) [measured].

## 9. Steps and order

1. Download `a2world-pretrained.pt`; (optional) run the zero-shot baseline from section 8.5.
2. Build the Isaac Sim scene and the scripted collector; record 50 pilot episodes with the camera setups from section 4.
3. Write the converter to the processed layout; run `a2world-validate-data`; look at frames against the bundled example (orientation!); compare action histograms with LIBERO's and fix `scale`.
4. Run the 100-iteration training pilot; record seconds per iteration and peak memory; decide batch size and iteration count.
5. Collect ~500 episodes (with ~25-30% failures); split by episode.
6. Fine-tune (`create_checkpoint_bundle.py` on the pretrained checkpoint, then `finetune_libero.sh` with `A2WORLD_LIBERO_DATA` pointing at the new dataset and overrides for `GPUS`, batch size, `max_iter`). Check intermediate checkpoints.
7. Evaluate (section 8), look at failures of the model, and collect more data where it is weak.

## 10. Risks

- **Action scale or frame mismatch**: output looks fine but ignores commands. Mitigation: histogram comparison (step 3) and the counterfactual test.
- **Orientation or colour channel mismatch**: the repo's converter output was upside-down relative to the bundled example. Mitigation: visual check of every new dataset.
- **Memory**: the full fine-tune might not fit on 4 x 48 GB at the default batch size. Mitigation: the pilot, smaller batch, gradient accumulation.
- **Failures not learned**: too few or too uniform failures. Mitigation: more failure variety, evaluation item 3.
- **Sim-only data**: the model may learn Isaac Sim rendering quirks and not transfer to a real UR5e. This plan only addresses the simulated robot.
- **Licence**: the checkpoints are governed by the NVIDIA Open Model License; check that your use is permitted.
