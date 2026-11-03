# Plan: one-day A2World fine-tune on the simulated UR5e + Robotiq, using GPU 0 and GPU 1 only

> **Outcome (added 2026-11-03).** This plan was executed on 2026-10-31 / 11-01 (GPU 0 + GPU 1). The run used global batch 16 and **3,600 iterations (~58k samples), which took ~29 h, not the ~19 h estimated** in the revised numbers of section 10:
> the first 700 iterations ran at ~18 s each, but self-forcing switches on at iteration 700 (`self_forcing_warmup_steps=700`) and the steady-state speed is ~32 s per iteration (~0.5 samples/s, as in the pilot).
> Results on the 15 held-out episodes are in the top-level `README.md` (fine-tuned agent-view PSNR 28.1 dB vs 18.7 dB pretrained and 21.9 dB for copying frame 0). The decision rule of section 8 gives "clear improvement".
> Paths in this document refer to the layout before the clean-up; see the README for the current one.


Status: plan only, nothing in sections 3-8 has been run. Tags: **[measured]** from our runs, **[repo]** from the A2World code/docs,
**[paper]** from arXiv 2606.29501 (as summarised by a fetch tool; check against the PDF), **[estimate]** my own arithmetic, to be replaced by the pilot (step 6).

## 1. Goal and what this run can honestly show

Question: does a short fine-tune of the pretrained A2World model on UR5e sim data fix the failures we saw
(arm and gripper turn into smeared geometry, the cube is taken for another object, the cube never falls off the table), and what would the paper's full recipe cost on this hardware?

Baseline **[measured]** (pretrained checkpoint, 15 held-out episodes, 3 views, 60-160 steps, open loop): mean PSNR ext1 18.7 dB, wrist 14.0 dB, ext2 20.0 dB,
all below "copy frame 0" (21.9 / 16.4 / 23.5); the predicted cube left the table in only 2 of the 9 (episode, view) pairs where the real cube left the view.

The run produces evidence for the GPU request either way. If the model improves a lot in a day, the request is "we need N days of 4 GPUs for the full recipe"; if it barely moves, the
request is "a day on 2 GPUs covers ~7% of the paper's fine-tuning compute and was not enough". Do not decide the conclusion in advance; the decision rule is in section 8.

## 2. Configuration: what runs, and why this one

**Recipe: the released LIBERO fine-tune (`scripts/finetune_libero.sh`, experiment `a2world_finetune_libero`), started from `a2world-pretrained.pt`** [repo].
It is the only fine-tuning path in the release, and it is how the paper adapts to new data ("A2World-sim", 30k steps on LIBERO and on their custom Flexiv data [paper]).
Consequences:
- **2 views**, not 3: `agentview` = our `ext1` (camera embedding 2), `eye_in_hand` = our `wrist` (embedding 0) [repo]. `ext2` is unused (optional extra data, section 4).
  The fine-tuned model is therefore a different (history-aware, 2-view) variant than the 3-view pretrained model we evaluated, so compare it with copy-frame-0 and the pretrained ext1/wrist numbers, not view-for-view.
- Settings in the recipe [repo]: 256x256, 21-frame windows, 20 actions, 20-frame history, full-parameter, FSDP, fused Adam lr 1e-4 (weight decay 0.1), constant LR after a 1000-step warm-up, EMA on,
  self-forcing (probability 0.4 per iteration after a 2000-iteration warm-up).
- The history layers do not exist in the pretrained checkpoint; they start from random initialisation. With few iterations they may learn slowly (risk R3).

**Hardware layout:** `GPUS=2`, CUDA_VISIBLE_DEVICES=0,1, FSDP shards the model over both GPUs (`fsdp_shard_size=-1` in the config [repo]), **1 sample per GPU**, gradient accumulation 8, **global batch 16**
(the paper uses 24 [paper]).

**Memory** [estimate]: ~20 bytes/parameter x 2.51 B = ~50 GB of weights + gradients + fp32 master + Adam + EMA, sharded -> ~25 GB per GPU, plus ~6 GB activations per sample and overhead -> **~33-38 GB of 48 GB** per GPU.
Feasible but tight; there is no room for batch 2. Fallbacks if it runs out of memory, in this order: EMA off; `history_video_length` 20 -> 10; LoRA (caveat: the new history layers would stay frozen at random init, so use only with a design that has no history).
If even batch 1 does not fit on 2 GPUs, that is itself evidence for the GPU request.

**Throughput** [estimate]: inference measured 0.77 s per forward pass for the 2-view history model on one A6000 (27 s per 35 steps) [measured]; a training sample is ~3 forward passes (+ ~15% average for self-forcing after its warm-up)
-> ~2.3-2.7 s per sample per GPU -> ~0.75 samples/s on 2 GPUs -> **~55k samples in 20 h**.

## 3. Budget and timeline (training is capped at ~20 h)

| Step | What | Time |
|---|---|---|
| 1 | Collect ~400 training episodes (two Isaac Sim processes, GPU 0 and GPU 1, ~25 s per episode [measured: 15 episodes ~6 min]) | ~1.5-2 h |
| 2 | Convert to the training layout, validate, check orientation and action histograms | ~1 h |
| 3 | Bundle the pretrained checkpoint (symlink), decide output location (section 4a) | minutes |
| 4 | 100-iteration pilot: speed, peak memory, loss | ~30-60 min |
| 5 | **Main run: 3,000 iterations x global batch 16 = 48k samples**, validation + checkpoint every 500 | **~18-20 h** |
| 6 | Evaluate checkpoints on the 15 held-out pilot episodes (1 GPU, ~1 h per checkpoint) | ~2 h |

Total about 1.3 days elapsed, of which the training run is under a day. Iteration count is set to 3,000 so the LR schedule completes in the budget; set `trainer.max_iter` from the pilot's measured seconds per iteration (target: 18 h).
The pilot may change this number by 2x in either direction.

## 4. Data

**Held-out test set: the 15 existing pilot episodes** (seed 0, in `ur5e_sim/data/pilot`). They must not be used for training. New collection uses seeds >= 1 (the episode seed is `seed * 100003 + index`).

**Training set: ~400 episodes** [estimate; no data-scaling result exists in the paper for new robots]:

| kind | episodes | purpose |
|---|---|---|
| success | 100 | pick, carry, place |
| drop_on_table | 80 | gripper opens in the air; cube falls onto the table |
| release_off_edge | 80 | cube released over the table edge, falls to the floor |
| push_off_edge | 80 | cube swept off the table by the closed gripper |
| miss_grasp | 60 | gripper closes beside the cube |

Failures are 75% of the data on purpose: falling is the behaviour the pretrained model never produced. Each episode is ~130 steps. With 64-frame clips every 16 frames and a training window every 3 frames [repo],
one episode yields ~75 overlapping 21-frame windows (heavily correlated), so 400 episodes ~ 30k windows, and 48k training samples is ~1.6 passes over them.
If results look data-limited, double the data with `ext2` as a second "agentview" for the same episodes (cheap: no new simulation).

Collection command (per GPU, different `--seed` / `--first-index`; counts per process):
```
CUDA_VISIBLE_DEVICES=0 python collect.py --headless --seed 1 --first-index 100 --plan success:50,drop_on_table:40,release_off_edge:40,push_off_edge:40,miss_grasp:30 --out <data_root>/train
CUDA_VISIBLE_DEVICES=1 python collect.py --headless --seed 2 --first-index 300 --plan success:50,drop_on_table:40,release_off_edge:40,push_off_edge:40,miss_grasp:30 --out <data_root>/train
```
Scene, cameras and camera poses stay fixed, matching the test set, so this experiment measures in-distribution adaptation only (not generalisation to new scenes or real robots).

### 4a. Disk: a blocker to decide first
`/home/user` has **39 GB free** (99% full) [measured]. The raw data (~17 MB per episode as npz -> ~7 GB) fits, but **training checkpoints do not**: with optimizer state and EMA a full checkpoint is probably 15-30 GB [estimate], and the plan saves 6.
The run writes to `$IMAGINAIRE_OUTPUT_ROOT` (default: `world_model/checkpoints` on `/home/user`) [repo]. Needed: a directory on a larger disk, e.g. the shared pool `/home/data` (9.6 TB free, world-writable, per-user directories; `/home/data/siddiquieu1` does not exist yet).
I have not created anything there; this needs your decision (also for the training data and the pretrained-checkpoint bundle). Also keep only the last ~2 checkpoints.

## 5. Data conversion (new code: a small converter, about one screen)

`a2world-prepare-libero` reads LIBERO HDF5 and, in my test, writes clips that are **not upright** (robot at the bottom, mirrored) while the repo's example frames and the model's inputs are upright (robot at the top) [measured].
So write our own converter from `data.npz` to the layout in `docs/DATA.md` [repo]: `clips/agentview_<id>.mp4` (= ext1, upright as rendered), `clips/eye_in_hand_<id>.mp4` (= wrist), `metadata/agentview_<id>.npz`, `dataset.json`.
Clips: 64 frames, stride 16, 10 fps label. Episodes shorter than 64 frames are skipped by the loader (ours are 77+ steps).

Metadata keys per clip [repo, `a2world/prepare_libero.py`]: `actions [T,2,7]` (arm 0 filled, arm 1 zeros), `end_position [T,2,3]`, `end_orientation [T,2,4]` (quaternion xyzw, arm 1 = identity), `effector_position [T,2,2]`, `start_frame`, `clip_len`, `fps`.

**Action convention (the main thing to get right)**: the loader multiplies the first six action values by 0.05, remaps the gripper `(1 - g) / 2` (raw -1 = open, +1 = closed -> 1 = open), and treats them as base-frame servo deltas (as LIBERO controller commands) [repo].
So per step, from the **measured** TCP poses: `raw_pos = clip(delta_pos_base / 0.0135 m, -1, 1)`, `raw_rot = clip(delta_rotvec_base / 0.3 rad, -1, 1)`, `raw_gripper = +1 closed / -1 open` from the commanded finger angle (threshold 0.375 rad).
The scales are my choice [estimate]: LIBERO's raw translation 95th percentile is ~0.9 while its per-step end-effector motion is ~9 mm mean / 14 mm 95th percentile [measured on 50 LIBERO demos], which gives ~0.0135 m per unit;
our carry speed is 8-12 mm per step. The rotation scale is a guess (LIBERO rotations are small: median ~0.03, 95th percentile 0.1-0.2). **Check in step 2:** plot the histograms of our raw actions against LIBERO's and adjust the scales so they overlap.
Multiplied by 0.05 the values land at ~0.045 at most, the same order as the metric deltas the pretrained model saw (DROID example peak ~0.02-0.05), which is why this mapping is a sensible starting point.

Validation before any training: `a2world-validate-data <root>`; view frame 0 of a converted clip next to `examples/assets/libero/agentview.png` (robot at the top); decode one clip and compare against the source frames.

## 6. Pilot (do before committing a day)

100 iterations, 2 GPUs, with overrides from the README pattern [repo]:
```
export A2WORLD_LIBERO_DATA=<data_root>/train_processed
export A2WORLD_PRETRAIN_CKPT=<bundle>        # python scripts/create_checkpoint_bundle.py --mode symlink ...
export IMAGINAIRE_OUTPUT_ROOT=<big_disk>/a2world_runs
CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/finetune_libero.sh \
  trainer.max_iter=100 dataloader_train.batch_size=1 trainer.grad_accum_iter=8 checkpoint.save_iter=0 \
  model.config.self_forcing_warmup_steps=0 trainer.run_validation=false
```
Record: seconds per iteration with and without self-forcing, peak GPU memory (`nvidia-smi` / trainer log), loss going down, and that the dataloader does not skip samples (the dataset warns and substitutes a random sample on errors [repo]; a flood of warnings means a format problem).
Open questions the pilot answers: does `trainer.grad_accum_iter` take effect with this trainer; where "val" samples come from (`mode="val"` reads the same folder; make sure the held-out pilot episodes stay out of it); that the checkpoint loads the pretrained weights with the new history layers missing without error.

Go / no-go: if memory or speed is far from section 2, change the plan (iterations, history length, EMA) before the main run.

## 7. Main run

```
CUDA_VISIBLE_DEVICES=0,1 GPUS=2 ./scripts/finetune_libero.sh \
  trainer.max_iter=3000 dataloader_train.batch_size=1 trainer.grad_accum_iter=8 \
  checkpoint.save_iter=500 trainer.validation_iter=500 trainer.max_val_iter=20 \
  model.config.self_forcing_warmup_steps=1000
```
(`self_forcing_warmup_steps` lowered from 2000 so self-forcing is active for the last two thirds; keep the default if the pilot shows it costs much more than estimated.)
Monitor the loss (logged every 10 iterations) and the validation loss. Run it under `nohup`/`tmux`, outputs on the big disk. Stop early only on a clear divergence (NaN, loss rising for several hundred iterations).

## 8. Evaluation and decision rule

Run the existing eval (`eval_ur5e.py`, to be adapted to `--variant libero`, 2 views, servo-format actions from the same converter) on the **15 held-out pilot episodes** with the iteration-1500 and the final checkpoints (and the EMA weights).
Also run the unmodified `a2world-libero.pt` (Panda, LIBERO) once as a second baseline.

Report, per episode kind and overall:
1. PSNR / SSIM / MAE per step vs the copy-frame-0 baseline, ext1 and wrist.
2. Cube-fall agreement: for the 6 falling episodes, in how many (episode, view) pairs does the predicted cube also leave the view (baseline: 2 of 9).
3. Qualitative: whether the arm keeps its structure at step 60-80, and the cube is not turned into another object (side-by-side videos).
4. Seconds per iteration and peak memory from the training log, and the projected time for the paper's recipe: 30k steps x batch 24 = 720k samples ~ 960k s ~ **11 days on these 2 GPUs, ~5.5 days on 4** [estimate, +/-2x].

Proposed reading (agree on it before looking at results): **clear improvement** = PSNR above copy-frame-0 on ext1 and wrist at horizons >= 40 steps AND cube-fall agreement >= 5 of 9 pairs AND intact arm geometry at step 60;
**partial** = better than the pretrained model but below one of those bars; **no improvement** = within about 1 dB of the pretrained ext1/wrist numbers and no falling cubes.

## 9. Risks

- **R1 Memory**: batch 1 may not fit on 2 x 48 GB (~33-38 GB estimated). Pilot first; fallbacks in section 2.
- **R2 Disk**: checkpoints need the larger disk (4a).
- **R3 Short run**: the history layers are new, the paper trains 15x more samples [paper]. A weak result may reflect the budget and not the approach; say so in the report.
- **R4 Action scale / frame**: a wrong scale gives plausible videos that ignore the commands. Check with the histogram comparison and, after training, a counterfactual (same start, different actions).
- **R5 Orientation / format**: the repo's converter output was not upright; our own converter must be checked visually.
- **R6 Fidelity**: the cube slides slightly in the grasp in some episodes and the scene is a bare block; the held-out episodes share the camera setup, so improvement does not show real-robot transfer.
- **R7 Shared machine**: GPUs 2 and 3 are used by others; stay on 0 and 1. `/tmp/isaaclab` belongs to another user; Isaac Lab logs go to `ur5e_sim/logs/`.
- **R8 Licence**: checkpoints are under the NVIDIA Open Model License.

## 10. Pilot results (2026-10-31) and changes to the plan

All **[measured]** on GPU 0 + GPU 1, 2 x RTX A6000, FSDP over both, self-forcing active from iteration 0 (probability 0.4), EMA on, history on:

| setting | seconds / iteration (steady) | samples / s | peak memory per GPU (allocated / reserved) |
|---|---|---|---|
| batch 1 per GPU (global 2) | 4.6 (range 3.4-5.7) | 0.43 | 26.8 / 29.9 GB |
| batch 2 per GPU (global 4) | 8.1 (range 7.8-8.7) | 0.49 | 31.1 / 34.9 GB |

- Throughput is about **0.49 samples/s, roughly 35% below the 0.75 estimated in section 2**. Memory is better than estimated (batch 2 fits with ~13 GB to spare).
- Iteration time varies because self-forcing runs in ~40% of iterations.
- Checkpoint save: ~20 s and **21 GB** per checkpoint (model 15 GB + optimizer). Export with `scripts/export_release_checkpoint.py --source ema --dtype bfloat16` gives a 4.7 GiB release-format file that `a2world-checkpoint` recognises as the `libero` variant, so evaluation with `--variant libero` is possible.
- The loss over the first 100 iterations is too noisy (0.16-1.1) to say anything.

Environment problems found and fixed on this machine (all needed to train at all):
1. **NCCL peer-to-peer between GPU 0 and GPU 1 hangs** (an all-reduce never returns); `NCCL_P2P_DISABLE=1` fixes it. GPUs 0/1 and 2/3 share a PCIe host bridge, the pairs are connected via the CPU interconnect (`nvidia-smi topo -m`). Communication goes through host memory, which is part of why throughput is lower.
2. apex `FusedAdam` needs compiled CUDA extensions (my earlier Python-only apex install was not enough for training). Built `amp_C` and `apex_C` only, for sm_86, with nvcc 12.9 against torch cu126 (the stock build check rejects the minor-version mismatch; I relaxed it to the major version in a local copy of the apex source at `ur5e_sim/tmp/apex`; other apex extensions do not compile with torch 2.6).
3. Transformer Engine 1.13 fails to load the stripped release checkpoint through torch's checkpoint loader (a placeholder for `_extra_state`); patched two functions in the `.venv` copy of TE to ignore the placeholder (the state is FP8 metadata, unused in bf16). Originals saved in `ur5e_sim/tmp/te_*.orig`.

Revised numbers (replace sections 2, 3 and 7):
- Layout: 2 GPUs, **batch 2 per GPU, gradient accumulation 4 -> global batch 16**.
- Budget 20 h x 0.49 samples/s ~ **35k samples ~ 2,200 iterations** (not 3,000 / 48k): ~5% of the paper's 720k samples.
- Full recipe (720k samples) at this speed: ~**17 days on these 2 GPUs**; about 8-9 days on 4 if scaling is linear (not verified; the P2P problem may limit it).
- Validation off (the repo's validation reads the training folder, so it would not be held-out and the path is untested); checkpoint every 500 iterations (21 GB each).
- Not verified yet: that `trainer.grad_accum_iter=4` multiplies the iteration time by ~4 (check in the first iterations of the main run: expect ~32 s per iteration), and that the `scheduler.warm_up_steps` override takes effect.
