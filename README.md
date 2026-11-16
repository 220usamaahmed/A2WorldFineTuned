# A2World fine-tuning on a simulated UR5e

A UR5e + Robotiq 2F-85 in Isaac Sim picks objects from drawers and a shelf and places them in a container
(`ur5e_sim/`). The A2World world model (`A2World/`, git submodule) is fine-tuned on those trajectories (`finetune/`).

## Setup

- **Simulation:** Isaac Sim 5.1, Isaac Lab 2.3.2 and cuRobo, in one Python environment.
- **Fine-tuning:** A2World's own environment, `A2World/world_model/.venv` (see `A2World/README.md`). The patches in `docs/env_patches/` are needed to train.
- **Data folder:** large files go under `$A2W_DATA` (default `./data`). Set it in your shell profile:
  ```bash
  export A2W_DATA=/home/data/siddiquieu1
  ```

Do this once on a new machine. On the current machine it is already done. Run from `ur5e_sim/setup/`:
```bash
python fetch_assets.py --list assets.txt                               # Isaac assets -> $A2W_DATA/assets
OMNI_KIT_ACCEPT_EULA=YES python export_ycb_meshes.py --headless
OMNI_KIT_ACCEPT_EULA=YES python make_ycb_physics.py --headless
OMNI_KIT_ACCEPT_EULA=YES python convert_arm.py --headless --simple     # robot USD
OMNI_KIT_ACCEPT_EULA=YES python build_pullbin.py --headless            # drawer USD
python ../scene/looks.py                                               # textures and light map
```

## Run one trajectory (to view it)

From `ur5e_sim/scene/`:
```bash
OMNI_KIT_ACCEPT_EULA=YES python pick_place.py --headless --episodes 1
```
This writes a video to `$A2W_DATA/episodes/ep00.mp4`.

To watch the run live instead, use `--livestream 2 --no-video`. Connect with the Isaac Sim WebRTC Streaming Client to the server's IP.

## Collect the dataset

Each episode picks either from a drawer or from the shelf (alternating), with random objects:
```bash
OMNI_KIT_ACCEPT_EULA=YES python pick_place.py --headless --episodes 200 --seed 1 --out $A2W_DATA/episodes/train
```
Use a different `--seed` for each batch.

> Not done yet: `pick_place.py` saves only a preview video per episode. It does not yet save the per-step data (`data.npz`) that `convert_to_train.py` needs.

## Fine-tune

```bash
# 1. episodes -> A2World training format
python finetune/convert_to_train.py --episodes $A2W_DATA/episodes/train --out $A2W_DATA/train_processed

# 2. starting weights (once), from A2World/world_model/ with its .venv active
hf download Fleurrr/A2World-World-Model a2world-pretrained.pt --local-dir a2world_ckpt
python scripts/create_checkpoint_bundle.py --checkpoint a2world_ckpt/a2world-pretrained.pt \
    --output $A2W_DATA/a2world_runs/pretrained_bundle --mode symlink

# 3. train (2 GPUs with about 31 GB each; it does not fit on a 16 GB card)
finetune/run_finetune.sh
```
