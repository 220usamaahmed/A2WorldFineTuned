#!/usr/bin/env bash
# Fine-tune A2World from a2world-pretrained.pt on the converted UR5e data (convert_to_train.py), on 2 GPUs.
# 2 GPUs x batch 2 x gradient accumulation 4 = global batch 16; needs ~31 GB per GPU.
# Inputs under $A2W_DATA: train_processed/ (the data), a2world_runs/pretrained_bundle/ (the starting weights).
# Output: $A2W_DATA/a2world_runs/runs/
set -euo pipefail
: "${A2W_DATA:?set A2W_DATA to the data folder}"
WM="$(cd "$(dirname "$0")/../A2World/world_model" && pwd)"
R=$A2W_DATA/a2world_runs
cd "$WM" && source .venv/bin/activate
export A2WORLD_LIBERO_DATA=$A2W_DATA/train_processed
export A2WORLD_PRETRAIN_CKPT=$R/pretrained_bundle
export IMAGINAIRE_OUTPUT_ROOT=$R/runs
export WANDB_MODE=disabled MASTER_PORT=29541
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} GPUS=2
export NCCL_P2P_DISABLE=1   # peer-to-peer between the two GPUs hung on the A6000 machine
./scripts/finetune_libero.sh \
  trainer.max_iter=3600 \
  dataloader_train.batch_size=2 trainer.grad_accum_iter=4 \
  checkpoint.save_iter=500 trainer.run_validation=false \
  model.config.self_forcing_warmup_steps=700 \
  scheduler.warm_up_steps=[200] \
  job.name=ur5e_main
