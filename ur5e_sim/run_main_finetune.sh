#!/usr/bin/env bash
# Main fine-tune run: A2World-sim from a2world-pretrained.pt on the simulated UR5e data, GPUs 0 and 1 only.
# 2 GPUs x batch 2 x gradient accumulation 4 = global batch 16; 3,600 iterations (~19 h at the 18.7 s per iteration measured in the first run).
set -euo pipefail
WM="$(cd "$(dirname "$0")/../A2World/world_model" && pwd)"
R=/home/data/siddiquieu1/a2world_runs
cd "$WM" && source .venv/bin/activate
export A2WORLD_LIBERO_DATA=/home/data/siddiquieu1/a2world_ur5e/train_processed
export A2WORLD_PRETRAIN_CKPT=$R/pretrained_bundle
export IMAGINAIRE_OUTPUT_ROOT=$R/runs
export WANDB_MODE=disabled MASTER_PORT=29541
export CUDA_VISIBLE_DEVICES=0,1 GPUS=2
export NCCL_P2P_DISABLE=1   # peer-to-peer between GPU 0 and GPU 1 hangs on this machine
./scripts/finetune_libero.sh \
  trainer.max_iter=3600 \
  dataloader_train.batch_size=2 trainer.grad_accum_iter=4 \
  checkpoint.save_iter=500 trainer.run_validation=false \
  model.config.self_forcing_warmup_steps=700 \
  scheduler.warm_up_steps=[200] \
  job.name=ur5e_main
