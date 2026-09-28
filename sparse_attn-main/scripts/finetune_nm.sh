#!/bin/bash

export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export MASTER_PORT=29501
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export OMP_NUM_THREADS=12

# NOTE: We no longer parse positional arguments here.
# All hyperparameters are passed as named flags ($@) from the submit script.

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "--- Starting run_learnable_mask_args.sh ---"
echo "  Num GPUs    : ${NUM_GPUS}"
echo "  Extra args  : $@"

PYTHON_SCRIPT="main.py"

torchrun --nproc_per_node="${NUM_GPUS}" --rdzv_endpoint="localhost:29500" "${PYTHON_SCRIPT}" \
    "$@" \
    --evaluate_perplexity \
    --wandb \
    --train \
    --test_lmharness
    # --grad_checkpoint \
    
