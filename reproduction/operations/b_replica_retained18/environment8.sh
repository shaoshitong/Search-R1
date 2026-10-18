#!/bin/bash
# Source this inside the existing private retained18 namespace. It sets a
# process-local environment only; it never reserves GPUs or starts a command.
export SEARCH_G1_PHYSICAL_WORLD_SIZE=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
