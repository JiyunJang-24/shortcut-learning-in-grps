#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini


# We train this transformer backbone using the Llava-1.5-Instruct Visual Question Answering (VQA) dataset, 
# the same dataset used for training the base Prismatic VLM in OpenVLA. 

# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "prism-qwen25-extra-dinosiglip-224px+0_5b" \