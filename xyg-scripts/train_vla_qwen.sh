#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini


# Next we train MiniVLA based on this Prism base VLM on LIBERO-90.
# Run from the root of the repository
# LIBERO_DATA_ROOT=/mnt/nfs/CMG/xiejunlin/datasets/Robotics/libero
LIBERO_DATA_ROOT=dataset
export PRISMATIC_DATA_ROOT='/mnt/nfs/CMG/xiejunlin/datasets/Robotics/libero'
export CUDA_VISIBLE_DEVICES="0,1"
LOG_ROOT=./logs/libero_qwen_pretrain_test
WANDB_PROJECT="libero_qwen1"
WANDB_ENTITY="1207481522" # should be you user name or team name in w&b account

# torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
#   --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
#   --data_root_dir $LIBERO_DATA_ROOT \
#   --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
#   --wandb_project "<PROJECT>" \
#   --wandb_entity "<ENTITY>"

export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29500'

# torchrun --standalone --nnodes 1 --nproc-per-node 1 --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} vla-scripts/train.py \
#   --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
#   --data_root_dir $LIBERO_DATA_ROOT \
#   --run_root_dir $LOG_ROOT \
#   --wandb_project $WANDB_PROJECT \
#   --wandb_entity $WANDB_ENTITY \
#   --hf_token "HF_TOKEN_R"

# torchrun --standalone --nnodes 1 --nproc-per-node 1 --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} vla-scripts/train.py \
#   --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
#   --data_root_dir $LIBERO_DATA_ROOT \
#   --run_root_dir $LOG_ROOT \
#   --wandb_project $WANDB_PROJECT \
#   --wandb_entity $WANDB_ENTITY \
#   --hf_token "HF_TOKEN_R"

WORLD_SIZE=2
CKPT_PATH='/home/xingyouguang/.cache/huggingface/hub/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498'
torchrun --standalone --nnodes 1 --nproc-per-node "${WORLD_SIZE}" --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --vla.base_vlm "${CKPT_PATH}" \
  --vla.expected_world_size "${WORLD_SIZE}" \
  --vla.global_batch_size 4 \
  --vla.per_device_batch_size 2 \
  --data_root_dir "${LIBERO_DATA_ROOT}" \
  --run_root_dir "${LOG_ROOT}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --is_resume False