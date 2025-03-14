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

LOG_ROOT=./logs/libero_qwen
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

CKPT_PATH='/home/xingyouguang/.cache/huggingface/hub/models--Stanford-ILIAD--minivla-libero90-prismatic/snapshots/4289f87e8e00706e188c7a3a61fc6e7d72ab2564/checkpoints/step-122500-epoch-55-loss=0.0743.pt'
torchrun --standalone --nnodes 1 --nproc-per-node 8 --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --data_root_dir "${LIBERO_DATA_ROOT}" \
  --run_root_dir "${LOG_ROOT}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --hf_token "HF_TOKEN_R" \
  --pretrained_checkpoint "${CKPT_PATH}" \
  --is_resume False