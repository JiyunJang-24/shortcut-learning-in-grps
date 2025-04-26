#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PRISMATIC_DATA_ROOT="${LIBERO_DATA_ROOT}"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29520'
conda activate openvla-mini


# Next we train MiniVLA based on this Prism base VLM on LIBERO-90.
# Run from the root of the repository
# LIBERO_DATA_ROOT=/mnt/nfs/CMG/xiejunlin/datasets/Robotics/libero
DATA_MIX="minivla-spatial-split-dataset-200300-700800"
# 判断路径是否存在, LIBERO_DATA_ROOT
LIBERO_DATA_ROOT="/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island_split_rlds/xyg_20_10_-10.0_90.0"
if [ -e ${LIBERO_DATA_ROOT} ] ; then 
    echo "LIBERO_PATH=${LIBERO_DATA_ROOT}"
else 
    # bash将字符串中的 hdd3 替换为 hdd2
    LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT/hdd3/hdd2}
    echo "LIBERO_PATH=${LIBERO_DATA_ROOT}"
fi

LOG_ROOT=libero_minivla_split_large_distance_20_10
WANDB_PROJECT="libero_minivla_split_large_distance_20_10"
WANDB_ENTITY="1207481522" # should be you user name or team name in w&b account

max_steps=10000
WORLD_SIZE=8
BATCH_SIZE=16
CUDA_VISIBLE_DEVICES_LIST="0,1,2,3,4,5,6,7"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"

CKPT_PATH='/home/xingyouguang/.cache/huggingface/hub/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498'
OMP_NUM_THREADS=4 torchrun --standalone --nnodes 1 --nproc-per-node "${WORLD_SIZE}" --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --vla.data_mix "${DATA_MIX}" \
  --vla.base_vlm "${CKPT_PATH}" \
  --vla.action_tokenizer libero_vq_extra_action_tokenizer \
  --vla.expected_world_size "${WORLD_SIZE}" \
  --vla.global_batch_size "$((${BATCH_SIZE} * ${WORLD_SIZE}))" \
  --vla.per_device_batch_size "${BATCH_SIZE}" \
  --vla.max_steps "${max_steps}" \
  --image_aug False \
  --data_root_dir "${LIBERO_DATA_ROOT}" \
  --run_root_dir "${LOG_ROOT}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --is_resume False
