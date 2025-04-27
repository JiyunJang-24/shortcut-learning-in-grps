#!/bin/bash
#SBATCH -J 300300-700700-large-20x10-minivla-xyg
#SBATCH -p gpu
#SBATCH -N 1 # <一个节点>
#SBATCH --ntasks=1 # <任务数：作业进程数, 指你下面要运行多少个程序，例如一个 torch 就是一个 task，所以设置为 1 就行>
#SBATCH --cpus-per-task=32 # <每个任务 cpu 核心数>
#SBATCH -G 4 # 调用多少张卡，这里是 2 张
#SBATCH --output=slurm_logs/slurm-%x-%j.out # 作业控制台输出，路径/名字可以随便改
#SBATCH --error=slurm_logs/slurm-%x-%j.err # 作业报错输出，路径/名字可以随便改
#SBATCH --exclude=gn[001,002,003,004,005,010,011] # 排除哪些节点

module load CUDA/12.4
### # # SBATCH --gres=gpu:4 # 调用多少张卡，这里是 2 张
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PRISMATIC_DATA_ROOT="${LIBERO_DATA_ROOT}"
export MASTER_ADDR='127.0.0.1'
export MASTER_PORT='29520'
conda activate openvla-mini.xyg

# 设置环境变量（适用于所有 Hugging Face 库） -- 离线模型
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Next we train MiniVLA based on this Prism base VLM on LIBERO-90.
# Run from the root of the repository
# LIBERO_DATA_ROOT=/mnt/nfs/CMG/xiejunlin/datasets/Robotics/libero
DATA_MIX="minivla-spatial-split-dataset-300300-700700"
# 判断路径是否存在, LIBERO_DATA_ROOT
LIBERO_SUFFIX="libero_spatial_no_noops_island_split_rlds/xyg_20_10_-10.0_90.0"
LIBERO_DATA_ROOT="/mnt/hdd3/xingyouguang/datasets/robotics/libero/${LIBERO_SUFFIX}"
if [ -e ${LIBERO_DATA_ROOT} ] ; then 
    echo "LIBERO_PATH=${LIBERO_DATA_ROOT}"
elif [[ -e "/mnt/hdd2" ]] ; then
    # bash将字符串中的 hdd3 替换为 hdd2
    LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT/hdd3/hdd2}
    echo "LIBERO_PATH=${LIBERO_DATA_ROOT}"
else
    LIBERO_DATA_ROOT="/home-ssd/Users/nsgm_lx/xingyouguang/datasets/robotics/libero/${LIBERO_SUFFIX}"
    echo "LIBERO_PATH=${LIBERO_DATA_ROOT}"
fi

LOG_ROOT=libero_minivla_split_large_distance_20_10
WANDB_PROJECT="libero_minivla_split_large_distance_20_10"
WANDB_ENTITY="1207481522" # should be you user name or team name in w&b account

max_steps=10000
# WORLD_SIZE=8
# BATCH_SIZE=16
# CUDA_VISIBLE_DEVICES_LIST="0,1,2,3,4,5,6,7"
WORLD_SIZE=4
BATCH_SIZE=32
CUDA_VISIBLE_DEVICES_LIST="0,1,2,3,"
# WORLD_SIZE=2
# BATCH_SIZE=32
# CUDA_VISIBLE_DEVICES_LIST="0,1,"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}"

CKPT_PATH=~/.cache/huggingface/hub/models--Stanford-ILIAD--prism-qwen25-extra-dinosiglip-224px-0_5b/snapshots/5cfd2cc6da00c06e0be7abf35d43ec792d8e9498
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
