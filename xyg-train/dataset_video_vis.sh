#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

export PRISMATIC_DATA_ROOT='/home/xingyouguang/data/projects/robotics/openvla-mini/LIBERO/dataset'

# Launch LIBERO-Spatial evals
# 写成一个函数的形式

declare -a dataset_name_list=(
  # "libero_90"
  "libero_10"
  "libero_object"
  "libero_spatial"
  "libero_goal"
)

libero_dir='LIBERO/dataset'
video_save_dir='datasets_vis/video_lan_no_noops'
no_noops=True

for dataset_name in ${dataset_name_list[@]}; do
  python experiments/robot/libero/visualize_libero_dataset.py \
  --libero_task_suite $dataset_name \
  --libero_dir "${libero_dir}" \
  --video_save_dir "${video_save_dir}" \
  --no_noops "${no_noops}"
done

wait




