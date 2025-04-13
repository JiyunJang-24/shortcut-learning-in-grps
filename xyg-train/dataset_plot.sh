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
  # "libero_10"
  "libero_object"
  "libero_spatial"
  "libero_goal"
)


for dataset_name in ${dataset_name_list[@]}; do
  python visual_dataset/plot_data_suite.py --task_suite_name $dataset_name &
done

wait




