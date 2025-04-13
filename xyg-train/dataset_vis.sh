#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

export PRISMATIC_DATA_ROOT='/home/xingyouguang/data/projects/robotics/openvla-mini/LIBERO/dataset'

# Launch LIBERO-Spatial evals
# 写成一个函数的形式

function save_task_suite() {
  task_suite_name=$1
  resize_size=$2
  num_trials_per_task=$3
  num_steps_wait=$4

  python visual_dataset/save_task_suite.py \
    --task_suite_name $task_suite_name \
    --resize_size $resize_size \
    --num_trials_per_task $num_trials_per_task \
    --num_steps_wait $num_steps_wait
}


declare -a dataset_name_list=(
  # "libero_90"
  # "libero_10"
  "libero_object"
  "libero_spatial"
  "libero_goal"
)


for dataset_name in ${dataset_name_list[@]}; do
  save_task_suite $dataset_name 224 1 10 &
done

wait





