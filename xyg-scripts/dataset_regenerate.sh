#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini


dataset_base_dir="/mnt/nfs/CMG/xiejunlin/datasets/Robotics"
task_suite_arr=(
  # "libero_spatial"
  "libero_goal"
  "libero_object"
  "libero_10"
  "libero_90"
)

for task_suite in ${task_suite_arr[@]}; do
  python experiments/robot/libero/regenerate_libero_dataset.py \
    --libero_task_suite "${task_suite}" \
    --libero_raw_data_dir "${dataset_base_dir}/libero/${task_suite}" \
    --libero_target_dir "${dataset_base_dir}/libero-xyg/${task_suite}" &
done

wait
