#!/usr/bin/env bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate shortcut-learning
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/LIBERO:${PYTHONPATH}"

base_ckpt_dir="${REPO_ROOT}/lerobot/outputs/train/2026-01-12/11-02-30_DP_libero_spatial_basis_ex1_angle_from_0_task_0_apply_basis_scale"
checkpoint_dir="${base_ckpt_dir}/checkpoints"
checkpoint_step="030000"
ckpt_path="${checkpoint_dir}/${checkpoint_step}/pretrained_model"
log_root="./logs-angle-test"
export MUJOCO_GL=egl
angles=(0)
tasks=(0)       # 0=A, 4=B
seeds=(7 8 9)
# angles=(0)
# tasks=(0)       # 0=A, 4=B
# seeds=(7)
export PYTHONUNBUFFERED=1
mkdir -p "${log_root}"
# 실행
for seed in "${seeds[@]}"; do
  (
    for task in "${tasks[@]}"; do
      for angle in "${angles[@]}"; do
        outdir="${log_root}/seed_${seed}/task_${task}/angle_${angle}"
        mkdir -p "$outdir"

        python -u experiments/robot/libero/run_libero_eval_dp_minivla_cam_info.py \
          --model_family diffusion \
          --pretrained_checkpoint "${ckpt_path}" \
          --task_suite_name libero_spatial \
          --prefix "angle_${angle}_task_${task}_seed_${seed}_$(basename "$base_ckpt_dir")_${checkpoint_step}" \
          --num_trials_per_task 25 \
          --num_tasks_in_suite 1 \
          --use_wandb true \
          --wandb_project libero_DP_eval \
          --wandb_entity DynamicVLA \
          --viewpoint_rotate_lower_bound "${angle}" \
          --viewpoint_rotate_upper_bound "${angle}" \
          --viewpoint_rotate_min_interpolate_weight 1.0 \
          --viewpoint_rotate_max_interpolate_weight 1.0 \
          --need_color_change false \
          --specific_task_id "${task}" \
          --local_log_dir "${outdir}" \
          --seed "${seed}" \
          --use_plucker false \
          --use_dynamics_basis true
      done
    done
  ) &
done

wait
echo "All seeds finished."
