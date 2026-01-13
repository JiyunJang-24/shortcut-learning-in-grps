#!/usr/bin/env bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate shortcut-learning
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}/LIBERO:${PYTHONPATH}"

base_ckpt_dir="${REPO_ROOT}/logs/2026-1-7/23-26-36_libero_qwen_pretrain_split_v-1.000-1.000_entire/prism-qwen25-dinosiglip-plucker-224px+0_5b+mx-libero-spatial+n0+b48+x7"
checkpoint_dir="${base_ckpt_dir}/checkpoints"
checkpoint_name="step-010000-epoch-07-loss=0.4346.pt"

# If Checkpoint Path points at a file, the path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
# Then, config.json should be in the parent folder of ckpt_path
# prismatic/models/load.py#L147
ckpt_path="${checkpoint_dir}/${checkpoint_name}"

log_root="./logs/eval/miniVLA-plucker"
WANDB_PROJECT="libero_spatial_miniVLA_plucker_eval_debug"
export MUJOCO_GL=egl
angles=(0)
tasks=(0 1 2 3 4 5 6 7 8 9)       # 0=A, 4=B
# tasks=(0)
num_threads=2
seeds=(7)
export PYTHONUNBUFFERED=1
mkdir -p "${log_root}"
# 실행
for seed in "${seeds[@]}"; do
  for thread_idx in $(seq 0 $(($num_threads - 1))); do
    (
      for task in "${tasks[@]}"; do
        if (( task % num_threads != thread_idx )); then
          continue
        fi

        for angle in "${angles[@]}"; do
          outdir="${log_root}/seed_${seed}/task_${task}/angle_${angle}"
          mkdir -p "$outdir"

          python -u experiments/robot/libero/run_libero_eval_dp_minivla_cam_info.py \
            --model_family prismatic \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name libero_spatial \
            --prefix "angle_${angle}_task_${task}_seed_${seed}_$(basename "$base_ckpt_dir")_${checkpoint_name}" \
            --num_trials_per_task 25 \
            --num_tasks_in_suite 1 \
            --use_wandb true \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_entity DynamicVLA \
            --viewpoint_rotate_lower_bound "${angle}" \
            --viewpoint_rotate_upper_bound "${angle}" \
            --viewpoint_rotate_min_interpolate_weight 1.0 \
            --viewpoint_rotate_max_interpolate_weight 1.0 \
            --need_color_change false \
            --specific_task_id "${task}" \
            --local_log_dir "${outdir}" \
            --seed "${seed}" \
            --mode plucker \
            --use_plucker True \
            --use_dynamics_basis false
        done
      done
    )
  done
done

wait
echo "All seeds finished."
