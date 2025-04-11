CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

base_ckpt_dir=/mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-31/22-10-10_diffusion/checkpoints
# /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-24-06_diffusion/checkpoints/030000/pretrained_model
# ls ${base_ckpt_dir} 写成一个 array数组
ckpt_paths=($(ls "${base_ckpt_dir}"))
min_weight1=0.25
max_weight1=0.25
local_log_dir="./experiments/logs-${min_weight1}-${max_weight1}-${min_weight1}-${max_weight1}-res18pretrained-nogroup"
num_trials_per_task=25
num_tasks_in_suite=1

for sub_dir in "${ckpt_paths[@]}"; do
    # if ckpt == 'last' continue
    # if != last, then 
    if [ "${sub_dir}" == "last" ]; then
        continue
    fi

    ckpt_path="${base_ckpt_dir}/${sub_dir}/pretrained_model"
    python experiments/robot/libero/run_libero_eval_dp.py \
        --model_family diffusion \
        --pretrained_checkpoint "${ckpt_path}" \
        --task_suite_name=libero_spatial \
        --prefix="libero_spatial_task_1_${sub_dir}_A" \
        --num_trials_per_task ${num_trials_per_task} \
        --num_tasks_in_suite ${num_tasks_in_suite} \
        --use_wandb false \
        --viewpoint_rotate_min_interpolate_weight ${min_weight1} \
        --viewpoint_rotate_max_interpolate_weight ${max_weight1} \
        --color_scale_min_interpolate_weight ${min_weight1} \
        --color_scale_max_interpolate_weight ${max_weight1} \
        --local_log_dir "${local_log_dir}" \
        --seed 7 &
    
    sleep 30    # must do it
    
done

wait

