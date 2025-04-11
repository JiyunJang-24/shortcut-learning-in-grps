CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

base_ckpt_dir=/mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-01/16-51-41_diffusion/checkpoints
# /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-24-06_diffusion/checkpoints/030000/pretrained_model
# ls ${base_ckpt_dir} 写成一个 array数组
ckpt_paths=($(ls "${base_ckpt_dir}"))
min_weight1=0.20
max_weight1=0.30
min_weight2=0.70
max_weight2=0.80
local_log_dir="./experiments-island/logs-${min_weight1}-${max_weight1}-${min_weight2}-${max_weight2}"
num_trials_per_task=25
num_tasks_in_suite=1

need_more_ood_eval=True
need_all_ood_eval=True

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

    sleep 30

    python experiments/robot/libero/run_libero_eval_dp.py \
        --model_family diffusion \
        --pretrained_checkpoint "${ckpt_path}" \
        --task_suite_name=libero_spatial \
        --prefix="libero_spatial_task_1_${sub_dir}_B" \
        --num_trials_per_task ${num_trials_per_task} \
        --num_tasks_in_suite ${num_tasks_in_suite} \
        --use_wandb false \
        --viewpoint_rotate_min_interpolate_weight ${min_weight2} \
        --viewpoint_rotate_max_interpolate_weight ${max_weight2} \
        --color_scale_min_interpolate_weight ${min_weight2} \
        --color_scale_max_interpolate_weight ${max_weight2} \
        --local_log_dir "${local_log_dir}" \
        --seed 7 &

    sleep 30

    python experiments/robot/libero/run_libero_eval_dp.py \
        --model_family diffusion \
        --pretrained_checkpoint "${ckpt_path}" \
        --task_suite_name=libero_spatial \
        --prefix="libero_spatial_task_1_${sub_dir}_C" \
        --num_trials_per_task ${num_trials_per_task} \
        --num_tasks_in_suite ${num_tasks_in_suite} \
        --use_wandb false \
        --viewpoint_rotate_min_interpolate_weight ${min_weight1} \
        --viewpoint_rotate_max_interpolate_weight ${max_weight1} \
        --color_scale_min_interpolate_weight ${min_weight2} \
        --color_scale_max_interpolate_weight ${max_weight2} \
        --local_log_dir "${local_log_dir}" \
        --seed 7 &

    sleep 30

    python experiments/robot/libero/run_libero_eval_dp.py \
        --model_family diffusion \
        --pretrained_checkpoint "${ckpt_path}" \
        --task_suite_name=libero_spatial \
        --prefix="libero_spatial_task_1_${sub_dir}_D" \
        --num_trials_per_task ${num_trials_per_task} \
        --num_tasks_in_suite ${num_tasks_in_suite} \
        --use_wandb false \
        --viewpoint_rotate_min_interpolate_weight ${min_weight2} \
        --viewpoint_rotate_max_interpolate_weight ${max_weight2} \
        --color_scale_min_interpolate_weight ${min_weight1} \
        --color_scale_max_interpolate_weight ${max_weight1} \
        --local_log_dir "${local_log_dir}" \
        --seed 7 &

    if [ "${need_more_ood_eval}" == "True" ]; then
        sleep 30

        python experiments/robot/libero/run_libero_eval_dp.py \
            --model_family diffusion \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name=libero_spatial \
            --prefix="libero_spatial_task_1_${sub_dir}_E" \
            --num_trials_per_task ${num_trials_per_task} \
            --num_tasks_in_suite ${num_tasks_in_suite} \
            --use_wandb false \
            --viewpoint_rotate_min_interpolate_weight ${min_weight1} \
            --viewpoint_rotate_max_interpolate_weight ${max_weight2} \
            --color_scale_min_interpolate_weight ${min_weight1} \
            --color_scale_max_interpolate_weight ${max_weight2} \
            --local_log_dir "${local_log_dir}" \
            --seed 7 &
    fi
    
    if [ "${need_all_ood_eval}" == "True" ]; then
        wait

        # 计算 (min_weight1 + max_weight2) / 2
        # average=$(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc). 使用 bc进行计算
        python experiments/robot/libero/run_libero_eval_dp.py \
            --model_family diffusion \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name=libero_spatial \
            --prefix="libero_spatial_task_1_${sub_dir}_F" \
            --num_trials_per_task ${num_trials_per_task} \
            --num_tasks_in_suite ${num_tasks_in_suite} \
            --use_wandb false \
            --viewpoint_rotate_min_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --viewpoint_rotate_max_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --color_scale_min_interpolate_weight ${min_weight1} \
            --color_scale_max_interpolate_weight ${max_weight1} \
            --local_log_dir "${local_log_dir}" \
            --seed 7 &
        
        sleep 30
        
        python experiments/robot/libero/run_libero_eval_dp.py \
            --model_family diffusion \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name=libero_spatial \
            --prefix="libero_spatial_task_1_${sub_dir}_G" \
            --num_trials_per_task ${num_trials_per_task} \
            --num_tasks_in_suite ${num_tasks_in_suite} \
            --use_wandb false \
            --viewpoint_rotate_min_interpolate_weight ${min_weight1} \
            --viewpoint_rotate_max_interpolate_weight ${max_weight1} \
            --color_scale_min_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --color_scale_max_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --local_log_dir "${local_log_dir}" \
            --seed 7 &
        
        sleep 30
        python experiments/robot/libero/run_libero_eval_dp.py \
            --model_family diffusion \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name=libero_spatial \
            --prefix="libero_spatial_task_1_${sub_dir}_H" \
            --num_trials_per_task ${num_trials_per_task} \
            --num_tasks_in_suite ${num_tasks_in_suite} \
            --use_wandb false \
            --viewpoint_rotate_min_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --viewpoint_rotate_max_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --color_scale_min_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --color_scale_max_interpolate_weight $(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc) \
            --local_log_dir "${local_log_dir}" \
            --seed 7 &

    fi
    wait

done

wait

