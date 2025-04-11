CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

base_ckpt_dir=/mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-02/16-01-56_diffusion/checkpoints
# /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-24-06_diffusion/checkpoints/030000/pretrained_model
# ls ${base_ckpt_dir} 写成一个 array数组
ckpt_paths=($(ls "${base_ckpt_dir}"))
min_weight1=0.25
max_weight1=0.25
task1_id=0
min_weight2=0.75
max_weight2=0.75
task2_id=4
mid_number1="$(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc)"
mid_number2="${mid_number1}"
sleep_time=30

local_log_dir="./experiments-island/logs-${min_weight1}-${max_weight1}-${task1_id}-${min_weight2}-${max_weight2}-${task2_id}-new"
num_trials_per_task=50
num_tasks_in_suite=1

delta_shift=0.05

echo "$min_weight1 $max_weight1 $task1_id $min_weight2 $max_weight2 $task2_id, mid_number1: $mid_number1, mid_number2: $mid_number2"


# 定义一个函数，输入为 view_point, color, task, 输出为 cur_viewpoint_weight_min, cur_viewpoint_weight_max, cur_color_weight_min, cur_color_weight_max, cur_task_id

function get_single_weight() {
    local x=$1
    # x: A, B, C, AS1, AS2, BS1, BS2
    if [ "${x}" == "A" ]; then
        cur_weight_min=${min_weight1}
        cur_weight_max=${max_weight1}
    elif [ "${x}" == "B" ]; then
        cur_weight_min=${min_weight2}
        cur_weight_max=${max_weight2}
    elif [ "${x}" == "C" ]; then
        cur_weight_min=${mid_number1}
        cur_weight_max=${mid_number1}
    elif [ "${x}" == "AS1" ]; then
        cur_weight_min="$(echo "scale=3; $max_weight1 + $delta_shift * 1" | bc)"
        cur_weight_max="$(echo "scale=3; $max_weight1 + $delta_shift * 1" | bc)"
    elif [ "${x}" == "AS2" ]; then
        cur_weight_min="$(echo "scale=3; $max_weight1 + $delta_shift * 2" | bc)"
        cur_weight_max="$(echo "scale=3; $max_weight1 + $delta_shift * 2" | bc)"
    elif [ "${x}" == "BS1" ]; then
        cur_weight_min="$(echo "scale=3; $min_weight2 - $delta_shift * 1" | bc)"
        cur_weight_max="$(echo "scale=3; $min_weight2 - $delta_shift * 1" | bc)"
    elif [ "${x}" == "BS2" ]; then
        cur_weight_min="$(echo "scale=3; $min_weight2 - $delta_shift * 2" | bc)"
        cur_weight_max="$(echo "scale=3; $min_weight2 - $delta_shift * 2" | bc)"
    else
        echo "Error: invalid viewpoint: ${x}"
        exit 1
    fi

    # 返回 cur_weight_min, cur_weight_max
    echo "${cur_weight_min} ${cur_weight_max}"
}

function get_cur_weight_and_task_id() {
    local viewpoint=$1
    local color=$2
    local task=$3
    # viewpoint, A, B, C, AS1, AS2, BS1, BS2
    # color, A, B, C, AS1, AS2, BS1, BS2
    # task, A, B

    # 从get_single_weight中接收cur_viewpoint_weight_min, cur_viewpoint_weight_max
    # cur_viewpoint_weight_min 和 cur_viewpoint_weight_max 是一块接收的
    read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"

    # 从get_single_weight中接收cur_color_weight_min, cur_color_weight_max
    read cur_color_weight_min cur_color_weight_max <<< "$(get_single_weight ${color})"

    if [ "${task}" == "A" ]; then
        cur_task_id=${task1_id}
    elif [ "${task}" == "B" ]; then
        cur_task_id=${task2_id}
    fi

    # 返回 cur_viewpoint_weight_min, cur_viewpoint_weight_max, cur_color_weight_min, cur_color_weight_max, cur_task_id
    echo "${cur_viewpoint_weight_min} ${cur_viewpoint_weight_max} ${cur_color_weight_min} ${cur_color_weight_max} ${cur_task_id}"
}


for sub_dir in "${ckpt_paths[@]}"; do
    echo "sub_dir: ${sub_dir}"

    if [ "${sub_dir}" == "last" ]; then
        continue
    fi

    # if ckpt_paths 无法整除 10000，continue
    # 因为 sub_dir 是 030000, 040000, 050000, ..., 先把前置的 0 去掉
    sub_dir_without_prefix=$(echo "${sub_dir}" | sed 's/^0*//')
    # echo "???: ${sub_dir}. $((sub_dir_without_prefix % 10000))"
    if [ $((sub_dir_without_prefix % 10000)) -ne 0 ]; then
        continue
    fi

    # if sub_dir != 010000, continue
    if [ "${sub_dir}" != "040000" ]; then
        continue
    fi

    ckpt_path="${base_ckpt_dir}/${sub_dir}/pretrained_model"

    # AAA & BBB
    python experiments/robot/libero/run_libero_eval_dp.py \
        --model_family diffusion \
        --pretrained_checkpoint "${ckpt_path}" \
        --task_suite_name=libero_spatial \
        --prefix="libero_spatial_task_multi_${sub_dir}_A_A_A" \
        --num_trials_per_task ${num_trials_per_task} \
        --num_tasks_in_suite ${num_tasks_in_suite} \
        --use_wandb false \
        --viewpoint_rotate_min_interpolate_weight ${min_weight1} \
        --viewpoint_rotate_max_interpolate_weight ${max_weight1} \
        --color_scale_min_interpolate_weight ${min_weight1} \
        --color_scale_max_interpolate_weight ${max_weight1} \
        --specific_task_id ${task1_id} \
        --local_log_dir "${local_log_dir}" \
        --seed 7 &
    
    sleep "${sleep_time}"

    python experiments/robot/libero/run_libero_eval_dp.py \
        --model_family diffusion \
        --pretrained_checkpoint "${ckpt_path}" \
        --task_suite_name=libero_spatial \
        --prefix="libero_spatial_task_multi_${sub_dir}_B_B_B" \
        --num_trials_per_task ${num_trials_per_task} \
        --num_tasks_in_suite ${num_tasks_in_suite} \
        --use_wandb false \
        --viewpoint_rotate_min_interpolate_weight ${min_weight2} \
        --viewpoint_rotate_max_interpolate_weight ${max_weight2} \
        --color_scale_min_interpolate_weight ${min_weight2} \
        --color_scale_max_interpolate_weight ${max_weight2} \
        --specific_task_id ${task2_id} \
        --local_log_dir "${local_log_dir}" \
        --seed 7 &

    sleep "${sleep_time}"
    

    for viewpoint in AS1 AS2; do
        for color in A; do
            for task in A; do
                echo "################# viewpoint: ${viewpoint}, color: ${color}, task: ${task} #################"
                read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_color_weight_min cur_color_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${color} ${task})"
                python experiments/robot/libero/run_libero_eval_dp.py \
                    --model_family diffusion \
                    --pretrained_checkpoint "${ckpt_path}" \
                    --task_suite_name=libero_spatial \
                    --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${color}_${task}" \
                    --num_trials_per_task ${num_trials_per_task} \
                    --num_tasks_in_suite ${num_tasks_in_suite} \
                    --use_wandb false \
                    --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                    --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                    --color_scale_min_interpolate_weight ${cur_color_weight_min} \
                    --color_scale_max_interpolate_weight ${cur_color_weight_max} \
                    --specific_task_id ${cur_task_id} \
                    --local_log_dir "${local_log_dir}" \
                    --seed 7 &

                sleep "${sleep_time}"
            done
        done
    done


    for viewpoint in BS1 BS2; do
        for color in B; do
            for task in B; do
                echo "################# viewpoint: ${viewpoint}, color: ${color}, task: ${task} #################"
                read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_color_weight_min cur_color_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${color} ${task})"
                python experiments/robot/libero/run_libero_eval_dp.py \
                    --model_family diffusion \
                    --pretrained_checkpoint "${ckpt_path}" \
                    --task_suite_name=libero_spatial \
                    --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${color}_${task}" \
                    --num_trials_per_task ${num_trials_per_task} \
                    --num_tasks_in_suite ${num_tasks_in_suite} \
                    --use_wandb false \
                    --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                    --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                    --color_scale_min_interpolate_weight ${cur_color_weight_min} \
                    --color_scale_max_interpolate_weight ${cur_color_weight_max} \
                    --specific_task_id ${cur_task_id} \
                    --local_log_dir "${local_log_dir}" \
                    --seed 7 &

                sleep "${sleep_time}"
            done
        done
    done


    for viewpoint in A; do
        for color in AS1 AS2; do
            for task in A; do
                echo "################# viewpoint: ${viewpoint}, color: ${color}, task: ${task} #################"
                read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_color_weight_min cur_color_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${color} ${task})"
                python experiments/robot/libero/run_libero_eval_dp.py \
                    --model_family diffusion \
                    --pretrained_checkpoint "${ckpt_path}" \
                    --task_suite_name=libero_spatial \
                    --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${color}_${task}" \
                    --num_trials_per_task ${num_trials_per_task} \
                    --num_tasks_in_suite ${num_tasks_in_suite} \
                    --use_wandb false \
                    --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                    --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                    --color_scale_min_interpolate_weight ${cur_color_weight_min} \
                    --color_scale_max_interpolate_weight ${cur_color_weight_max} \
                    --specific_task_id ${cur_task_id} \
                    --local_log_dir "${local_log_dir}" \
                    --seed 7 &

                sleep "${sleep_time}"
            done
        done
    done


    for viewpoint in B; do
        for color in BS1 BS2; do
            for task in B; do
                echo "################# viewpoint: ${viewpoint}, color: ${color}, task: ${task} #################"
                read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_color_weight_min cur_color_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${color} ${task})"
                python experiments/robot/libero/run_libero_eval_dp.py \
                    --model_family diffusion \
                    --pretrained_checkpoint "${ckpt_path}" \
                    --task_suite_name=libero_spatial \
                    --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${color}_${task}" \
                    --num_trials_per_task ${num_trials_per_task} \
                    --num_tasks_in_suite ${num_tasks_in_suite} \
                    --use_wandb false \
                    --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                    --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                    --color_scale_min_interpolate_weight ${cur_color_weight_min} \
                    --color_scale_max_interpolate_weight ${cur_color_weight_max} \
                    --specific_task_id ${cur_task_id} \
                    --local_log_dir "${local_log_dir}" \
                    --seed 7 &

                sleep "${sleep_time}"
            done
        done
    done

    wait

done


