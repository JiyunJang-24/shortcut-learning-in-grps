CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

base_ckpt_dir=logs/2025-4-20/22-44-44_libero_qwen_pretrain_split_large/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b16+x7/checkpoints
# /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-24-06_diffusion/checkpoints/030000/pretrained_model
# ls ${base_ckpt_dir} 写成一个 array数组
ckpt_paths=($(ls "${base_ckpt_dir}"))
min_weight1=0.400
max_weight1=0.400
task1_id_arr=(0 1 3 5 8)
min_weight2=0.600
max_weight2=0.600
task2_id_arr=(2 4 6 7 9)
mid_number1="$(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc)"
mid_number2="${mid_number1}"
sleep_time=10

task1_id_arr_str=$(printf "%s," "${task1_id_arr[@]}" | sed 's/,$//')
task2_id_arr_str=$(printf "%s," "${task2_id_arr[@]}" | sed 's/,$//')
local_log_dir="./experiments-test/logs-50-02-${min_weight1}-${max_weight1}-${task1_id_arr_str}-${min_weight2}-${max_weight2}-${task2_id_arr_str}-large-specify_tmp"
num_trials_per_task=10
num_tasks_in_suite=1    # 测试的时候还是单个测试

delta_shift=0.01

viewpoint_rotate_lower_bound=-10.0
viewpoint_rotate_upper_bound=90.0

need_inner_interpolate=True
need_10000_multi_ckpt=False
model_family="prismatic"

echo "$min_weight1 $max_weight1 $task1_id $min_weight2 $max_weight2 $task2_id, mid_number1: $mid_number1, mid_number2: $mid_number2"


# 定义一个函数，输入为 view_point, task, 输出为 cur_viewpoint_weight_min, cur_viewpoint_weight_max, cur_task_id
function get_single_weight() {
    local x=$1
    # x: A, B, C, AS1, AS2, BS1, BS2
    if [ "${x}" == "A" ]; then
        cur_weight_min=${min_weight1}
        cur_weight_max=${max_weight1}
    elif [ "${x}" == "AL" ]; then   # A left
        cur_weight_min=${min_weight1}
        cur_weight_max=${min_weight1}
    elif [ "${x}" == "B" ]; then
        cur_weight_min=${min_weight2}
        cur_weight_max=${max_weight2}
    elif [ "${x}" == "BR" ]; then   # B right
        cur_weight_min=${max_weight2}
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
    elif [ "${x}" == "AS4" ]; then
        cur_weight_min="$(echo "scale=3; $max_weight1 + $delta_shift * 4" | bc)"
        cur_weight_max="$(echo "scale=3; $max_weight1 + $delta_shift * 4" | bc)"
    elif [ "${x}" == "AS8" ]; then
        cur_weight_min="$(echo "scale=3; $max_weight1 + $delta_shift * 8" | bc)"
        cur_weight_max="$(echo "scale=3; $max_weight1 + $delta_shift * 8" | bc)"
    elif [ "${x}" == "BS1" ]; then
        cur_weight_min="$(echo "scale=3; $min_weight2 - $delta_shift * 1" | bc)"
        cur_weight_max="$(echo "scale=3; $min_weight2 - $delta_shift * 1" | bc)"
    elif [ "${x}" == "BS2" ]; then
        cur_weight_min="$(echo "scale=3; $min_weight2 - $delta_shift * 2" | bc)"
        cur_weight_max="$(echo "scale=3; $min_weight2 - $delta_shift * 2" | bc)"
    elif [ "${x}" == "BS4" ]; then
        cur_weight_min="$(echo "scale=3; $min_weight2 - $delta_shift * 4" | bc)"
        cur_weight_max="$(echo "scale=3; $min_weight2 - $delta_shift * 4" | bc)"
    elif [ "${x}" == "BS8" ]; then
        cur_weight_min="$(echo "scale=3; $min_weight2 - $delta_shift * 8" | bc)"
        cur_weight_max="$(echo "scale=3; $min_weight2 - $delta_shift * 8" | bc)"
    else
        echo "Error: invalid viewpoint: ${x}"
        exit 1
    fi

    # 返回 cur_weight_min, cur_weight_max
    echo "${cur_weight_min} ${cur_weight_max}"
}

# 使用 i 计数，4的倍数才进行 echo
i=0
for sub_dir in "${ckpt_paths[@]}"; do
    i=$((i + 1))
    echo "sub_dir: ${sub_dir}"

    # if [ "${sub_dir}" == "last" ]; then
    #     continue
    # fi

    # if [ "${sub_dir}" == "005000" ]; then
    #     continue
    # fi

    # # if sub_dir != step-010000-epoch-09-loss=0.1884.pt and sub_dir != step-006000-epoch-09-loss=0.1884.pt, continue
    # if [ "${sub_dir}" != "step-002500-epoch-03-loss=1.2504.pt" ] && [ "${sub_dir}" != "step-010000-epoch-12-loss=0.2114.pt" ] && [ "${sub_dir}" != "step-015000-epoch-18-loss=0.5499.pt" ] && [ "${sub_dir}" != "step-020000-epoch-24-loss=0.2956.pt" ] && [ "${sub_dir}" != "step-025000-epoch-30-loss=0.0936.pt" ]; then
    #     continue
    # fi

    # if ckpt_paths 无法整除 10000，continue
    # 因为 sub_dir 是 030000, 040000, 050000, ..., 先把前置的 0 去掉
    # sub_dir_without_prefix=$(echo "${sub_dir}" | sed 's/^0*//')
    # # echo "???: ${sub_dir}. $((sub_dir_without_prefix % 10000))", if need_10000_multi_ckpt
    # if [ $((sub_dir_without_prefix % 10000)) -ne 0 ] && [ "${need_10000_multi_ckpt}" == "True" ]; then
    #     continue
    # fi

    ckpt_path="${base_ckpt_dir}/${sub_dir}"

    # AA
    for a_task_id in "8" ; do
        python experiments/robot/libero/run_libero_eval_dp_minivla.py \
            --model_family "${model_family}" \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name=libero_spatial \
            --prefix="libero_spatial_task_multi_${sub_dir}_A_${a_task_id}" \
            --num_trials_per_task ${num_trials_per_task} \
            --num_tasks_in_suite ${num_tasks_in_suite} \
            --use_wandb false \
            --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
            --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
            --viewpoint_rotate_min_interpolate_weight ${min_weight1} \
            --viewpoint_rotate_max_interpolate_weight ${max_weight1} \
            --need_color_change False \
            --specific_task_id ${a_task_id} \
            --local_log_dir "${local_log_dir}" \
            --seed 7 &
        
        sleep "${sleep_time}"
    done

    # BB
    for b_task_id in "7" ; do
        python experiments/robot/libero/run_libero_eval_dp_minivla.py \
            --model_family "${model_family}" \
            --pretrained_checkpoint "${ckpt_path}" \
            --task_suite_name=libero_spatial \
            --prefix="libero_spatial_task_multi_${sub_dir}_B_${b_task_id}" \
            --num_trials_per_task ${num_trials_per_task} \
            --num_tasks_in_suite ${num_tasks_in_suite} \
            --use_wandb false \
            --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
            --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
            --viewpoint_rotate_min_interpolate_weight ${min_weight2} \
            --viewpoint_rotate_max_interpolate_weight ${max_weight2} \
            --need_color_change False \
            --specific_task_id ${b_task_id} \
            --local_log_dir "${local_log_dir}" \
            --seed 7 &

        sleep "${sleep_time}"
    done

    # need_inner_interpolate check
    if [ "${need_inner_interpolate}" == "True" ]; then
        # AL-B
        viewpoint="AL"
        read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"
        for b_task_id in "7" ; do
            python experiments/robot/libero/run_libero_eval_dp_minivla.py \
                --model_family "${model_family}" \
                --pretrained_checkpoint "${ckpt_path}" \
                --task_suite_name=libero_spatial \
                --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${b_task_id}" \
                --num_trials_per_task ${num_trials_per_task} \
                --num_tasks_in_suite ${num_tasks_in_suite} \
                --use_wandb false \
                --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
                --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
                --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                --need_color_change False \
                --specific_task_id ${b_task_id} \
                --local_log_dir "${local_log_dir}" \
                --seed 7 &

            sleep "${sleep_time}"
        done

        # BR-A
        viewpoint="BR"
        read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"
        for a_task_id in "8" ; do
            python experiments/robot/libero/run_libero_eval_dp_minivla.py \
                --model_family "${model_family}" \
                --pretrained_checkpoint "${ckpt_path}" \
                --task_suite_name=libero_spatial \
                --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${a_task_id}" \
                --num_trials_per_task ${num_trials_per_task} \
                --num_tasks_in_suite ${num_tasks_in_suite} \
                --use_wandb false \
                --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
                --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
                --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                --need_color_change False \
                --specific_task_id ${a_task_id} \
                --local_log_dir "${local_log_dir}" \
                --seed 7 &
            sleep "${sleep_time}"
        done

    fi

    if ((i % 4 == 0)); then
        wait
    fi
done

wait
