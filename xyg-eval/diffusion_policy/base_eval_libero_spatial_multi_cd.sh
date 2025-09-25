CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini

min_weight1="$1"
max_weight1="$2"
min_weight2="$3"
max_weight2="$4"
base_ckpt_dir="$5"
need_inner_interpolate="$6"

task1_id=0
task2_id=4
viewpoint_rotate_lower_bound=15.0
viewpoint_rotate_upper_bound=65.0

model_family="diffusion"

num_trials_per_task=25
num_tasks_in_suite=1
local_log_dir="./experiments-dp/logs-${min_weight1}-${max_weight1}-${task1_id}-${min_weight2}-${max_weight2}-${task2_id}"

# the others
mid_number1="$(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc)"
mid_number2="${mid_number1}"

delta_shift=0.01
ckpt_paths=($(ls "${base_ckpt_dir}"))

echo "$min_weight1 $max_weight1 $task1_id $min_weight2 $max_weight2 $task2_id, mid_number1: $mid_number1, mid_number2: $mid_number2"
sleep_time=30


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

function get_cur_weight_and_task_id() {
    local viewpoint=$1
    local task=$2
    # viewpoint, A, B, C, AS1, AS2, BS1, BS2
    # task, A, B

    # 从get_single_weight中接收cur_viewpoint_weight_min, cur_viewpoint_weight_max
    # cur_viewpoint_weight_min 和 cur_viewpoint_weight_max 是一块接收的
    read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"


    if [ "${task}" == "A" ]; then
        cur_task_id=${task1_id}
    elif [ "${task}" == "B" ]; then
        cur_task_id=${task2_id}
    fi

    # 返回 cur_viewpoint_weight_min, cur_viewpoint_weight_max, cur_task_id
    echo "${cur_viewpoint_weight_min} ${cur_viewpoint_weight_max} ${cur_task_id}"
}


for sub_dir in "${ckpt_paths[@]}"; do

    echo "sub_dir: ${sub_dir}"

    if [ "${sub_dir}" == "last" ]; then
        continue
    fi

    if [ "${sub_dir}" != "020000" ]; then
        continue
    fi

    ckpt_path="${base_ckpt_dir}/${sub_dir}/pretrained_model"

    # need_inner_interpolate check
    if [ "${need_inner_interpolate}" == "True" ]; then
        # AL-A
        # for seed in 8 9 ; do
        for seed in 7 ; do
            mkdir -p "${local_log_dir}/${seed}"

            viewpoint="AL"
            task="B"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${task})"
            python experiments/robot/libero/run_libero_eval_dp_minivla.py \
                --model_family "${model_family}" \
                --pretrained_checkpoint "${ckpt_path}" \
                --task_suite_name=libero_spatial \
                --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${task}" \
                --num_trials_per_task ${num_trials_per_task} \
                --num_tasks_in_suite ${num_tasks_in_suite} \
                --use_wandb false \
                --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
                --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
                --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                --need_color_change False \
                --specific_task_id ${cur_task_id} \
                --local_log_dir "${local_log_dir}/${seed}" \
                --seed "$seed" &

            sleep "${sleep_time}"

            viewpoint="BR"
            task="A"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${task})"
            python experiments/robot/libero/run_libero_eval_dp_minivla.py \
                --model_family "${model_family}" \
                --pretrained_checkpoint "${ckpt_path}" \
                --task_suite_name=libero_spatial \
                --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${task}" \
                --num_trials_per_task ${num_trials_per_task} \
                --num_tasks_in_suite ${num_tasks_in_suite} \
                --use_wandb false \
                --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
                --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
                --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                --need_color_change False \
                --specific_task_id ${cur_task_id} \
                --local_log_dir "${local_log_dir}/${seed}" \
                --seed "$seed" &
        done
    else
        for seed in 7 8 9 ; do 
        # for seed in 7 ; do 
            mkdir -p "${local_log_dir}/${seed}"

            # 0.375-B
            viewpoint="AL"
            task="B"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${task})"
            cur_viewpoint_weight_min="0.375"
            cur_viewpoint_weight_max="0.375"

            python experiments/robot/libero/run_libero_eval_dp_minivla.py \
                --model_family "${model_family}" \
                --pretrained_checkpoint "${ckpt_path}" \
                --task_suite_name=libero_spatial \
                --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${task}" \
                --num_trials_per_task ${num_trials_per_task} \
                --num_tasks_in_suite ${num_tasks_in_suite} \
                --use_wandb false \
                --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
                --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
                --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                --need_color_change False \
                --specific_task_id ${cur_task_id} \
                --local_log_dir "${local_log_dir}/${seed}" \
                --seed "$seed" &

            sleep "${sleep_time}"

            viewpoint="BR"
            task="A"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max cur_task_id <<< "$(get_cur_weight_and_task_id ${viewpoint} ${task})"
            cur_viewpoint_weight_min="0.625"
            cur_viewpoint_weight_max="0.625"

            python experiments/robot/libero/run_libero_eval_dp_minivla.py \
                --model_family "${model_family}" \
                --pretrained_checkpoint "${ckpt_path}" \
                --task_suite_name=libero_spatial \
                --prefix="libero_spatial_task_multi_${sub_dir}_${viewpoint}_${task}" \
                --num_trials_per_task ${num_trials_per_task} \
                --num_tasks_in_suite ${num_tasks_in_suite} \
                --use_wandb false \
                --viewpoint_rotate_upper_bound ${viewpoint_rotate_upper_bound} \
                --viewpoint_rotate_lower_bound ${viewpoint_rotate_lower_bound} \
                --viewpoint_rotate_min_interpolate_weight ${cur_viewpoint_weight_min} \
                --viewpoint_rotate_max_interpolate_weight ${cur_viewpoint_weight_max} \
                --need_color_change False \
                --specific_task_id ${cur_task_id} \
                --local_log_dir "${local_log_dir}/${seed}" \
                --seed "$seed" &
        done
    fi
    # wait
done

wait
