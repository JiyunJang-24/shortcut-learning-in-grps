#!/bin/bash

: << 'TAG'
bash xyg-eval/base_eval_libero_spatial_shortcut.sh \
    --task1_id_arr="0 1 3 5 8" --task2_id_arr="2 4 6 7 9" \
    --min_weight1=0.050 --max_weight1=0.150 --min_weight2=0.850 --max_weight2=0.950 \
    --model_family="prismatic" --base_log_dir=experiments-test  \
    --base_ckpt_dir=logs/2025-4-25/0-6-14_libero_minivla_split_large_distance_20_10/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b16+x7 \
    --log_prefix=20-10 --eval_interval=2 --need_diversity=false --need_distance=true --ADIS_num=0.150 --BDIS_num=0.850 --need_mid_wait=false

bash xyg-eval/base_eval_libero_spatial_shortcut.sh \
    --task1_id_arr="0 1 3 5 8" --task2_id_arr="2 4 6 7 9" \
    --min_weight1=0.050 --max_weight1=0.150 --min_weight2=0.850 --max_weight2=0.950 \
    --model_family=pi0 --base_log_dir=experiments-pi0 \
    --base_ckpt_dir=None \
    --log_prefix=20-10 --eval_interval=2 --need_diversity=false --need_distance=true --ADIS_num=0.150 --BDIS_num=0.850 --need_mid_wait=true \
    --server_port=8000 --save_scripts_path=None
TAG


CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [[ -d '/mnt/hdd3/xingyouguang' ]] || [[ -d '/mnt/hdd2/xingyouguang' ]] ; then
    conda activate openvla-mini
    echo "conda activate openvla-mini"
else
    conda activate openvla-mini.xyg
    echo "conda activate openvla-mini.xyg"
fi


# 基础默认参数值
task1_id_arr=(0 1 3 5 8)
task2_id_arr=(2 4 6 7 9)
viewpoint_rotate_lower_bound=-10.0
viewpoint_rotate_upper_bound=90.0
min_weight1=0.050
max_weight1=0.150
min_weight2=0.850
max_weight2=0.950

num_trials_per_task=10

model_family="prismatic"
base_ckpt_dir=logs/2025-4-22/23-20-52_libero_minivla_split_large_distance_20_10/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b16+x7
sleep_time=15
log_prefix=20-10
base_log_dir=experiments-test
eval_interval=1
need_diversity="true"
need_distance="false"
ADIS_num=0.150
BDIS_num=0.850
need_mid_wait=false
server_port=8000
save_scripts_path=None

# 解析命名参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task1_id_arr=*)
      task1_ids="${1#*=}"
      # 将字符串转换为数组
      IFS=' ' read -ra task1_id_arr <<< "$task1_ids"
      shift
      ;;
    --task2_id_arr=*)
      task2_ids="${1#*=}"
      # 将字符串转换为数组
      IFS=' ' read -ra task2_id_arr <<< "$task2_ids"
      shift
      ;;
    --viewpoint_rotate_lower_bound=*)
      viewpoint_rotate_lower_bound="${1#*=}"
      shift
      ;;
    --viewpoint_rotate_upper_bound=*)
      viewpoint_rotate_upper_bound="${1#*=}"
      shift
      ;;
    --min_weight1=*)
      min_weight1="${1#*=}"
      shift
      ;;
    --max_weight1=*)
      max_weight1="${1#*=}"
      shift
      ;;
    --min_weight2=*)
      min_weight2="${1#*=}"
      shift
      ;;
    --max_weight2=*)
      max_weight2="${1#*=}"
      shift
      ;;
    --num_trials_per_task=*)
      num_trials_per_task="${1#*=}"
      shift
      ;;
    --model_family=*)
      model_family="${1#*=}"
      shift
      ;;
    --base_ckpt_dir=*)
      base_ckpt_dir="${1#*=}"
      shift
      ;;
    --sleep_time=*)
      sleep_time="${1#*=}"
      shift
      ;;
    --log_prefix=*)
      log_prefix="${1#*=}"
      shift
      ;;
    --base_log_dir=*)
      base_log_dir="${1#*=}"
      shift
      ;;
    --eval_interval=*)
      eval_interval="${1#*=}"
      shift
      ;;
    --need_diversity=*)
      need_diversity="${1#*=}"
      shift
      ;;
    --need_distance=*)
      need_distance="${1#*=}"
      shift
      ;;
    --ADIS_num=*)
      ADIS_num="${1#*=}"
      shift
      ;;
    --BDIS_num=*)
      BDIS_num="${1#*=}"
      shift
      ;;
    --need_mid_wait=*)
      need_mid_wait="${1#*=}"
      shift
      ;;
    --server_port=*)
      server_port="${1#*=}"
      shift
      ;;
    --save_scripts_path=*)
      save_scripts_path="${1#*=}"
      shift
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo "  --task1_id_attr=<数字列表>                任务1 ID列表，用空格分隔，例如 '0 1 3 5 8'"
      echo "  --task2_id_attr=<数字列表>                任务2 ID列表，用空格分隔，例如 '2 4 6 7 9'"
      echo "  --viewpoint_rotate_lower_bound=<数值>     视角旋转下限，默认值: -10.0"
      echo "  --viewpoint_rotate_upper_bound=<数值>     视角旋转上限，默认值: 90.0"
      echo "  --min_weight1=<数值>                      最小权重1，默认值: 0.050"
      echo "  --max_weight1=<数值>                      最大权重1，默认值: 0.150"
      echo "  --min_weight2=<数值>                      最小权重2，默认值: 0.850"
      echo "  --max_weight2=<数值>                      最大权重2，默认值: 0.950"
      echo "  --num_trials_per_task=<数值>              每个任务的试验次数，默认值: 10"
      echo "  --model_family=<字符串>                   模型类型，默认值: prismatic, 可选值 diffusion, prismatic, pi0, pi0_fast"
      echo "  --base_ckpt_dir=<路径>                    检查点目录路径"
      echo "  --sleep_time=<数值>                       睡眠时间，默认值: 15"
      echo "  --log_prefix=<字符串>                     日志前缀，默认值: 20-10"
      echo "  --base_log_dir=<路径>                     日志目录路径，默认值: experiments-test"
      echo "  --need_diversity=<布尔值>                 是否需要多样性，默认值: true"
      echo "  --need_distance=<布尔值>                  是否需要距离，默认值: false"
      echo "  --ADIS_num=<数值>                         ADIS数值，默认值: 0.150"
      echo "  --BDIS_num=<数值>                         BDIS数值，默认值: 0.850"
      echo "  --need_mid_wait=<布尔值>                  多进程是否需要在AA,BB后wait一下，默认值: false"
      echo "  --server_port=<数值>                      服务器端口，默认值: 8000"
      echo "  --save_scripts_path=<路径>                有一些测试脚本，需要保存在 experiments 的save文件中，方便查看参数，默认值: None"
      echo "  --help                                    显示此帮助信息"
      exit 0
      ;;
    *)
      echo "警告: 未知选项: $1"
      exit 1
      ;;
  esac
done

if [[ "${model_family}" == "pi0" ]] || [[ "${model_family}" == "pi0_fast" ]]; then
    export MEMORY_SIZE=small
fi

mid_number1="$(echo "scale=3; ($min_weight1 + $max_weight2) / 2" | bc)"
mid_number2="${mid_number1}"
# /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-24-06_diffusion/checkpoints/030000/pretrained_model
# ls ${base_ckpt_dir} 写成一个 array数组

if [[ "${model_family}" == "pi0" ]] || [[ "${model_family}" == "pi0_fast" ]]; then
    ckpt_paths=("${base_ckpt_dir}")
else
    base_ckpt_dir="${base_ckpt_dir}/checkpoints"
    ckpt_paths=($(ls "${base_ckpt_dir}"))
fi

task1_id_arr_str=$(printf "%s," "${task1_id_arr[@]}" | sed 's/,$//')
task2_id_arr_str=$(printf "%s," "${task2_id_arr[@]}" | sed 's/,$//')

if [[ ${need_diversity} == "true" ]] ; then
    log_prefix="${log_prefix}-div"
fi
if [[ ${need_distance} == "true" ]] ; then
    log_prefix="${log_prefix}-dis-(${ADIS_num}-${BDIS_num})"
fi

if [[ "${viewpoint_rotate_lower_bound}" == "-10.0" ]] && [[ "${viewpoint_rotate_upper_bound}" == "90.0" ]]; then
    local_log_dir="./${base_log_dir}/logs-${log_prefix}-${min_weight1}-${max_weight1}-${task1_id_arr_str}-${min_weight2}-${max_weight2}-${task2_id_arr_str}-large"
elif [[ "${viewpoint_rotate_lower_bound}" == "15.0" ]] && [[ "${viewpoint_rotate_upper_bound}" == "65.0" ]]; then
    local_log_dir="./${base_log_dir}/logs-${log_prefix}-${min_weight1}-${max_weight1}-${task1_id_arr_str}-${min_weight2}-${max_weight2}-${task2_id_arr_str}"
else
    echo "Error: invalid viewpoint_rotate_lower_bound: ${viewpoint_rotate_lower_bound}, viewpoint_rotate_upper_bound: ${viewpoint_rotate_upper_bound}"
    exit 0
fi

if [[ "${save_scripts_path}" != "None" ]]; then
    if [[ -f "${save_scripts_path}" ]]; then
        mkdir -p ${local_log_dir}
        cp ${save_scripts_path} ${local_log_dir}
    else
        echo "Error: save_scripts_path: ${save_scripts_path} not found"
        exit 1
    fi
fi


num_tasks_in_suite=1    # 测试的时候还是单个测试

delta_shift=0.01

need_inner_interpolate=true
need_10000_multi_ckpt=false

echo "$min_weight1 $max_weight1 $task1_id_arr_str $min_weight2 $max_weight2 $task2_id_arr_str, mid_number1: $mid_number1, mid_number2: $mid_number2"



echo "========== 脚本参数调试信息 =========="
echo "  save_scripts_path: ${save_scripts_path}"

echo "任务ID数组:"
echo "  task1_id_arr: ${task1_id_arr[@]}"
echo "  task2_id_arr: ${task2_id_arr[@]}"
echo "  task1_id_arr_str: ${task1_id_arr_str:-未设置}"
echo "  task2_id_arr_str: ${task2_id_arr_str:-未设置}"
echo ""

echo "视角旋转参数:"
echo "  viewpoint_rotate_lower_bound: $viewpoint_rotate_lower_bound"
echo "  viewpoint_rotate_upper_bound: $viewpoint_rotate_upper_bound"
echo ""

echo "权重参数:"
echo "  min_weight1: $min_weight1"
echo "  max_weight1: $max_weight1"
echo "  min_weight2: $min_weight2"
echo "  max_weight2: $max_weight2"
echo "  ADIS_num: $ADIS_num"
echo "  BDIS_num: $BDIS_num"
echo "  delta_shift: $delta_shift"
echo ""

echo "计算的中间值:"
echo "  mid_number1: ${mid_number1:-未计算}"
echo "  mid_number2: ${mid_number2:-未计算}"
echo ""

echo "路径信息:"
echo "  base_ckpt_dir: $base_ckpt_dir"
echo "  local_log_dir: ${local_log_dir:-未设置}"
echo ""

echo "检查点路径数组(前5个):"
if [ ${#ckpt_paths[@]} -gt 0 ]; then
  for i in $(seq 0 $((${#ckpt_paths[@]} > 5 ? 4 : ${#ckpt_paths[@]}-1))); do
    echo "  ckpt_paths[$i]: ${ckpt_paths[$i]}"
  done
  if [ ${#ckpt_paths[@]} -gt 5 ]; then
    echo "  ... 共${#ckpt_paths[@]}个检查点"
  fi
else
  echo "  ckpt_paths: 未设置或为空"
fi
echo ""

echo "控制参数:"
echo "  num_trials_per_task: $num_trials_per_task"
echo "  num_tasks_in_suite: ${num_tasks_in_suite:-未设置}"
echo "  eval_interval: $eval_interval"
echo "  sleep_time: $sleep_time"
echo "  log_prefix: $log_prefix"
echo "  need_diversity: $need_diversity"
echo "  need_distance: $need_distance"
echo "  need_mid_wait: $need_mid_wait"
echo "  need_inner_interpolate: ${need_inner_interpolate:-未设置}"
echo "  need_10000_multi_ckpt: ${need_10000_multi_ckpt:-未设置}"
echo "  model_family: ${model_family:-未设置}"
echo "========================================="




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
    elif [ "${x}" == "ADIS" ]; then   # 距离测试的时候，保证公平性，都在一个点测试
        cur_weight_min=${ADIS_num}
        cur_weight_max=${ADIS_num}
    elif [ "${x}" == "B" ]; then
        cur_weight_min=${min_weight2}
        cur_weight_max=${max_weight2}
    elif [ "${x}" == "BR" ]; then   # B right
        cur_weight_min=${max_weight2}
        cur_weight_max=${max_weight2}
    elif [ "${x}" == "BDIS" ]; then   # 距离测试的时候，保证公平性，都在一个点测试
        cur_weight_min=${BDIS_num}
        cur_weight_max=${BDIS_num}
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


i=0
for sub_dir in "${ckpt_paths[@]}"; do

    echo "sub_dir: ${sub_dir}"

    if [ "${sub_dir}" == "last" ]; then
        continue
    fi

    i=$((i+1))
    if [[ $(("${i}"%"${eval_interval}")) -ne 0 ]] && [[ "${model_family}" != "pi0" ]] && [[ "${model_family}" != "pi0_fast" ]]; then 
        continue
    fi

    # # if sub_dir != step-010000-epoch-09-loss=0.1884.pt and sub_dir != step-006000-epoch-09-loss=0.1884.pt, continue
    # if [ "${sub_dir}" != "step-002500-epoch-03-loss=1.2504.pt" ] && [ "${sub_dir}" != "step-010000-epoch-12-loss=0.2114.pt" ] && [ "${sub_dir}" != "step-015000-epoch-18-loss=0.5499.pt" ] && [ "${sub_dir}" != "step-020000-epoch-24-loss=0.2956.pt" ] && [ "${sub_dir}" != "step-025000-epoch-30-loss=0.0936.pt" ]; then
    #     continue
    # fi

    # if ckpt_paths 无法整除 10000，continue
    # 因为 sub_dir 是 030000, 040000, 050000, ..., 先把前置的 0 去掉
    # sub_dir_without_prefix=$(echo "${sub_dir}" | sed 's/^0*//')
    # # echo "???: ${sub_dir}. $((sub_dir_without_prefix % 10000))", if need_10000_multi_ckpt
    # if [ $((sub_dir_without_prefix % 10000)) -ne 0 ] && [ "${need_10000_multi_ckpt}" == "true" ]; then
    #     continue
    # fi

    ckpt_path="${base_ckpt_dir}/${sub_dir}"

    # AA
    for a_task_id in "${task1_id_arr[@]}" ; do
        python experiments/robot/libero/run_libero_eval_dp_minivla_pi0.py \
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
            --server_port ${server_port} \
            --seed 7 &
        
        sleep "${sleep_time}"
    done
    
    # BB
    for b_task_id in "${task2_id_arr[@]}" ; do
        python experiments/robot/libero/run_libero_eval_dp_minivla_pi0.py \
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
            --server_port ${server_port} \
            --seed 7 &

        sleep "${sleep_time}"
    done

    if [[ "${need_mid_wait}" == "true" ]] ; then
        wait
    fi

    # need_inner_interpolate check
    if [ "${need_inner_interpolate}" == "true" ]; then

        if [[ "${need_diversity}" == "true" ]]; then
            # AL-B
            viewpoint="AL"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"
            for b_task_id in "${task2_id_arr[@]}" ; do
                python experiments/robot/libero/run_libero_eval_dp_minivla_pi0.py \
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
                    --server_port ${server_port} \
                    --seed 7 &

                sleep "${sleep_time}"
            done

            # BR-A
            viewpoint="BR"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"
            for a_task_id in "${task1_id_arr[@]}" ; do
                python experiments/robot/libero/run_libero_eval_dp_minivla_pi0.py \
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
                    --server_port ${server_port} \
                    --seed 7 &
                sleep "${sleep_time}"
            done
        fi

        if [[ "${need_distance}" == "true" ]]; then
            # ADIS-B
            viewpoint="ADIS"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"
            for b_task_id in "${task2_id_arr[@]}" ; do
                python experiments/robot/libero/run_libero_eval_dp_minivla_pi0.py \
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
                    --server_port ${server_port} \
                    --seed 7 &

                sleep "${sleep_time}"
            done

            # BDIS-A
            viewpoint="BDIS"
            read cur_viewpoint_weight_min cur_viewpoint_weight_max <<< "$(get_single_weight ${viewpoint})"
            for a_task_id in "${task1_id_arr[@]}" ; do
                python experiments/robot/libero/run_libero_eval_dp_minivla_pi0.py \
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
                    --server_port ${server_port} \
                    --seed 7 &
                sleep "${sleep_time}"
            done
        fi
    fi

    wait

done

wait
