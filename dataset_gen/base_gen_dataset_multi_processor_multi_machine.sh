# 首先生成 hdf5, lerobot 文件，然后根据 hdf5 转为 rlds 文件
unset CUDA_VISIBLE_DEVICES
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ -d "/home/xingyouguang" ]] ; then
    conda activate openvla-mini
else 
    conda activate openvla-mini.xyg
fi

CUR_PATH=$(pwd)
export MUJOCO_GL="egl"
libero_task_suite="libero_spatial"
libero_raw_data_dir="/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial"
if [[ ! -d $libero_raw_data_dir ]]; then
    libero_raw_data_dir="/home-ssd/Users/nsgm_lx/xingyouguang/datasets/robotics/libero/libero_spatial"
fi

if [[ ! -d $libero_raw_data_dir ]]; then
    echo "ERROR: ${libero_raw_data_dir} not found"
    exit
fi

echo "libero_raw_data_dir: ${libero_raw_data_dir}"

libero_base_save_dir="${libero_raw_data_dir}_no_noops_island"


: << 'TAG'
bash dataset_gen/base_gen_dataset_multi_processor_multi_machine.sh \
    --viewpoint_rotate_lower_bound=-10.0 \
    --viewpoint_rotate_upper_bound=90.0 \
    --vmin=0.050 \
    --vmax=0.250 \
    --num_tasks_in_suite=5 \
    --specify_task_id=0,1,3,5,8 \
    --number_demo_per_task=20 \
    --demo_repeat_times=10 \
    --number_parallel_process=1

bash dataset_gen/base_gen_dataset_multi_processor_multi_machine.sh \
    --viewpoint_rotate_lower_bound=-10.0 \
    --viewpoint_rotate_upper_bound=90.0 \
    --vmin=0.050 \
    --vmax=0.250 \
    --num_tasks_in_suite=1 \
    --specify_task_id=0 \
    --number_demo_per_task=20 \
    --demo_repeat_times=10 \
    --number_parallel_process=1
TAG


viewpoint_rotate_lower_bound=-10.0
viewpoint_rotate_upper_bound=90.0
vmin=0.050
vmax=0.250
num_tasks_in_suite=5
specify_task_id=0,1,3,5,8
number_demo_per_task=20
demo_repeat_times=10
number_parallel_process=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --viewpoint_rotate_lower_bound=*)
      viewpoint_rotate_lower_bound="${1#*=}"
      shift
      ;;
    --viewpoint_rotate_upper_bound=*)
      viewpoint_rotate_upper_bound="${1#*=}"
      shift
      ;;
    --vmin=*)
      vmin="${1#*=}"
      shift
      ;;
    --vmax=*)
      vmax="${1#*=}"
      shift
      ;;
    --num_tasks_in_suite=*)
      num_tasks_in_suite="${1#*=}"
      shift
      ;;
    --specify_task_id=*)
      specify_task_id="${1#*=}"
      shift
      ;;
    --number_demo_per_task=*)
      number_demo_per_task="${1#*=}"
      shift
      ;;
    --demo_repeat_times=*)
      demo_repeat_times="${1#*=}"
      shift
      ;;
    --number_parallel_process=*)
      number_parallel_process="${1#*=}"
      shift
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo "  --viewpoint_rotate_lower_bound=<数值>     视角旋转下限，默认值: -10.0"
      echo "  --viewpoint_rotate_upper_bound=<数值>     视角旋转上限，默认值: 90.0"
      echo "  --vmin=<数值>                             最小视角值，默认值: 0.050"
      echo "  --vmax=<数值>                             最大视角值，默认值: 0.250"
      echo "  --num_tasks_in_suite=<数值>               套件中任务数量，默认值: 5". 可选1
      echo "  --specify_task_id=<数字列表>              指定任务ID，例如 '0,1,3,5,8'", 可选 0
      echo "  --number_demo_per_task=<数值>             每个任务的演示数量，默认值: 20"
      echo "  --demo_repeat_times=<数值>                演示重复次数，默认值: 10"
      echo "  --number_parallel_process=<数值>          并行进程数，默认值: 1"
      echo "  --help                                    显示此帮助信息"
      exit 0
      ;;
    *)
      echo "警告: 未知选项: $1"
      exit 1
      ;;
  esac
done


if [ 1 -eq 1 ]; then
    python experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_repeat_split_multi_processor.py \
        --libero_task_suite $libero_task_suite \
        --libero_raw_data_dir $libero_raw_data_dir \
        --libero_base_save_dir $libero_base_save_dir \
        --need_hdf5 True --show_diff True --user_name xyg \
        --viewpoint_rotate_lower_bound $viewpoint_rotate_lower_bound \
        --viewpoint_rotate_upper_bound $viewpoint_rotate_upper_bound \
        --vmin $vmin --vmax $vmax --need_color_change False \
        --num_tasks_in_suite $num_tasks_in_suite --specify_task_id $specify_task_id --number_demo_per_task $number_demo_per_task \
        --demo_repeat_times $demo_repeat_times --change_light False \
        --number_parallel_process "${number_parallel_process}"
fi

cd "${CUR_PATH}/dataset_git/rlds_dataset_builder/LIBERO_Spatial_XYG"

export NO_GCE_CHECK="true"
export CUDA_VISIBLE_DEVICES=""

if [ $num_tasks_in_suite -eq 1 ]; then
    hdf5_dir="${libero_base_save_dir}_1_hdf5"
    rlds_dir="${libero_base_save_dir}_1_rlds"
elif [ $num_tasks_in_suite -eq 5 ]; then
    hdf5_dir="${libero_base_save_dir}_split_hdf5"
    rlds_dir="${libero_base_save_dir}_split_rlds"
else
    hdf5_dir="${libero_base_save_dir}_full_hdf5"
    rlds_dir="${libero_base_save_dir}_full_rlds"
fi
user_name="xyg_$(echo ${number_demo_per_task} | awk '{printf "%02d\n", $1}')_$(echo ${demo_repeat_times} | awk '{printf "%02d\n", $1}')_$(echo $viewpoint_rotate_lower_bound | awk '{printf "%.1f\n", $1}')_$(echo $viewpoint_rotate_upper_bound | awk '{printf "%.1f\n", $1}')"
if [[ $num_tasks_in_suite -eq 1 ]]; then
    viewpoint_path="v-$(echo $vmin | awk '{printf "%.3f\n", $1}')-$(echo $vmax | awk '{printf "%.3f\n", $1}')_num$((specify_task_id+1))"
else
    viewpoint_path="v-$(echo $vmin | awk '{printf "%.3f\n", $1}')-$(echo $vmax | awk '{printf "%.3f\n", $1}')_${specify_task_id}"
fi

echo "${hdf5_dir}/${user_name}/${viewpoint_path}"
echo "${rlds_dir}/${user_name}/${viewpoint_path}"

export XYG_HDF5_PATH="${hdf5_dir}/${user_name}/${viewpoint_path}"

# check "${rlds_dir}/${user_name}/${viewpoint_path}" 和 "${hdf5_dir}/${user_name}/${viewpoint_path}" 是否存在
if [[ -d "${rlds_dir}/${user_name}/${viewpoint_path}" ]] || [[ ! -d "${hdf5_dir}/${user_name}/${viewpoint_path}" ]] ; then
    echo "WARNING: ${rlds_dir}/${user_name}/${viewpoint_path} found or ${hdf5_dir}/${user_name}/${viewpoint_path} not found"
    exit
fi

tfds_start_time=$(date +%s)
tfds build --data_dir ${rlds_dir}/${user_name}/${viewpoint_path}
tfds_end_time=$(date +%s)
tfds_delta_time=$((tfds_end_time - tfds_start_time))
tfds_hours=$((tfds_delta_time / 3600))
tfds_minutes=$((tfds_delta_time % 3600 / 60))
tfds_seconds=$((tfds_delta_time % 60))
echo "tfds build time: ${tfds_hours}:${tfds_minutes}:${tfds_seconds}"
