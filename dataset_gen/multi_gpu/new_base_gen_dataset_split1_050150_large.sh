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

viewpoint_rotate_lower_bound=-10.0
viewpoint_rotate_upper_bound=90.0
vmin=0.050
vmax=0.250
num_tasks_in_suite=5
specify_task_id=0,1,3,5,8
# num_tasks_in_suite=1
# specify_task_id=0
number_demo_per_task=50
demo_repeat_times=4
number_parallel_process=1

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
