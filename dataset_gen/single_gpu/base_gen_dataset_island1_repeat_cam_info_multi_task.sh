#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 여러 LIBERO task_suite + 원하는 task id들을 한 번에 생성:
# 1) HDF5/Lerobot 생성
# 2) HDF5 -> RLDS (tfds build)

unset CUDA_VISIBLE_DEVICES

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate shortcut-learning

# ====== 절대경로 고정 ======
ROOT="/data1/local/shortcut-learning-in-grps"
username="cam_info"

export PYTHONPATH="${ROOT}/LIBERO:${PYTHONPATH}"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/dataset_git/rlds_dataset_builder"

# ====== 공통 설정 ======
viewpoint_rotate=(0.0)
vmin=1.00
vmax=1.00
num_tasks_in_suite=1
number_demo_per_task=10
demo_repeat_times=10

# ====== (중요) suite별로 "원하는 task id"를 여기서만 관리 ======
# 요청한 것:
# - libero_spatial: 0
# - libero_goal:    0, 7
# - libero_object:  4
# - libero_10:      2, 9
declare -A SUITE_TASK_IDS=(
  ["libero_spatial"]="0"
  ["libero_goal"]="0 7"
  ["libero_object"]="4"
  ["libero_10"]="2 9"
)

# ====== suite별 경로/빌더 매핑 함수 ======
get_suite_paths () {
  local suite="$1"

  case "$suite" in
    libero_spatial)
      LIBERO_RAW_DATA_DIR="${ROOT}/dataset_git/libero_spatial"
      BUILDER_DIR="${ROOT}/dataset_git/rlds_dataset_builder/LIBERO_Spatial_XYG_cam_info"
      ;;
    libero_goal)
      LIBERO_RAW_DATA_DIR="${ROOT}/dataset_git/libero_goal"
      BUILDER_DIR="${ROOT}/dataset_git/rlds_dataset_builder/LIBERO_Goal_XYG_cam_info"
      ;;
    libero_object)
      LIBERO_RAW_DATA_DIR="${ROOT}/dataset_git/libero_object"
      BUILDER_DIR="${ROOT}/dataset_git/rlds_dataset_builder/LIBERO_Object_XYG_cam_info"
      ;;
    libero_10)
      LIBERO_RAW_DATA_DIR="${ROOT}/dataset_git/libero_10"
      BUILDER_DIR="${ROOT}/dataset_git/rlds_dataset_builder/LIBERO_10_XYG_cam_info"
      ;;
    *)
      echo "[ERROR] Unknown suite: $suite"
      exit 1
      ;;
  esac

  LIBERO_BASE_SAVE_DIR="${LIBERO_RAW_DATA_DIR}_no_noops_island"
}

run_one_task () {
  local suite="$1"
  local specify_task_id="$2"
  local rot="$3"

  local viewpoint_rotate_lower_bound="$rot"
  local viewpoint_rotate_upper_bound="$rot"

  echo
  echo "============================================================"
  echo "SUITE: ${suite} | TASK_ID: ${specify_task_id} | ROT: ${rot}"
  echo "RAW_DIR: ${LIBERO_RAW_DATA_DIR}"
  echo "SAVE_BASE: ${LIBERO_BASE_SAVE_DIR}"
  echo "BUILDER_DIR: ${BUILDER_DIR}"
  echo "============================================================"

  # 1) HDF5/Lerobot 생성
  python "${ROOT}/experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_repeat_split_with_cam_info.py" \
    --libero_task_suite "$suite" \
    --libero_raw_data_dir "$LIBERO_RAW_DATA_DIR" \
    --libero_base_save_dir "$LIBERO_BASE_SAVE_DIR" \
    --need_hdf5 True --show_diff True --user_name "${username}" \
    --viewpoint_rotate_lower_bound "$viewpoint_rotate_lower_bound" \
    --viewpoint_rotate_upper_bound "$viewpoint_rotate_upper_bound" \
    --vmin "$vmin" --vmax "$vmax" --need_color_change False \
    --num_tasks_in_suite "$num_tasks_in_suite" \
    --specify_task_id "$specify_task_id" \
    --number_demo_per_task "$number_demo_per_task" \
    --demo_repeat_times "$demo_repeat_times" \
    --change_light False

  # 2) tfds build (RLDS)
  (
    export NO_GCE_CHECK="true"
    export CUDA_VISIBLE_DEVICES=""

    local hdf5_dir rlds_dir
    if [ "$num_tasks_in_suite" -eq 1 ]; then
      hdf5_dir="${LIBERO_BASE_SAVE_DIR}_1_hdf5"
      rlds_dir="${LIBERO_BASE_SAVE_DIR}_1_rlds"
    elif [ "$num_tasks_in_suite" -eq 5 ]; then
      hdf5_dir="${LIBERO_BASE_SAVE_DIR}_split_hdf5"
      rlds_dir="${LIBERO_BASE_SAVE_DIR}_split_rlds"
    else
      hdf5_dir="${LIBERO_BASE_SAVE_DIR}_full_hdf5"
      rlds_dir="${LIBERO_BASE_SAVE_DIR}_full_rlds"
    fi

    local user_name
    user_name=$(
      printf "${username}_%02d_%02d_%.1f_%.1f" \
        "$number_demo_per_task" "$demo_repeat_times" \
        "$viewpoint_rotate_lower_bound" "$viewpoint_rotate_upper_bound"
    )

    local viewpoint_path
    if [ "$num_tasks_in_suite" -eq 1 ]; then
      viewpoint_path=$(printf "v-%.3f-%.3f_num%d" "$vmin" "$vmax" $((specify_task_id+1)))
    else
      viewpoint_path=$(printf "v-%.3f-%.3f_%d" "$vmin" "$vmax" "$specify_task_id")
    fi

    echo "HDF5 path: ${hdf5_dir}/${user_name}/${viewpoint_path}"
    echo "RLDS  path: ${rlds_dir}/${user_name}/${viewpoint_path}"

    export XYG_HDF5_PATH="${hdf5_dir}/${user_name}/${viewpoint_path}"
    mkdir -p "${rlds_dir}/${user_name}/${viewpoint_path}"

    cd "${BUILDER_DIR}"

    tfds_start_time=$(date +%s)
    tfds build --data_dir "${rlds_dir}/${user_name}/${viewpoint_path}"
    tfds_end_time=$(date +%s)

    tfds_delta_time=$((tfds_end_time - tfds_start_time))
    printf "tfds build time: %02d:%02d:%02d\n" \
      $((tfds_delta_time/3600)) $((tfds_delta_time%3600/60)) $((tfds_delta_time%60))
  )
}

# ====== 메인 루프: rot -> suite -> task_id ======
for rot in "${viewpoint_rotate[@]}"; do
  echo "=== Viewpoint: ${rot} deg ==="

  for suite in libero_spatial libero_goal libero_object libero_10; do
    get_suite_paths "$suite"

    # suite별 task id 목록 읽기
    read -r -a task_ids <<< "${SUITE_TASK_IDS[$suite]}"

    for tid in "${task_ids[@]}"; do
      run_one_task "$suite" "$tid" "$rot"
    done
  done
done
