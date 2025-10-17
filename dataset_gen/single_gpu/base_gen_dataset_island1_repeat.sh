#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -euo pipefail

# 首先生成 hdf5, lerobot 文件，然后根据 hdf5 转为 rlds 文件
unset CUDA_VISIBLE_DEVICES

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate shortcut-learning

# 절대경로 고정
ROOT="/root/Desktop/workspace/shortcut-learning-in-grps"
CUR_PATH="${ROOT}"                              # 루트 기준으로 고정
BUILDER_DIR="${ROOT}/dataset_git/rlds_dataset_builder/LIBERO_Spatial_XYG"

libero_task_suite="libero_spatial"
libero_raw_data_dir="${ROOT}/dataset_git/libero_spatial"
libero_base_save_dir="${libero_raw_data_dir}_no_noops_island"

viewpoint_rotate=(0.0 45.0 90.0 135.0 225.0 270.0 315.0)
vmin=1.00
vmax=1.00
num_tasks_in_suite=1
specify_task_id_list=(0 4)
number_demo_per_task=10
demo_repeat_times=10

# TFDS가 커스텀 빌더를 볼 수 있도록 보장
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/dataset_git/rlds_dataset_builder"

for i in "${viewpoint_rotate[@]}"; do
    for specify_task_id in "${specify_task_id_list[@]}"; do
        viewpoint_rotate_lower_bound="$i"
        viewpoint_rotate_upper_bound="$i"

        echo "=== Viewpoint: ${i} deg ==="

        # 1) HDF5/Lerobot 재생성 (필요 시)
        if [ 1 -eq 1 ]; then
            python "${ROOT}/experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_repeat_split.py" \
            --libero_task_suite "$libero_task_suite" \
            --libero_raw_data_dir "$libero_raw_data_dir" \
            --libero_base_save_dir "$libero_base_save_dir" \
            --need_hdf5 True --show_diff True --user_name xyg \
            --viewpoint_rotate_lower_bound "$viewpoint_rotate_lower_bound" \
            --viewpoint_rotate_upper_bound "$viewpoint_rotate_upper_bound" \
            --vmin "$vmin" --vmax "$vmax" --need_color_change False \
            --num_tasks_in_suite "$num_tasks_in_suite" \
            --specify_task_id "$specify_task_id" \
            --number_demo_per_task "$number_demo_per_task" \
            --demo_repeat_times "$demo_repeat_times" \
            --change_light False
        fi

        # 2) 각 반복을 서브쉘로 격리해서 cd 영향이 바깥으로 새지 않게 함
        (
            export NO_GCE_CHECK="true"
            export CUDA_VISIBLE_DEVICES=""

            if [ "$num_tasks_in_suite" -eq 1 ]; then
            hdf5_dir="${libero_base_save_dir}_1_hdf5"
            rlds_dir="${libero_base_save_dir}_1_rlds"
            elif [ "$num_tasks_in_suite" -eq 5 ]; then
            hdf5_dir="${libero_base_save_dir}_split_hdf5"
            rlds_dir="${libero_base_save_dir}_split_rlds"
            else
            hdf5_dir="${libero_base_save_dir}_full_hdf5"
            rlds_dir="${libero_base_save_dir}_full_rlds"
            fi

            # 경로 포맷 고정
            user_name=$(
            printf "xyg_%02d_%02d_%.1f_%.1f" \
                "$number_demo_per_task" "$demo_repeat_times" \
                "$viewpoint_rotate_lower_bound" "$viewpoint_rotate_upper_bound"
            )

            if [ "$num_tasks_in_suite" -eq 1 ]; then
            viewpoint_path=$(printf "v-%.3f-%.3f_num%d" "$vmin" "$vmax" $((specify_task_id+1)))
            else
            viewpoint_path=$(printf "v-%.3f-%.3f_%d" "$vmin" "$vmax" "$specify_task_id")
            fi

            echo "HDF5 path: ${hdf5_dir}/${user_name}/${viewpoint_path}"
            echo "RLDS  path: ${rlds_dir}/${user_name}/${viewpoint_path}"

            export XYG_HDF5_PATH="${hdf5_dir}/${user_name}/${viewpoint_path}"

            # data_dir이 없으면 만들어 두기 (tfds build는 leaf 디렉토리 존재해도 OK)
            mkdir -p "${rlds_dir}/${user_name}/${viewpoint_path}"

            # 빌더 디렉토리에서 빌드 (현재작업폴더 의존 제거)
            cd "${BUILDER_DIR}"

            # 각 반복마다 캐시/락 문제 방지: 필요 시 미완성 폴더 제거(선택)
            # find "${rlds_dir}/${user_name}/${viewpoint_path}" -name "*.incomplete" -type d -exec rm -rf {} +

            tfds_start_time=$(date +%s)
            tfds build --data_dir "${rlds_dir}/${user_name}/${viewpoint_path}"
            tfds_end_time=$(date +%s)

            tfds_delta_time=$((tfds_end_time - tfds_start_time))
            printf "tfds build time: %02d:%02d:%02d\n" \
            $((tfds_delta_time/3600)) $((tfds_delta_time%3600/60)) $((tfds_delta_time%60))
        )

        # 서브쉘을 빠져나오면 자동으로 원래 디렉토리/환경으로 복귀
        echo "current path: $(pwd)"
    done
done
