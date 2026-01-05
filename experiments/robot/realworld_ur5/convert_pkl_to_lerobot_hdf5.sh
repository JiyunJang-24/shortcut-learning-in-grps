#!/bin/bash

# ================= 사용자가 수정해야 할 부분 =================

# 1. 원본 PKL 파일들이 들어있는 폴더 경로 (마지막 슬래시 없이)
INPUT_DIR="/home/vai/Desktop/yujin/serl_vai/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/Pick_up_the_gray_plush_from_the_bin_and_place_it_on_the_brown_plate"

# 2. 데이터셋이 저장될 루트 폴더
OUTPUT_DIR="/home/vai/Desktop/yujin/shortcut-learning-in-grps/dataset_git"

# 3. LeRobot용 리포지토리 ID (HDF5 파일명에도 사용됨)
REPO_ID="yujin/ur5_plush_pickup"

# ==========================================================

echo "=========================================="
echo " 데이터셋 변환을 시작합니다."
echo " 입력 폴더: $INPUT_DIR"
echo " 출력 폴더: $OUTPUT_DIR"
echo " Repo ID : $REPO_ID"
echo "=========================================="

# Python 스크립트 실행
python convert_pkl_to_lerobot.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --repo_id "$REPO_ID"

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo " [성공] 변환이 완료되었습니다."
    echo "=========================================="
else
    echo "=========================================="
    echo " [실패] 변환 중 오류가 발생했습니다."
    echo "=========================================="
fi