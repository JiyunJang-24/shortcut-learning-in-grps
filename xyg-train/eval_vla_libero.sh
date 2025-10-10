#!/bin/bash
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate openvla-mini


# Launch LIBERO-Spatial evals
base_ckpt_path="/home/xingyouguang/.cache/huggingface/hub"
minivla_base="${base_ckpt_path}/models--Stanford-ILIAD--minivla-libero90-prismatic/snapshots/4289f87e8e00706e188c7a3a61fc6e7d72ab2564/checkpoints/step-122500-epoch-55-loss=0.0743.pt"
minivla_vq="${base_ckpt_path}/models--Stanford-ILIAD--minivla-vq-libero90-prismatic/snapshots/b08980cdc05dcaea09673e0d97e2c3bc6aef2494/checkpoints/step-150000-epoch-67-loss=0.0934.pt"
minivla_vq_history="${base_ckpt_path}/models--Stanford-ILIAD--minivla-history2-vq-libero90-prismatic/snapshots/1fba66ea2fbb6cad0de253b83be7bbbbef49ce4e/checkpoints/step-170000-epoch-38-loss=0.3163.pt"


export PRISMATIC_DATA_ROOT='/home/xingyouguang/data/projects/robotics/openvla-mini/LIBERO/dataset'
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint "${minivla_base}" \
  --task_suite_name libero_90 \
  --center_crop True \
  --num_trials_per_task 1 \
  --seed 42 \
  --hf_token "HF_TOKEN_R"       # 写错了


# # Launch LIBERO-Spatial evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True \
#   --num_trials_per_task 10 \
#   --seed 42


# # Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
#   --task_suite_name libero_object \
#   --center_crop True

# # Launch LIBERO-Goal evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
#   --task_suite_name libero_goal \
#   --center_crop True

# # Launch LIBERO-10 (LIBERO-Long) evals
# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
#   --task_suite_name libero_10 \
#   --center_crop True
