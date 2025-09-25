############################ 第一部分 ############################
# File "/home/xingyouguang/data/software/miniconda3/envs/openvla-mini/lib/python3.10/site-packages/robosuite/utils/log_utils.py", line 71, in __init__
#     fh = logging.FileHandler("/tmp/robosuite.log")
# 直接给他修改，修改到你自己目录的文件夹就可以了

# bash 中 true/false 判断可以这样写:
if true ; then echo "条件为真" ; else echo "条件为假" ; fi
if false ; then echo "条件为真" ; else echo "条件为假" ; fi



if false ; then
    bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
        0.400 0.400 \
        0.600 0.600 \
        /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/10-57-58_diffusion/checkpoints \
        True &

    bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
        0.400 0.500 \
        0.500 0.600 \
        /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/11-00-20_diffusion/checkpoints \
        True &

    bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
        0.400 0.550 \
        0.450 0.600 \
        /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/11-00-50_diffusion/checkpoints \
        True &
fi

############################ 第二部分 ############################
bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
    0.375 0.575 \
    0.425 0.625 \
    /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/22-33-05_diffusion/checkpoints \
    False &

wait

bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
    0.350 0.550 \
    0.450 0.650 \
    /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/22-33-17_diffusion/checkpoints \
    False &

bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
    0.300 0.500 \
    0.500 0.700 \
    /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/22-33-31_diffusion/checkpoints \
    False &

bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
    0.250 0.450 \
    0.550 0.750 \
    /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-16/22-36-17_diffusion/checkpoints \
    False &
wait
