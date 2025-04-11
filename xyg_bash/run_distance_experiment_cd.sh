sleep_time=5

bash xyg_bash/run_distance_experiment_base_cd.sh 0.375 0.575 0.425 0.625 /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-04/22-37-38_diffusion/checkpoints &

sleep ${sleep_time}
bash xyg_bash/run_distance_experiment_base_cd.sh 0.35 0.55 0.45 0.65 /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-04/22-39-12_diffusion/checkpoints &

sleep ${sleep_time}
bash xyg_bash/run_distance_experiment_base_cd.sh 0.30 0.50 0.50 0.70 /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-04/22-40-31_diffusion/checkpoints &

sleep ${sleep_time}
bash xyg_bash/run_distance_experiment_base_cd.sh 0.25 0.45 0.55 0.75 /mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-04-04/22-43-35_diffusion/checkpoints &

wait

