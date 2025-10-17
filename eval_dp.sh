# `lerobot/outputs/train/2025-10-09/02-34-42_diffusion/checkpoints` is the checkpoint directory for the trained Diffusion Policy.
# `True` is the value for `need_inner_interpolate` and should always be true.
# To evaluate other settings, change the viewpoint ranges and checkpoint directory accordingly.
export PYTHONPATH="/root/Desktop/workspace/shortcut-learning-in-grps/LIBERO:$PYTHONPATH"

bash ./xyg-eval/diffusion_policy/base_eval_libero_spatial_multi_cd.sh \
	0.400 0.400 \
	0.600 0.600 \
	lerobot/outputs/train/2025-10-15/15-15-26_DP_ex1_angle_from_0_to_315/checkpoints \
	True &