# -*- coding: utf-8 -*-
export PYTHONPATH="/root/Desktop/workspace/shortcut-learning-in-grps/LIBERO:$PYTHONPATH"
# IMPORTANT: Update the `libero_raw_data_dir` variable and ensure the environment name (`conda activate shortcut-learning`) is correct inside the bash scripts below.
# ################# Diffusion Policy settings #################
# Diversity setting 1 (lowest viewpoint diversity)
# bash dataset_gen/single_gpu/base_gen_dataset_island1_400400.sh # task id=0, camera range: 40% → 40% within [15.0, 65.0], i.e., 15 + (65 - 15) * 0.40 → 15 + (65 - 15) * 0.40
bash dataset_gen/single_gpu/base_gen_dataset_island1_custom.sh # task id=4, camera range: 60% → 60% within [15.0, 65.0], i.e., 15 + (65 - 15) * 0.60 → 15 + (65 - 15) * 0.60
# bash dataset_gen/single_gpu/base_gen_dataset_island1_repeat.sh # task id=4, camera range: 60% → 60% within [15.0, 65.0], i.e., 15 + (65 - 15) * 0.60 → 15 + (65 - 15) * 0.60
# # Diversity setting 2 (medium viewpoint diversity)
# bash dataset_gen/single_gpu/base_gen_dataset_island1_400500.sh # task id=0, camera range: 40% → 50% within [15.0, 65.0]
# bash dataset_gen/single_gpu/base_gen_dataset_island1_500600.sh # task id=4, camera range: 50% → 60% within [15.0, 65.0]

# # Diversity setting 3 (highest viewpoint diversity)
# bash dataset_gen/single_gpu/base_gen_dataset_island1_400550.sh # task id=0, camera range: 40% → 55% within [15.0, 65.0]
# bash dataset_gen/single_gpu/base_gen_dataset_island1_450600.sh # task id=4, camera range: 45% → 60% within [15.0, 65.0]

# # Viewpoint disparity settings (you can adjust parameters to generate these datasets)
# # [0.375, 0.575] task 0 ; [0.425, 0.625] task 4. highest viewpoint disparity.
# # [0.350, 0.550] task 0 ; [0.450, 0.650] task 4. high viewpoint disparity.
# # [0.300, 0.500] task 0 ; [0.500, 0.700] task 4. low viewpoint disparity.
# # [0.250, 0.450] task 0 ; [0.550, 0.750] task 4. lowest viewpoint disparity.


# # IMPORTANT: Again, verify `libero_raw_data_dir` and the environment name in the scripts below.
# # ################# MiniVLA (OpenVLA-Mini) settings #################
# # Diversity setting 1 (lowest viewpoint diversity)
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_200200_large.sh # task list={0,1,3,5,8}, camera range: 20% → 20% within [-10.0, 90.0], i.e., -10 + (90 - (-10)) * 0.20 → -10 + (90 - (-10)) * 0.20
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_800800_large.sh # task list={2,4,6,7,9}, camera range: 80% → 80% within [-10.0, 90.0]

# # Diversity setting 2 (low viewpoint diversity)
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_200350_large.sh # task list={0,1,3,5,8}, camera range: 20% → 35% within [-10.0, 90.0], i.e., -10 + (90 - (-10)) * 0.20 → -10 + (90 - (-10)) * 0.35
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_650800_large.sh # task list={2,4,6,7,9}, camera range: 65% → 80% within [-10.0, 90.0]

# # Diversity setting 3 (high viewpoint diversity)
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_200500_large.sh # task list={0,1,3,5,8}, camera range: 20% → 50% within [-10.0, 90.0]
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_500800_large.sh # task list={2,4,6,7,9}, camera range: 50% → 80% within [-10.0, 90.0]

# # Diversity setting 4 (highest viewpoint diversity)
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_200650_large.sh # task list={0,1,3,5,8}, camera range: 20% → 65% within [-10.0, 90.0]
# bash dataset_gen/multi_gpu/new_base_gen_dataset_split1_50_02_350800_large.sh # task list={2,4,6,7,9}, camera range: 35% → 80% within [-10.0, 90.0]

# # Viewpoint disparity presets (you can adjust parameters to generate these datasets)
# # [0.200, 0.200] task list={0,1,3,5,8} ; [0.800, 0.800] task list={2,4,6,7,9}. highest viewpoint disparity.
# # [0.300, 0.300] task list={0,1,3,5,8} ; [0.700, 0.700] task list={2,4,6,7,9}. high viewpoint disparity.
# # [0.400, 0.400] task list={0,1,3,5,8} ; [0.600, 0.600] task list={2,4,6,7,9}. low viewpoint disparity.
# # [0.450, 0.450] task list={0,1,3,5,8} ; [0.550, 0.550] task list={2,4,6,7,9}. lowest viewpoint disparity.
