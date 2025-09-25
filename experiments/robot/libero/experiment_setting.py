"""
python experiments/robot/libero/experiment_setting.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --libero_base_save_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island \
    --need_hdf5 True \
    --show_diff True \
    --user_name xyg \
    --viewpoint_rotate_lower_bound 15 \
    --viewpoint_rotate_upper_bound 65 \
    --vmin 0.500 \
    --vmax 0.200 \
    --need_color_change False \
    --num_tasks_in_suite 1 \
    --specify_task_id 0 \
    --number_demo_per_task 20 \
    --demo_repeat_times 1 \
    --change_light False


python experiments/robot/libero/experiment_setting.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --libero_base_save_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island \
    --need_hdf5 True \
    --show_diff True \
    --user_name xyg \
    --viewpoint_rotate_lower_bound 15 \
    --viewpoint_rotate_upper_bound 65 \
    --vmin 0.600 \
    --vmax 0.600 \
    --need_color_change False \
    --num_tasks_in_suite 5 \
    --specify_task_id 2,4,6,7,9 \
    --number_demo_per_task 50 \
    --demo_repeat_times 2 \
    --change_light False

"""

import argparse
import json
import os
os.environ["PRISMATIC_DATA_ROOT"] = "/mnt/hdd3/xingyouguang/datasets/robotics/libero"

import cv2
import h5py
import numpy as np
from PIL import Image
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import LIBERO.xyg_scripts.rotate_recolor_dataset as rotate_recolor_dataset
IMAGE_RESOLUTION = 256


import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds

TIME_COUNTER = {
    'hdf5_gen': 0,
    'lerobot_gen': 0,
    'show_diff': 0,
}


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def interpolate_number(number_min, number_max, interpolate_weight):
    return number_min + (number_max - number_min) * interpolate_weight


def main(args):
    save_image_list = []
    
    number_demo_per_task = args.number_demo_per_task
    demo_repeat_times = args.demo_repeat_times
    
    need_color_change = args.need_color_change
    
    print(f"Regenerating {args.libero_task_suite} dataset!")
    # transparent object
    transparent_alpha = 0.0
    transparent_object_name = "akita_black_bowl_2"

    # uniform distribution
    robot_base_name = 'robot0_link0'
    camera_name = "agentview"
    viewpoint_rotate_lower_bound = args.viewpoint_rotate_lower_bound
    viewpoint_rotate_upper_bound = args.viewpoint_rotate_upper_bound
    color_light_a = np.array([1.0, 0.0, 0.0])
    color_light_b = np.array([1.0, 1.0, 0.0])
    color_scale_upper_bound = 1.0
    color_scale_lower_bound = 0.0
    
    viewpoint_rotate_min_interpolate_weight = args.vmin
    viewpoint_rotate_max_interpolate_weight = args.vmax
    color_scale_min_interpolate_weight = args.cmin
    color_scale_max_interpolate_weight = args.cmax
    
    change_light = args.change_light
    base_num = args.base_num
    
    # "xyg/v-0.25-0.25-c-0.25-0.25"
    user_name = f"{args.user_name}_{number_demo_per_task:02d}_{demo_repeat_times:02d}_{viewpoint_rotate_lower_bound:.1f}_{viewpoint_rotate_upper_bound:.1f}"
    if need_color_change:
        repo_name = f"{user_name}/v-{viewpoint_rotate_min_interpolate_weight:.3f}-{viewpoint_rotate_max_interpolate_weight:.3f}-c-{color_scale_min_interpolate_weight:.3f}-{color_scale_max_interpolate_weight:.3f}"
    else:
        repo_name = f"{user_name}/v-{viewpoint_rotate_min_interpolate_weight:.3f}-{viewpoint_rotate_max_interpolate_weight:.3f}"
    
    if args.suffix is not None and args.suffix != "":
        repo_name = repo_name + f"-{args.suffix}"
            
    
    viewpoint_rotate_min = interpolate_number(viewpoint_rotate_lower_bound, viewpoint_rotate_upper_bound, viewpoint_rotate_min_interpolate_weight)
    viewpoint_rotate_max = interpolate_number(viewpoint_rotate_lower_bound, viewpoint_rotate_upper_bound, viewpoint_rotate_max_interpolate_weight)
    color_scale_min = interpolate_number(color_scale_lower_bound, color_scale_upper_bound, color_scale_min_interpolate_weight)
    color_scale_max = interpolate_number(color_scale_lower_bound, color_scale_upper_bound, color_scale_max_interpolate_weight)
    print(f"v: {viewpoint_rotate_min_interpolate_weight}->{viewpoint_rotate_min}, {viewpoint_rotate_max_interpolate_weight}->{viewpoint_rotate_max}, "
          f"c: {color_scale_min_interpolate_weight}->{color_scale_min}, {color_scale_max_interpolate_weight}->{color_scale_max}")

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./experiments/robot/libero/{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = min(task_suite.n_tasks, args.num_tasks_in_suite)

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    if num_tasks_in_suite == task_suite.n_tasks:
        libero_home = args.libero_base_save_dir + "_full" + "_lerobot"
        task_id_list = list(range(num_tasks_in_suite))
    else:
        if args.specify_task_id is None:
            task_id_list = list(range(num_tasks_in_suite))
            # libero_home = args.lerobot_home + f"_{num_tasks_in_suite}"
            libero_home = args.libero_base_save_dir + f"_{num_tasks_in_suite}" + "_lerobot"
        else:
            if isinstance(args.specify_task_id, list):
                task_id_list = args.specify_task_id
                assert len(task_id_list) * 2 == task_suite.n_tasks and len(task_id_list) == num_tasks_in_suite  # list就得是task suite的一半
                libero_home = args.libero_base_save_dir + "_split" + "_lerobot"
                repo_name = repo_name + "_" + ",".join([str(task_id) for task_id in task_id_list])
            elif isinstance(args.specify_task_id, int):
                task_id_list = [args.specify_task_id]
                assert num_tasks_in_suite == 1
                libero_home = args.libero_base_save_dir + "_1" + "_lerobot"
                repo_name = repo_name + f"_num{args.specify_task_id+1}" # 这个很关键
            else:
                raise ValueError(f"specify_task_id must be a list or an int, but got {type(args.specify_task_id)}")
            
        
    libero_home = Path(libero_home)
    lerobot_output_path = libero_home / repo_name
    
    hdf5_output_path = str(lerobot_output_path).replace("_lerobot/", "_hdf5/")
    
    # task loop in task_suite
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"tasks-{args.libero_task_suite}"):
    all_done_list = []
    for task_id in tqdm.tqdm(task_id_list, desc=f"tasks-{args.libero_task_suite}"):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # modify every episode in this task
        # 应该换一种形式了
        # for i in range(len(orig_data.keys())):  # demo_0, demo_1, ..., demo_49, ...
        # 这边加入 tqdm 
        done = False
        real_success_demo_num = 0
        i = -1
        # for orig_data_key in tqdm.tqdm(orig_data.keys(), desc="episide in tasks"):  # demo_0, demo_1, ..., demo_49, ...
        for orig_data_key in orig_data.keys():  # demo_0, demo_1, ..., demo_49, ...
            # get demo data
            i += 1
            if number_demo_per_task is not None and i >= number_demo_per_task:  # 如果 number_demo_per_task 不为 None，则只处理前 number_demo_per_task 个 demo
                break
            
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]     #  The () is used to indicate that you want to read the entire dataset
            orig_states = demo_data["states"][()]       
            
            for repeat_idx in range(1): # min(demo_repeat_times, 2) -> 2
                # Reset environment, set initial state, and wait a few steps for environment to settle
                env.reset()
                env.set_init_state(orig_states[0])      # [0] mean the state of the first timestep in this episode
                
                # 给定 min, max 均匀采样 viewpoint_rotate, color_scale
                viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
                # viewpoint_rotate = viewpoint_rotate_min + (viewpoint_rotate_max - viewpoint_rotate_min) * (i / (number_demo_per_task - 1))
                color_scale = np.random.uniform(color_scale_min, color_scale_max)
                
                print(f'i: {i}, viewpoint_rotate: {viewpoint_rotate}')
                # recolor and rotate scene
                camera_id = env.sim.model.camera_name2id(camera_name)
                if need_color_change:
                    env = rotate_recolor_dataset.recolor_and_rotate_scene(env, alpha=color_scale, color_light_a=color_light_a, color_light_b=color_light_b, 
                                                                    camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                                    theta=viewpoint_rotate, debug=False, need_change_light=change_light, base_num=base_num)
                else:
                    env = rotate_recolor_dataset.rotate_camera(env, camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                              theta=viewpoint_rotate, debug=False)
                env = rotate_recolor_dataset.change_object_transparency(env, object_name=transparent_object_name, alpha=transparent_alpha, debug=False)
                
                for _ in range(10):
                    obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

                # Set up new data lists
                states, actions, ee_states, gripper_states, joint_states, robot_states, agentview_images, eye_in_hand_images = [], [], [], [], [], [], [], []

                # Replay original demo actions in environment and record observations
                for _, action in enumerate(orig_actions):
                    # Skip transitions with no-op actions
                    prev_action = actions[-1] if len(actions) > 0 else None
                    if is_noop(action, prev_action):
                        print(f"\tSkipping no-op action: {action}")
                        num_noops += 1
                        continue

                    if states == []:
                        states.append(orig_states[0])
                        robot_states.append(demo_data["robot_states"][0])
                    else:
                        states.append(env.sim.get_state().flatten())
                        robot_states.append(
                            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                        )

                    # Record original action (from demo)
                    actions.append(action)

                    # Record data returned by environment
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])
                    joint_states.append(obs["robot0_joint_pos"])
                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )
                    agentview_images.append(obs["agentview_image"])
                    save_image_list.append(obs["agentview_image"])
                    break
    
    print('this is a test')
    # save_image_list 是 image list，将其保存为 video
    # setting_save_path = r'datasets_vis/libero_setting/BB'
    setting_save_path = args.setting_save_path
    os.makedirs(setting_save_path, exist_ok=True)
    
    save_image_list = [image[::-1, :, :] for image in save_image_list]
    for i, image in enumerate(save_image_list):
        image = Image.fromarray(image[:, :, :])
        image.save(os.path.join(setting_save_path, f"{i:02d}.png"))
    
    # 生成视频
    import imageio
    imageio.mimsave(os.path.join(setting_save_path, 'libero_setting.mp4'), save_image_list, fps=10)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    # --------------- input task suite --------------- #
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        # required=True,
        default="libero_spatial",
    )
    
    # --------------- raw hdf5 dir, save hdf5 dir, lerobot dir --------------- #
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help=("Path to directory containing raw HDF5 dataset. " "Example: ./LIBERO/libero/datasets/libero_spatial"),
        # required=True,
        default="/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial",
    )
    
    parser.add_argument(
        "--libero_base_save_dir",
        type=str,
        help="path to lerobot home",
        required=True,
        # "/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_lerobot_island"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        help="suffix of the dataset, e.g. {user_name}/v-{viewpoint_min:.3f}-{viewpoint_max:.3f}-{args.suffix}",
        default=""  # flip
    )
    
    parser.add_argument(
        "--need_hdf5",
        type=str,
        help="need hdf5 ??",
        default="False"
    )
    
    parser.add_argument(
        "--user_name",
        type=str,
        help="user name",
        default="xyg"
    )
    
    parser.add_argument(
        "--show_diff",
        type=str,
        help=("difference between the original and the regenerated dataset, e.g. image"),
        default="False"
    )
    
    # vmin, vmax, cmin, cmax
    parser.add_argument(
        "--vmin",
        type=float,
        help="minimum value for viewpoint rotation. viewpoint_rotate_min_interpolate_weight",
        default=0.25
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="maximum value for viewpoint rotation. viewpoint_rotate_max_interpolate_weight",
        default=0.25
    )
    
    parser.add_argument(
        "--cmin",
        type=float,
        help="minimum value for color scale. color_scale_min_interpolate_weight",
        default=0.25
    )   
    parser.add_argument(
        "--cmax",
        type=float,
        help="maximum value for color scale. color_scale_max_interpolate_weight",
        default=0.25
    )
    
    parser.add_argument(
        "--change_light",
        type=str,
        help="change light",
        default="True"
    )
    parser.add_argument(
        "--base_num",
        type=float,
        help="base num, for scene light",
        default=0.1
    )
    
    parser.add_argument(
        "--num_tasks_in_suite",
        type=int,
        help="number of tasks in suite",
        default=100
    )
    
    parser.add_argument(
        "--specify_task_id",
        type=str,
        help="specify task id,0, 1, 2, ...",
        default=None
    )
    
    parser.add_argument(
        "--number_demo_per_task",
        type=int,
        help="number of demo per task",
        default=None    # 默认为全部
    )
    
    parser.add_argument(
        "--demo_repeat_times",
        type=int,
        help="demo repeat times",
        default=1    # 默认为1
    )
    
    # viewpoint_rotate_upper_bound, viewpoint_rotate_lower_bound
    parser.add_argument(
        "--viewpoint_rotate_upper_bound",
        type=float,
        help="viewpoint rotate upper bound",
        default=-10
    )
    
    parser.add_argument(
        "--viewpoint_rotate_lower_bound",
        type=float,
        help="viewpoint rotate lower bound",
        default=90
    )
    
    
    parser.add_argument(
        "--need_color_change",
        type=str,
        help="need color change",
        default="True"
    )
    
    parser.add_argument(
        "--setting_save_path",
        type=str,
        help="setting save path",
        default="datasets_vis/libero_setting"
    )
    
    
    args = parser.parse_args()
    if args.specify_task_id is not None:
        if "," in args.specify_task_id:
            args.specify_task_id = [int(task_id) for task_id in args.specify_task_id.split(",")]
        else:
            args.specify_task_id = int(args.specify_task_id)
    
    args.show_diff = args.show_diff.lower() == "true"
    args.change_light = args.change_light.lower() == "true"
    args.need_color_change = args.need_color_change.lower() == "true"
    args.need_hdf5 = args.need_hdf5.lower() == "true"
    # Start data regeneration
    import time
    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    def print_time_counter(output_time):
        hours = output_time // 3600
        minutes = (output_time % 3600) // 60
        seconds = output_time % 60
        return f"{int(hours)}:{int(minutes)}:{int(seconds)}"
        
    print(f"Total Time taken: " + print_time_counter(elapsed_time))
    print(f"hdf5_gen: " + print_time_counter(TIME_COUNTER['hdf5_gen']))
    print(f"lerobot_gen: " + print_time_counter(TIME_COUNTER['lerobot_gen']))
    print(f"show_diff: " + print_time_counter(TIME_COUNTER['show_diff']))
    print(f"env_time: " + print_time_counter(elapsed_time - TIME_COUNTER['hdf5_gen'] - TIME_COUNTER['lerobot_gen'] - TIME_COUNTER['show_diff']))
