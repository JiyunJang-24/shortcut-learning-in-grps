"""
python experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_repeat_split_multi_processor.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --libero_base_save_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island \
    --need_hdf5 True \
    --show_diff True \
    --user_name xyg \
    --viewpoint_rotate_lower_bound 15 \
    --viewpoint_rotate_upper_bound 65 \
    --vmin 0.400 \
    --vmax 0.400 \
    --need_color_change False \
    --num_tasks_in_suite 5 \
    --specify_task_id 0,1,3,5,8 \
    --number_demo_per_task 50 \
    --demo_repeat_times 2 \
    --change_light False


python experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_repeat_split_multi_processor.py \
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
import multiprocessing
from functools import partial
import time

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

def process_task(args, task_id, libero_task_suite, shared_values):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_task_suite]()
    
    print(f"current task id: {task_id}, pid: {os.getpid()}")
    current_start_time = time.time()
    # Extract shared values
    libero_home = shared_values['libero_home']
    repo_name = shared_values['repo_name']
    hdf5_output_path = shared_values['hdf5_output_path']
    number_demo_per_task = shared_values['number_demo_per_task']
    demo_repeat_times = shared_values['demo_repeat_times']
    viewpoint_rotate_min = shared_values['viewpoint_rotate_min']
    viewpoint_rotate_max = shared_values['viewpoint_rotate_max']
    color_scale_min = shared_values['color_scale_min']
    color_scale_max = shared_values['color_scale_max']
    need_color_change = shared_values['need_color_change']
    transparent_alpha = shared_values['transparent_alpha']
    transparent_object_name = shared_values['transparent_object_name']
    robot_base_name = shared_values['robot_base_name']
    camera_name = shared_values['camera_name']
    color_light_a = shared_values['color_light_a']
    color_light_b = shared_values['color_light_b']
    change_light = shared_values['change_light']
    base_num = shared_values['base_num']
    
    # Local counters
    local_time_counter = {
        'total': 0,
        'hdf5_gen': 0,
        'lerobot_gen': 0,
        'show_diff': 0,
    }
    num_replays = 0
    num_success = 0
    num_noops = 0
    all_done_list = []
    metainfo_json_dict = {}
    
    # Create task-specific HDF5 output path
    # task_hdf5_output_path = os.path.join(hdf5_output_path, f"task_{task_id}")
    task_hdf5_output_path = os.path.join(hdf5_output_path)
    os.makedirs(task_hdf5_output_path, exist_ok=True)
    
    # Get task in suite
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
    
    print(f"Processing task {task_id}: {task.name}")

    # Get dataset for task
    orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    
    # Create new HDF5 file for regenerated demos
    if args.need_hdf5:
        new_data_path = os.path.join(task_hdf5_output_path, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")
    
    # Process episodes
    real_success_demo_num = 0
    for orig_data_key in tqdm.tqdm(orig_data.keys(), desc=f"episodes in task {task_id}", leave=False):
        # get demo data
        i = int(orig_data_key.split("_")[-1])
        if number_demo_per_task is not None and real_success_demo_num >= number_demo_per_task:
            break
        
        demo_data = orig_data[f"demo_{i}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        done_list = []
        states_list = []
        actions_list = []
        ee_states_list = []
        gripper_states_list = []
        joint_states_list = []
        robot_states_list = []
        agentview_images_list = []
        eye_in_hand_images_list = []
            
        for repeat_idx in range(2): # min(demo_repeat_times, 2) -> 2
            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])      # [0] mean the state of the first timestep in this episode
            
            # 给定 min, max 均匀采样 viewpoint_rotate, color_scale
            viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
            color_scale = np.random.uniform(color_scale_min, color_scale_max)
            
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
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            done_list.append(done)
            states_list.append(states)
            actions_list.append(actions)
            ee_states_list.append(ee_states)
            gripper_states_list.append(gripper_states)
            joint_states_list.append(joint_states)
            robot_states_list.append(robot_states)
            agentview_images_list.append(agentview_images)
            eye_in_hand_images_list.append(eye_in_hand_images)
        
        if sum(done_list) == len(done_list):    # 没有这个条件就放弃这个trajectory了。
            rep_idx = 0
            for done, states, actions, ee_states, gripper_states, joint_states, robot_states, agentview_images, eye_in_hand_images in zip(done_list, states_list, actions_list, ee_states_list, gripper_states_list, joint_states_list, robot_states_list, agentview_images_list, eye_in_hand_images_list):
                if done:        # 这里很机智，保证他是done的。数量可能会从原来的 50 下降到40出头
                    rep_idx += 1

                    if args.need_hdf5:  # 遵循的一个原则，hdf5都没有翻转
                        time_cal_1 = time.time()
                        dones = np.zeros(len(actions)).astype(np.uint8)
                        dones[-1] = 1
                        rewards = np.zeros(len(actions)).astype(np.uint8)
                        rewards[-1] = 1
                        assert len(actions) == len(agentview_images)
                        ep_data_grp = grp.create_group(f"demo_{i}_{rep_idx}")
                        obs_grp = ep_data_grp.create_group("obs")
                        obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                        obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                        obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                        obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                        obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                        obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                        obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                        ep_data_grp.create_dataset("actions", data=actions)
                        ep_data_grp.create_dataset("states", data=np.stack(states))
                        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                        ep_data_grp.create_dataset("rewards", data=rewards)
                        ep_data_grp.create_dataset("dones", data=dones)
                        time_cal_2 = time.time()
                        local_time_counter['hdf5_gen'] += time_cal_2 - time_cal_1                        

                    if args.show_diff:
                        time_cal_1 = time.time()
                        cur_agentview_rgb = agentview_images
                        ori_agentview_rgb = demo_data["obs"]["agentview_rgb"][()]
                        ori_agentview_rgb = [ori_agentview_rgb[j] for j in range(len(ori_agentview_rgb))]
                        
                        # cur_agent_view_rgb 和 ori_agentview_rgb 都是list of numpy.ndarray, shape: (256, 256, 3)
                        # 将两者竖直方向concat 起来，存为 jpg
                        # ori_agentview_rgb 图像插值为 (128, 128, 3) -> (256, 256, 3)
                        ori_agentview_rgb = [cv2.resize(ori_agentview_rgb[j], (256, 256)) for j in range(len(ori_agentview_rgb))]
                        whole_ori_agentview_rgb = np.concatenate(ori_agentview_rgb[::10], axis=1)
                        whole_cur_agentview_rgb = np.concatenate(cur_agentview_rgb[::10], axis=1)
                        if whole_ori_agentview_rgb.shape[1] == whole_cur_agentview_rgb.shape[1]:
                            whole_ori_agentview_rgb, whole_cur_agentview_rgb = np.flipud(whole_ori_agentview_rgb), np.flipud(whole_cur_agentview_rgb)
                            all_agentview_rgb = np.concatenate((whole_ori_agentview_rgb, whole_cur_agentview_rgb), axis=0)
                            # 将all_agentview_rgb:(H, W, 3)保存为图片
                            
                            task_diff_dir = os.path.join(f"{repo_name}_task_{task_id}", f"{args.libero_task_suite}", f"{task.name}")
                            os.makedirs(task_diff_dir, exist_ok=True)
                            Image.fromarray(all_agentview_rgb).save(os.path.join(task_diff_dir, f"demo_{i}_{rep_idx}.jpg"))
                        time_cal_2 = time.time()
                        local_time_counter['show_diff'] += time_cal_2 - time_cal_1
                        
                    num_success += 1
                if demo_repeat_times == 1:
                    break
            if demo_repeat_times == 1:
                cur_demo_success = 1
            else:
                cur_demo_success = sum(done_list)
            while cur_demo_success < demo_repeat_times:
                env.reset()
                env.set_init_state(orig_states[0])      # [0] mean the state of the first timestep in this episode
                
                # 给定 min, max 均匀采样 viewpoint_rotate, color_scale
                viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
                color_scale = np.random.uniform(color_scale_min, color_scale_max)
                
                camera_id = env.sim.model.camera_name2id(camera_name)
                if need_color_change:
                    env = rotate_recolor_dataset.recolor_and_rotate_scene(env, alpha=color_scale, color_light_a=color_light_a, color_light_b=color_light_b, 
                                                                    camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                                    theta=viewpoint_rotate, debug=False, need_change_light=change_light, base_num=base_num)
                else:
                    # env = rotate_camera(env, camera_id, camera_name, robot_base_name, theta)
                    env = rotate_recolor_dataset.rotate_camera(env, camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                            theta=viewpoint_rotate, debug=False)
                
                # change the transparency of the transparent object
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
                        # In the first timestep, since we're using the original initial state to initialize the environment,
                        # copy the initial state (first state in episode) over from the original HDF5 to the new one
                        states.append(orig_states[0])
                        robot_states.append(demo_data["robot_states"][0])
                    else:
                        # For all other timesteps, get state from environment and record it
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
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                    # Execute demo action in environment
                    obs, reward, done, info = env.step(action.tolist())
                
                if done:
                    rep_idx += 1
                    cur_demo_success += 1
                    
                    if args.need_hdf5:
                        time_cal_1 = time.time()
                        dones = np.zeros(len(actions)).astype(np.uint8)
                        dones[-1] = 1
                        rewards = np.zeros(len(actions)).astype(np.uint8)
                        rewards[-1] = 1
                        assert len(actions) == len(agentview_images)
                        ep_data_grp = grp.create_group(f"demo_{i}_{rep_idx}")
                        obs_grp = ep_data_grp.create_group("obs")
                        obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                        obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                        obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                        obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                        obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                        obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                        obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                        ep_data_grp.create_dataset("actions", data=actions)
                        ep_data_grp.create_dataset("states", data=np.stack(states))
                        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                        ep_data_grp.create_dataset("rewards", data=rewards)
                        ep_data_grp.create_dataset("dones", data=dones)
                        time_cal_2 = time.time()
                        local_time_counter['hdf5_gen'] += time_cal_2 - time_cal_1

                    if args.show_diff:
                        time_cal_1 = time.time()
                        cur_agentview_rgb = agentview_images
                        ori_agentview_rgb = demo_data["obs"]["agentview_rgb"][()]
                        ori_agentview_rgb = [ori_agentview_rgb[j] for j in range(len(ori_agentview_rgb))]
                        
                        # cur_agent_view_rgb 和 ori_agentview_rgb 都是list of numpy.ndarray, shape: (256, 256, 3)
                        # 将两者竖直方向concat 起来，存为 jpg
                        # ori_agentview_rgb 图像插值为 (128, 128, 3) -> (256, 256, 3)
                        ori_agentview_rgb = [cv2.resize(ori_agentview_rgb[j], (256, 256)) for j in range(len(ori_agentview_rgb))]
                        whole_ori_agentview_rgb = np.concatenate(ori_agentview_rgb[::10], axis=1)
                        whole_cur_agentview_rgb = np.concatenate(cur_agentview_rgb[::10], axis=1)
                        if whole_ori_agentview_rgb.shape[1] == whole_cur_agentview_rgb.shape[1]:
                            whole_ori_agentview_rgb, whole_cur_agentview_rgb = np.flipud(whole_ori_agentview_rgb), np.flipud(whole_cur_agentview_rgb)
                            all_agentview_rgb = np.concatenate((whole_ori_agentview_rgb, whole_cur_agentview_rgb), axis=0)
                            # 将all_agentview_rgb:(H, W, 3)保存为图片
                            
                            task_diff_dir = os.path.join(f"{repo_name}_task_{task_id}", f"{args.libero_task_suite}", f"{task.name}")
                            os.makedirs(task_diff_dir, exist_ok=True)
                            Image.fromarray(all_agentview_rgb).save(os.path.join(task_diff_dir, f"demo_{i}_{rep_idx}.jpg"))
                        time_cal_2 = time.time()
                        local_time_counter['show_diff'] += time_cal_2 - time_cal_1
                        
            real_success_demo_num += 1

        num_replays += 1
        
        if sum(done_list) != 0 and sum(done_list) != len(done_list):
            pass
        if done_list[-1]:
            all_done_list.append(done_list)
        # Record success/false and initial environment state in metainfo dict
        task_key = task_description.replace(" ", "_")
        episode_key = f"demo_{i}"
        if task_key not in metainfo_json_dict:
            metainfo_json_dict[task_key] = {}
        if episode_key not in metainfo_json_dict[task_key]:
            metainfo_json_dict[task_key][episode_key] = {}
        metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
        metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

        # Count total number of successful replays so far
        print(
            f"Task {task_id}: Episodes replayed: {num_replays}, "
            f"Successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
        )

    # Close HDF5 files
    orig_data_file.close()
    if args.need_hdf5:
        new_data_file.close()
    current_end_time = time.time()
    local_time_counter['total'] += current_end_time - current_start_time
    
    # 返回收集到的所有数据，而不是直接写入数据集
    return {
        'task_id': task_id,
        'task_name': task.name,
        'task_description': task_description,
        'time_counter': local_time_counter,
        'num_success': num_success,
        'num_replays': num_replays,
        'num_noops': num_noops,
        'all_done_list': all_done_list,
        'metainfo_json_dict': metainfo_json_dict,
    }


def main(args):
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

    # Determine which tasks to process
    if num_tasks_in_suite == task_suite.n_tasks:
        libero_home = args.libero_base_save_dir + "_full" + "_lerobot"
        task_id_list = list(range(num_tasks_in_suite))
    else:
        if args.specify_task_id is None:
            task_id_list = list(range(num_tasks_in_suite))
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
    
    # Create the base output directories
    libero_home = Path(libero_home)
    lerobot_output_path = libero_home / repo_name
    
    hdf5_output_path = str(lerobot_output_path).replace("_lerobot/", "_hdf5/")
    os.makedirs(hdf5_output_path, exist_ok=True)
    
    if lerobot_output_path.exists():
        print(f"Removing existing dataset at {lerobot_output_path}")
        raise ValueError(f"Dataset already exists at {lerobot_output_path}")
    
    # 在主进程中创建单一的LeRobotDataset实例
    print(f"Creating LeRobotDataset at {lerobot_output_path}")
    # os.makedirs(lerobot_output_path, exist_ok=True)
    
    # 确保HDF5目录存在
    if args.need_hdf5:
        os.makedirs(hdf5_output_path, exist_ok=True)
    
    # Prepare shared values dictionary for multiprocessing
    shared_values = {
        'libero_home': libero_home,
        'repo_name': repo_name,
        'hdf5_output_path': hdf5_output_path,
        'number_demo_per_task': number_demo_per_task,
        'demo_repeat_times': demo_repeat_times,
        'viewpoint_rotate_min': viewpoint_rotate_min,
        'viewpoint_rotate_max': viewpoint_rotate_max,
        'color_scale_min': color_scale_min,
        'color_scale_max': color_scale_max,
        'need_color_change': need_color_change,
        'transparent_alpha': transparent_alpha,
        'transparent_object_name': transparent_object_name,
        'robot_base_name': robot_base_name,
        'camera_name': camera_name,
        'color_light_a': color_light_a,
        'color_light_b': color_light_b,
        'change_light': change_light,
        'base_num': base_num,
    }
    
    # Run tasks in parallel
    print(f"Processing {len(task_id_list)} tasks in parallel")
    num_processes = len(task_id_list)  # One process per task
    
    # Use multiprocessing Pool to process tasks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_task_with_args = partial(process_task, args, libero_task_suite=args.libero_task_suite, shared_values=shared_values)
        results = list(tqdm.tqdm(
            pool.imap(process_task_with_args, task_id_list),
            total=len(task_id_list),
            desc="Processing tasks"
        ))
    
    # Combine results from all processes
    combined_metainfo_json_dict = {}
    combined_time_counter = {
        'total': 0,
        'hdf5_gen': 0,
        'show_diff': 0,
        'lerobot_gen': 0,
    }
    total_num_success = 0
    total_num_replays = 0
    total_num_noops = 0
    all_done_list = []
    
    # 现在在主进程中创建数据集并填充它
    print(f"Creating the unified LeRobotDataset and populating it with collected episodes...")
    
    # 创建单一的LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        root=lerobot_output_path,
        fps=10,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (92,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
            
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    for result in results:
        task_id = result['task_id']
        task_name = result['task_name']
        task_description = result['task_description']
        
        # Add task info to metainfo dict
        combined_metainfo_json_dict.update(result['metainfo_json_dict'])    # dict_keys(['pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate']), 加入新的 k-v
        
        # Combine time counters
        if result['time_counter']['total'] > combined_time_counter['total']:    # 并行运算，只看最耗时的那一个进程
            combined_time_counter['total'] = result['time_counter']['total']
            combined_time_counter['hdf5_gen'] = result['time_counter']['hdf5_gen']
            combined_time_counter['lerobot_gen'] = result['time_counter']['lerobot_gen']
            combined_time_counter['show_diff'] = result['time_counter']['show_diff']
        
        # Combine other statistics
        total_num_success += result['num_success']
        total_num_replays += result['num_replays']
        total_num_noops += result['num_noops']
        all_done_list.extend(result['all_done_list'])
        
        print(f"Task {task_id} ({task_name}) completed: {result['num_success']}/{result['num_replays']} episodes successful")
        
        time_cal_1 = time.time()
        # 读取 hdf5 文件，将其写入到 lerobot 中
        # hdf5_file_path = os.path.join(hdf5_output_path, f"task_{task_id}", f"{task_name}_demo.hdf5")
        hdf5_file_path = os.path.join(hdf5_output_path, f"{task_name}_demo.hdf5")
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        for key in hdf5_file["data"].keys():   # demo_{i}_{rep_idx}
            agentview_images = hdf5_file["data"][key]["obs"]["agentview_rgb"][()]
            eye_in_hand_images = hdf5_file["data"][key]["obs"]["eye_in_hand_rgb"][()]
            robot_states = hdf5_file["data"][key]["robot_states"][()]
            states = hdf5_file["data"][key]["states"][()]
            actions = hdf5_file["data"][key]["actions"][()]
            
            for lerobot_idx in range(len(agentview_images)):
                dataset.add_frame({
                    "observation.image": np.flipud(agentview_images[lerobot_idx]),
                    "observation.wrist_image": np.flipud(eye_in_hand_images[lerobot_idx]),
                    "observation.state": robot_states[lerobot_idx].astype(np.float32),
                    "observation.environment_state": states[lerobot_idx].astype(np.float32),
                    "task": task_description,
                    "action": actions[lerobot_idx].astype(np.float32),
                })
            dataset.save_episode()
        time_cal_2 = time.time()
        combined_time_counter['lerobot_gen'] += time_cal_2 - time_cal_1
    
    # Write combined metainfo to file
    with open(metainfo_json_out_path, "w") as f:
        json.dump(combined_metainfo_json_dict, f, indent=2)
    
    print(f"Dataset regeneration complete!")
    print(f"Saved dataset at: {lerobot_output_path}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    
    print(f"Total episodes replayed: {total_num_replays}")
    print(f"Total successful episodes: {total_num_success} ({total_num_success / total_num_replays * 100:.1f}% success rate)")
    print(f"Total no-op actions filtered: {total_num_noops}")
    
    # Return combined time counter for overall timing statistics
    return combined_time_counter


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    # --------------- input task suite --------------- #
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        required=True,
    )
    
    # --------------- raw hdf5 dir, save hdf5 dir, lerobot dir --------------- #
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help=("Path to directory containing raw HDF5 dataset. " "Example: ./LIBERO/libero/datasets/libero_spatial"),
        required=True,
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
    
    # Start data regeneration with multiprocessing
    import time
    start_time = time.time()
    combined_time_counter = main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    def print_time_counter(output_time):
        hours = output_time // 3600
        minutes = (output_time % 3600) // 60
        seconds = output_time % 60
        return f"{int(hours)}:{int(minutes)}:{int(seconds)}"
        
    print(f"Total Time taken: " + print_time_counter(elapsed_time))
    print(f"hdf5_gen: " + print_time_counter(combined_time_counter['hdf5_gen']))
    print(f"lerobot_gen: " + print_time_counter(combined_time_counter['lerobot_gen']))
    print(f"show_diff: " + print_time_counter(combined_time_counter['show_diff']))
    print(f"env_time: " + print_time_counter(elapsed_time - combined_time_counter['hdf5_gen'] - combined_time_counter['lerobot_gen'] - combined_time_counter['show_diff']))
