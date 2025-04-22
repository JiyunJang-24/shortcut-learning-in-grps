"""
python experiments/robot/libero/regenerate_libero_hdf5_lerobot_dataset_repeat.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --libero_base_save_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island \
    --need_hdf5 True \
    --show_diff True \
    --user_name xyg \
    --viewpoint_rotate_lower_bound 15 \
    --viewpoint_rotate_upper_bound 65 \
    --vmin 0.00 \
    --vmax 1.00 \
    --need_color_change False \
    --num_tasks_in_suite 1 \
    --specify_task_id 4 \
    --number_demo_per_task 20 \
    --demo_repeat_times 10 \
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

def process_task(task_id, args, task_suite, hdf5_output_path, repo_name, libero_home, number_demo_per_task, demo_repeat_times, need_color_change,
                viewpoint_rotate_min, viewpoint_rotate_max, color_scale_min, color_scale_max, 
                color_light_a, color_light_b, change_light, base_num, transparent_alpha, transparent_object_name):
    """Process a single task using a separate process"""
    # Initialize local time counter for this process
    local_time_counter = {
        'hdf5_gen': 0,
        'lerobot_gen': 0,
        'show_diff': 0,
    }
    
    # Create a separate dataset instance for this process
    lerobot_output_path = libero_home / repo_name
    dataset = LeRobotDataset.create(
        repo_id=f"{repo_name}_task_{task_id}",  # Make repo_id unique per task
        robot_type="panda",
        root=lerobot_output_path / f"task_{task_id}",  # Separate directory per task
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
        image_writer_threads=5,  # Reduced to prevent resource contention
        image_writer_processes=2,  # Reduced to prevent resource contention
    )
    
    # Get task in suite
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

    # Get dataset for task
    orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
    assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
    orig_data_file = h5py.File(orig_data_path, "r")
    orig_data = orig_data_file["data"]
    
    # Create new HDF5 file for regenerated demos
    if args.need_hdf5:  
        new_data_path = os.path.join(hdf5_output_path, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

    # Process variables
    num_replays = 0
    num_success = 0
    num_noops = 0
    
    # modify every episode in this task
    done = False
    real_success_demo_num = 0
    for orig_data_key in tqdm.tqdm(orig_data.keys(), desc=f"Process {task_id}: episode in tasks"):
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
            env.set_init_state(orig_states[0])
            
            # 给定 min, max 均匀采样 viewpoint_rotate, color_scale
            viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
            color_scale = np.random.uniform(color_scale_min, color_scale_max)
            
            # recolor and rotate scene
            camera_id = env.sim.model.camera_name2id("agentview")
            if need_color_change:
                env = rotate_recolor_dataset.recolor_and_rotate_scene(env, alpha=color_scale, color_light_a=color_light_a, color_light_b=color_light_b, 
                                                                camera_id=camera_id, camera_name="agentview", robot_base_name="robot0_link0", 
                                                                theta=viewpoint_rotate, debug=False, need_change_light=change_light, base_num=base_num)
            else:
                env = rotate_recolor_dataset.rotate_camera(env, camera_id=camera_id, camera_name="agentview", robot_base_name="robot0_link0", 
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

        # Rest of the processing for this episode
        if sum(done_list) == len(done_list):
            rep_idx = 0
            for done, states, actions, ee_states, gripper_states, joint_states, robot_states, agentview_images, eye_in_hand_images in zip(done_list, states_list, actions_list, ee_states_list, gripper_states_list, joint_states_list, robot_states_list, agentview_images_list, eye_in_hand_images_list):
                if done:
                    rep_idx += 1
                    
                    time_cal_1 = time.time()
                    for lerobot_idx in range(len(states)):
                        dataset.add_frame(
                            {
                                "observation.image": np.flipud(agentview_images[lerobot_idx]),
                                "observation.wrist_image": np.flipud(eye_in_hand_images[lerobot_idx]),
                                "observation.state": robot_states[lerobot_idx].astype(np.float32),
                                "observation.environment_state": states[lerobot_idx].astype(np.float32),
                                "task": task_description,
                                "action": actions[lerobot_idx].astype(np.float32), 
                            }
                        )
                    dataset.save_episode()
                    time_cal_2 = time.time()
                    local_time_counter['lerobot_gen'] += time_cal_2 - time_cal_1

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
                        
                        ori_agentview_rgb = [cv2.resize(ori_agentview_rgb[j], (256, 256)) for j in range(len(ori_agentview_rgb))]
                        whole_ori_agentview_rgb = np.concatenate(ori_agentview_rgb[::10], axis=1)
                        whole_cur_agentview_rgb = np.concatenate(cur_agentview_rgb[::10], axis=1)
                        if whole_ori_agentview_rgb.shape[1] == whole_cur_agentview_rgb.shape[1]:
                            whole_ori_agentview_rgb, whole_cur_agentview_rgb = np.flipud(whole_ori_agentview_rgb), np.flipud(whole_cur_agentview_rgb)
                            all_agentview_rgb = np.concatenate((whole_ori_agentview_rgb, whole_cur_agentview_rgb), axis=0)
                            
                            os.makedirs(os.path.join(repo_name, f"{args.libero_task_suite}", f"{task.name}"), exist_ok=True)
                            Image.fromarray(all_agentview_rgb).save(os.path.join(repo_name, f"{args.libero_task_suite}", f"{task.name}",f"demo_{i}_{rep_idx}.jpg"))
                        time_cal_2 = time.time()
                        local_time_counter['show_diff'] += time_cal_2 - time_cal_1
                        
                    num_success += 1
                    real_success_demo_num += 1
                if demo_repeat_times == 1:
                    break
                    
            # Rest of the episode processing...
            # ... (Continue copying the original episode processing code)
            
    # Clean up
    if args.need_hdf5:
        new_data_file.close()
    orig_data_file.close()
    
    return local_time_counter, num_replays, num_success, num_noops

def main(args):
    number_demo_per_task = args.number_demo_per_task
    demo_repeat_times = args.demo_repeat_times
    
    need_color_change = args.need_color_change
    
    if args.specify_task_id is not None:
        assert args.num_tasks_in_suite == 1, "specify_task_id is not None, num_tasks_in_suite must be 1"
    
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
        if args.suffix is not None and args.suffix != "":
            repo_name = f"{user_name}/v-{viewpoint_rotate_min_interpolate_weight:.3f}-{viewpoint_rotate_max_interpolate_weight:.3f}-c-{color_scale_min_interpolate_weight:.3f}-{color_scale_max_interpolate_weight:.3f}-{args.suffix}"
        else:
            repo_name = f"{user_name}/v-{viewpoint_rotate_min_interpolate_weight:.3f}-{viewpoint_rotate_max_interpolate_weight:.3f}-c-{color_scale_min_interpolate_weight:.3f}-{color_scale_max_interpolate_weight:.3f}"
    else:
        if args.suffix is not None and args.suffix != "":
            repo_name = f"{user_name}/v-{viewpoint_rotate_min_interpolate_weight:.3f}-{viewpoint_rotate_max_interpolate_weight:.3f}-{args.suffix}"
        else:
            repo_name = f"{user_name}/v-{viewpoint_rotate_min_interpolate_weight:.3f}-{viewpoint_rotate_max_interpolate_weight:.3f}"
    
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

    # Setup global counters
    TIME_COUNTER = {
        'hdf5_gen': 0,
        'lerobot_gen': 0,
        'show_diff': 0,
    }

    if num_tasks_in_suite == task_suite.n_tasks:
        libero_home = args.libero_base_save_dir + "_full" + "_lerobot"
        task_id_list = list(range(num_tasks_in_suite))
    else:
        if args.specify_task_id is None:
            task_id_list = list(range(num_tasks_in_suite))
            libero_home = args.libero_base_save_dir + f"_{num_tasks_in_suite}" + "_lerobot"
        else:
            assert num_tasks_in_suite == 1, "specify_task_id is not None, num_tasks_in_suite must be 1"
            libero_home = args.libero_base_save_dir + "_1" + "_lerobot"
            task_id_list = [args.specify_task_id]
            repo_name = repo_name + f"_num{args.specify_task_id+1}"
        
    libero_home = Path(libero_home)
    lerobot_output_path = libero_home / repo_name
    
    hdf5_output_path = str(lerobot_output_path).replace("_lerobot/", "_hdf5/")
    os.makedirs(hdf5_output_path, exist_ok=True)
    
    if lerobot_output_path.exists():
        print(f"Warning: Dataset already exists at {lerobot_output_path}")
        if args.overwrite:
            print(f"Removing existing dataset")
            shutil.rmtree(lerobot_output_path)
        else:
            raise ValueError(f"Dataset already exists at {lerobot_output_path}. Use --overwrite to force regeneration.")
    
    os.makedirs(lerobot_output_path, exist_ok=True)
    
    # Multiprocessing - Process task_id_list in parallel
    print(f"Starting multiprocessing with {len(task_id_list)} processes")
    
    # Create process pool with number of processes equal to length of task_id_list
    num_processes = len(task_id_list)
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create a partial function with all the common arguments
    process_task_partial = partial(
        process_task,
        args=args,
        task_suite=task_suite,
        hdf5_output_path=hdf5_output_path,
        repo_name=repo_name,
        libero_home=libero_home,
        number_demo_per_task=number_demo_per_task,
        demo_repeat_times=demo_repeat_times,
        need_color_change=need_color_change,
        viewpoint_rotate_min=viewpoint_rotate_min,
        viewpoint_rotate_max=viewpoint_rotate_max,
        color_scale_min=color_scale_min,
        color_scale_max=color_scale_max,
        color_light_a=color_light_a,
        color_light_b=color_light_b,
        change_light=change_light,
        base_num=base_num,
        transparent_alpha=transparent_alpha,
        transparent_object_name=transparent_object_name
    )
    
    # Run the tasks in parallel and collect results
    results = pool.map(process_task_partial, task_id_list)
    
    # Process results
    num_replays = 0
    num_success = 0
    num_noops = 0
    
    for time_counter, task_replays, task_success, task_noops in results:
        # Update time counters
        for key in TIME_COUNTER:
            TIME_COUNTER[key] += time_counter[key]
        
        # Update other counters
        num_replays += task_replays
        num_success += task_success
        num_noops += task_noops
    
    # Print final statistics
    print(f"Total replays: {num_replays}")
    print(f"Total successes: {num_success}")
    print(f"Total no-ops filtered: {num_noops}")
    
    # Print time statistics
    print_time_counter(TIME_COUNTER)
    
    # Cleanup
    pool.close()
    pool.join()

    print(f"Dataset regeneration complete! Saved new dataset at: {lerobot_output_path}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


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
        type=int,
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset",
        default=False
    )
    
    
    args = parser.parse_args()
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
