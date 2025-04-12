"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

这里每次只是创建一个 island, 需要创建多个 island, 需要多次运行这个脚本
python experiments/robot/libero/regenerate_libero_lerobot_island_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --show_diff True \
    --vmin 0.25 \
    --vmax 0.25 \
    --cmin 0.25 \
    --cmax 0.25 \
    --num_tasks_in_suite 1 \
    --suffix "flip"


python experiments/robot/libero/regenerate_libero_lerobot_island_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --show_diff True \
    --vmin 0.25 \
    --vmax 0.25 \
    --cmin 0.25 \
    --cmax 0.25 \
    --num_tasks_in_suite 1 \
    --specify_task_id 0 \
    --suffix "flip"


python experiments/robot/libero/regenerate_libero_lerobot_island_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --show_diff True \
    --vmin 0.25 \
    --vmax 0.25 \
    --cmin 0.25 \
    --cmax 0.25 \
    --num_tasks_in_suite 1 \
    --suffix "flip-light" \
    --change_light True \
    --base_num 0.05


python experiments/robot/libero/regenerate_libero_lerobot_island_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --show_diff True \
    --vmin 0.75 \
    --vmax 0.75 \
    --cmin 0.75 \
    --cmax 0.75 \
    --num_tasks_in_suite 1 \
    --suffix "flip-light" \
    --change_light True \
    --base_num 0.05


python experiments/robot/libero/regenerate_libero_lerobot_island_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --show_diff True \
    --vmin 0.25 \
    --vmax 0.25 \
    --cmin 0.25 \
    --cmax 0.25 \
    --num_tasks_in_suite 1 \
    --specify_task_id 9 \
    --change_light False &

python experiments/robot/libero/regenerate_libero_lerobot_island_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial \
    --show_diff True \
    --vmin 0.75 \
    --vmax 0.75 \
    --cmin 0.75 \
    --cmax 0.75 \
    --num_tasks_in_suite 1 \
    --specify_task_id 9 \
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
    if args.specify_task_id is not None:
        assert args.num_tasks_in_suite == 1, "specify_task_id is not None, num_tasks_in_suite must be 1"
    
    print(f"Regenerating {args.libero_task_suite} dataset!")
    # transparent object
    transparent_alpha = 0.0
    transparent_object_name = "akita_black_bowl_2"

    # uniform distribution
    robot_base_name = 'robot0_link0'
    camera_name = "agentview"
    viewpoint_rotate_upper_bound = 90.0
    viewpoint_rotate_lower_bound = -10.0
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
    repo_name = f"{args.user_name}/v-{viewpoint_rotate_min_interpolate_weight:.2f}-{viewpoint_rotate_max_interpolate_weight:.2f}-c-{color_scale_min_interpolate_weight:.2f}-{color_scale_max_interpolate_weight:.2f}-{args.suffix}"
    
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
        libero_home = args.lerobot_home + "_full"
        task_id_list = list(range(num_tasks_in_suite))
    else:
        if args.specify_task_id is None:
            task_id_list = list(range(num_tasks_in_suite))
            # libero_home = args.lerobot_home + f"_{num_tasks_in_suite}"
            libero_home = args.lerobot_home + f"_{num_tasks_in_suite}"
        else:
            libero_home = args.lerobot_home + "_1"
            task_id_list = [args.specify_task_id]
            repo_name = repo_name + f"_num{args.specify_task_id+1}" # 这个很关键
        
    libero_home = Path(libero_home)
    lerobot_output_path = libero_home / repo_name
    if lerobot_output_path.exists():
        print(f"Removing existing dataset at {lerobot_output_path}")
        raise ValueError(f"Dataset already exists at {lerobot_output_path}")
        shutil.rmtree(lerobot_output_path)
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        root=lerobot_output_path,   # xyg added
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
            "observation.state": {      # robot states
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "observation.environment_state": {      # 不同的环境environment_state 的维度不一样
                "dtype": "float32",
                "shape": (92,),
                "names": ["state"],
            },
            # "task": {     这里不写task，但是 add_frame 的时候需要传入 task。否则就报错，这是奇了怪了
            #     "dtype": "string",
            #     "shape": (1,),
            #     "names": ["task"],
            # },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # task loop in task_suite
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"tasks-{args.libero_task_suite}"):
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
        for orig_data_key in tqdm.tqdm(orig_data.keys(), desc="episide in tasks"):  # demo_0, demo_1, ..., demo_49, ...
            # get demo data
            i = int(orig_data_key.split("_")[-1])
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]     #  The () is used to indicate that you want to read the entire dataset
            orig_states = demo_data["states"][()]       

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])      # [0] mean the state of the first timestep in this episode
            
            # 给定 min, max 均匀采样 viewpoint_rotate, color_scale
            viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
            color_scale = np.random.uniform(color_scale_min, color_scale_max)
            
            # recolor and rotate scene
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = rotate_recolor_dataset.recolor_and_rotate_scene(env, alpha=color_scale, color_light_a=color_light_a, color_light_b=color_light_b, 
                                                                 camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                                 theta=viewpoint_rotate, debug=False, need_change_light=change_light, base_num=base_num)
            
            # change the transparency of the transparent object
            env = rotate_recolor_dataset.change_object_transparency(env, object_name=transparent_object_name, alpha=transparent_alpha, debug=False)
            
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

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

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:        # 这里很机智，保证他是done的。数量可能会从原来的 50 下降到40出头
                """
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")     # ep 指的是 episode
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
                """
                for lerobot_idx in range(len(states)):
                    dataset.add_frame(
                        {
                            "observation.image": np.flipud(agentview_images[lerobot_idx]),
                            "observation.wrist_image": np.flipud(eye_in_hand_images[lerobot_idx]),
                            "observation.state": robot_states[lerobot_idx].astype(np.float32),   # 转化为 float32, numpy
                            "observation.environment_state": states[lerobot_idx].astype(np.float32),
                            "task": task_description,  # task_description.decode()
                            "action": actions[lerobot_idx].astype(np.float32), 
                        }
                    )
                
                dataset.save_episode()

                if args.show_diff:
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
                        
                        os.makedirs(os.path.join(repo_name.split("/")[-1], f"{args.libero_task_suite}", f"{task.name}"), exist_ok=True)
                        Image.fromarray(all_agentview_rgb).save(os.path.join(repo_name.split("/")[-1], f"{args.libero_task_suite}", f"{task.name}",f"demo_{i}.jpg"))
                
                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, "
                f"Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()

    print(f"Dataset regeneration complete! Saved new dataset at: {lerobot_output_path}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        required=True,
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help=("Path to directory containing raw HDF5 dataset. " "Example: ./LIBERO/libero/datasets/libero_spatial"),
        required=True,
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
        help="base num",
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
        "--lerobot_home",
        type=str,
        help="path to lerobot home",
        default="/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_lerobot_island"
        # "/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_lerobot_island"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        help="suffix of the dataset",
        default="flip"
    )
    
    parser.add_argument(
        "--user_name",
        type=str,
        help="user name",
        default="xyg"
    )
    
    args = parser.parse_args()
    args.show_diff = args.show_diff == "True"
    args.change_light = args.change_light == "True"

    # Start data regeneration
    import time
    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"Time taken: {int(hours)}:{int(minutes)}:{int(seconds)}")
