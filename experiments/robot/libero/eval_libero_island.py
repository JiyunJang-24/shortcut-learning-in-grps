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
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

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
LEROBOT_HOME = Path("/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_lerobot_island_full")
REPO_NAME = "xyg/v-0.25-0.25-c-0.25-0.25"
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds



def interpolate_number(number_min, number_max, interpolate_weight):
    return number_min + (number_max - number_min) * interpolate_weight

def main(args):
    
    #################################################################################################################
    # model policy parameters
    #################################################################################################################
    # "lerobot/diffusion_pusht"
    pretrained_policy_path = "/mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-02-51_diffusion/checkpoints/020000/pretrained_model"
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    
    #################################################################################################################
    # environment parameters
    #################################################################################################################
    num_tasks_in_suite, eval_episode_num = 1, 10
    print(f"testing {args.libero_task_suite} dataset!")
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
    
    viewpoint_rotate_min_interpolate_weight = 0.25
    viewpoint_rotate_max_interpolate_weight = 0.25
    color_scale_min_interpolate_weight = 0.25
    color_scale_max_interpolate_weight = 0.25
    
    viewpoint_rotate_min = interpolate_number(viewpoint_rotate_lower_bound, viewpoint_rotate_upper_bound, viewpoint_rotate_min_interpolate_weight)
    viewpoint_rotate_max = interpolate_number(viewpoint_rotate_lower_bound, viewpoint_rotate_upper_bound, viewpoint_rotate_max_interpolate_weight)
    color_scale_min = interpolate_number(color_scale_lower_bound, color_scale_upper_bound, color_scale_min_interpolate_weight)
    color_scale_max = interpolate_number(color_scale_lower_bound, color_scale_upper_bound, color_scale_max_interpolate_weight)
    print(f"v: {viewpoint_rotate_min_interpolate_weight}->{viewpoint_rotate_min}, {viewpoint_rotate_max_interpolate_weight}->{viewpoint_rotate_max}, "
          f"c: {color_scale_min_interpolate_weight}->{color_scale_min}, {color_scale_max_interpolate_weight}->{color_scale_max}")

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"tasks-{args.libero_task_suite}"):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)


        for episode_idx in range(eval_episode_num):

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            # env.set_init_state(orig_states[0])      # [0] mean the state of the first timestep in this episode
            
            # 给定 min, max 均匀采样 viewpoint_rotate, color_scale
            viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
            color_scale = np.random.uniform(color_scale_min, color_scale_max)
            
            # recolor and rotate scene
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = rotate_recolor_dataset.recolor_and_rotate_scene(env, alpha=color_scale, color_light_a=color_light_a, color_light_b=color_light_b, 
                                                                 camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                                 theta=viewpoint_rotate, debug=False)
            
            # change the transparency of the transparent object
            env = rotate_recolor_dataset.change_object_transparency(env, object_name=transparent_object_name, alpha=transparent_alpha, debug=False)
            
            done = False
            while not done:
                action = policy.get_action(obs)
                obs, reward, done, info = env.step(action.tolist())

                
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
                            "observation.image": agentview_images[lerobot_idx],
                            "observation.wrist_image": eye_in_hand_images[lerobot_idx],
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
                        whole_ori_agentview_rgb, whole_cur_agentview_rgb = whole_ori_agentview_rgb[::-1], whole_cur_agentview_rgb[::-1]
                        all_agentview_rgb = np.concatenate((whole_ori_agentview_rgb, whole_cur_agentview_rgb), axis=0)
                        # 将all_agentview_rgb:(H, W, 3)保存为图片
                        
                        os.makedirs(os.path.join("visualize-v-0.25-0.25-c-0.25-0.25-full", f"{args.libero_task_suite}", f"{task.name}"), exist_ok=True)
                        Image.fromarray(all_agentview_rgb).save(os.path.join("visualize-v-0.25-0.25-c-0.25-0.25-full", f"{args.libero_task_suite}", f"{task.name}",f"demo_{i}.jpg"))
                
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

    args = parser.parse_args()

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
