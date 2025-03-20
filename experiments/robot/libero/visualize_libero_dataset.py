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
        --libero_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

"""

import argparse
import json
import os
os.environ["PRISMATIC_DATA_ROOT"] = "/mnt/nfs/CMG/xiejunlin/datasets/Robotics/libero"
import cv2
import mediapy as media

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

IMAGE_RESOLUTION = 256


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    os.makedirs(args.video_save_dir, exist_ok=True)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # task loop in task_suite
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        if args.no_noops:
            orig_data_path = os.path.join(args.libero_dir, f'{args.libero_task_suite}_no_noops', f"{task.name}_demo.hdf5")
        else:
            orig_data_path = os.path.join(args.libero_dir, f'{args.libero_task_suite}', f"{task.name}_demo.hdf5")
            
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]
        
        # Create video save directory, 并将task_description保存为txt文件
        task_save_dir = os.path.join(args.video_save_dir, f'{args.libero_task_suite}', f'{task.name}')
        os.makedirs(task_save_dir, exist_ok=True)
        with open(os.path.join(task_save_dir, 'task_description.txt'), 'w') as f:
            f.write(task_description)
        
        # loop episodes in task
        for i in range(len(orig_data.keys())):
            # Get demo data
            if f"demo_{i}" not in orig_data.keys(): # 可能被filter掉了
                continue
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]     #  The () is used to indicate that you want to read the entire dataset
            orig_states = demo_data["states"][()]       

            # print(args.libero_task_suite)
            agentview_rgb = demo_data["obs"]["agentview_rgb"][()]   
    
            # 将agentview_rgb保存为视频, 要求是mp4格式
            # agentview_rgb: numpy.ndarray, shape: (episode_length, 256, 256, 3)
            # 我们等间隔选取20帧保存为视频
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_save_path = os.path.join(args.video_save_dir, f'{args.libero_task_suite}', f'{task.name}', f'demo_{i}.gif')
            # video_writer = cv2.VideoWriter(video_save_path, fourcc, 5, (agentview_rgb.shape[1], agentview_rgb.shape[2]))
            # for j in range(0, agentview_rgb.shape[0], 20):
            #     # agentview_rgb[j]: numpy.ndarray, shape: (256, 256, 3)
            #     # rgb -> bgr
            #     video_writer.write(cv2.cvtColor(agentview_rgb[j], cv2.COLOR_RGB2BGR))
            # video_writer.write(cv2.cvtColor(agentview_rgb[-1], cv2.COLOR_RGB2BGR))
            # video_writer.release()
            agentview_rgb_save_list = []
            for j in range(0, agentview_rgb.shape[0], 20):
                agentview_rgb_save_list.append(agentview_rgb[j])
            agentview_rgb_save_list.append(agentview_rgb[-1])
            # list of numpy.ndarray, shape: (256, 256, 3) -> stack (B, H, W, C). new axis
            agentview_rgb_save_array = np.stack(agentview_rgb_save_list, axis=0)
            agentview_rgb_save_array = agentview_rgb_save_array[:, ::-1, :]  # flip the image(W)
            media.write_video(video_save_path, agentview_rgb_save_array, fps=5, codec='gif')
            # print(f"Save video to {video_save_path}")
    
        # import ipdb; ipdb.set_trace()
        # print('this is the debug point')    


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
        "--libero_dir",
        type=str,
        help=("Path to regenerated dataset directory. " "Example: ./LIBERO/libero/datasets/libero_spatial_no_noops"),
        required=True,
    )
    
    parser.add_argument(
        "--video_save_dir",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--no_noops",
        default="False",
        help="Whether to use the no_noops version of the dataset.",
    )
    args = parser.parse_args()
    args.no_noops = args.no_noops == "True"

    # Start data regeneration
    main(args)
