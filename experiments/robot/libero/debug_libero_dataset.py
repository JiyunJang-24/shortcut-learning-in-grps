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
    print(f"{args.libero_task_suite} dataset!")

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # task loop in task_suite
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        import ipdb; ipdb.set_trace()
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        data_path = os.path.join(args.libero_dir, f'{args.libero_task_suite}', f"{task.name}_demo.hdf5")
        assert os.path.exists(data_path), f"Cannot find raw data file {data_path}."
        data_file = h5py.File(data_path, "r")
        data = data_file["data"]


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        default="libero_spatial",
    )
    parser.add_argument(
        "--libero_dir",
        type=str,
        default="LIBERO/dataset",
    )    
    args = parser.parse_args()

    main(args)
