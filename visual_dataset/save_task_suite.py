import os
os.environ['PRISMATIC_DATA_ROOT'] = "/mnt/nfs/CMG/xiejunlin/datasets/Robotics/libero"

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image

import wandb
import mediapy

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class Config:
    task_suite_name: str = 'libero_spatial'
    num_trials_per_task: int = 2
    n_tasks: int = 3
    seed: int = 42
    num_steps_wait: int = 10
    resize_size: int = 224
    save_dir: str = 'datasets_vis/env_init'

@draccus.wrap()
def main(cfg: Config) -> None:
    task_suite_name = cfg.task_suite_name
    resize_size = cfg.resize_size
    num_trials_per_task = cfg.num_trials_per_task
    num_steps_wait = cfg.num_steps_wait

    base_save_img_dir = f'{cfg.save_dir}/{task_suite_name}'
    os.makedirs(base_save_img_dir, exist_ok=True)
    
    # 先拿task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = min(task_suite.n_tasks, cfg.n_tasks)

    # 遍历task suite中的每个task
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        save_img_dir = f'{base_save_img_dir}/{task_id}'
        os.makedirs(save_img_dir, exist_ok=True)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, '', resolution=resize_size)
        
        # 保存task description到txt
        task_description_path = f'{save_img_dir}/task_description.txt'
        with open(task_description_path, 'w') as f:
            f.write(task_description)
        
        # 每个task还可以遍历它的episode
        images = []
        cur_task_trials = min(num_trials_per_task, len(initial_states))
        for episode_idx in tqdm.tqdm(range(cur_task_trials)):
            print(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            t = 0
            while t < num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(''))
                t += 1
                images.append(get_libero_image(obs, resize_size))
                continue
        
            image = get_libero_image(obs, resize_size)
            images.append(image)
        
            # List[np.array] 通过 mediapy 转化为 video
            mediapy.write_video(f'{save_img_dir}/video_{episode_idx:02d}.gif', images, fps=10, codec='gif')        
            
        
        
        
if __name__ == "__main__":
    main()


