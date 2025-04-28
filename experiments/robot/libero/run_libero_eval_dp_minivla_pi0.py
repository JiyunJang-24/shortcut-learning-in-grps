"""
run_libero_eval_dp_minivla.py for diffusion policy and minivla model

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""


import subprocess

def run_cmd(cmd):
    out = (subprocess.check_output(cmd, shell=True)).decode('utf-8')[:-1]
    return out

def get_free_gpu_indices(num_gpus=1, debug=False):
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    total_gpu_memory = []
    reserved_gpu_memory = []
    used_gpu_memory = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])
        total_gpu_memory.append(int([l.strip().split(':')[-1].strip().split(' ')[0].strip() for l in out[i*5+2:i*5+3]][0]))
        reserved_gpu_memory.append(int([l.strip().split(':')[-1].strip().split(' ')[0].strip() for l in out[i*5+3:i*5+4]][0]))
        used_gpu_memory.append(int([l.strip().split(':')[-1].strip().split(' ')[0].strip() for l in out[i*5+4:i*5+5]][0]))  
        
    remaining_gpu_memory = [total_gpu_memory[i] - used_gpu_memory[i] - reserved_gpu_memory[i] for i in range(total_gpu_num)]
    for i in range(total_gpu_num):
        tmp_memory = used_gpu_memory[i]+reserved_gpu_memory[i]
        if debug:
            print(f"GPU {i} {tmp_memory}/{total_gpu_memory[i]}")
        
    # 选择剩余显存最大的num_gpus个GPU
    sorted_gpu_indices = sorted(range(total_gpu_num), key=lambda i: remaining_gpu_memory[i], reverse=True)
    return sorted_gpu_indices[:num_gpus]

gpu_indices = get_free_gpu_indices()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_indices))
os.environ["PRISMATIC_DATA_ROOT"] = "/mnt/hdd3/xingyouguang/datasets/robotics/libero"
import torch
if os.environ.get('MEMORY_SIZE', None) == 'small':
    memory_x = torch.ones(1, 1, 1).cuda()
else:
    memory_x = torch.ones(6*8*8, 1024, 1024).cuda()

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import time
import glob

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import torch
import wandb
import shutil
import collections
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    save_rollout_video_dir,
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

import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import LIBERO.xyg_scripts.rotate_recolor_dataset as rotate_recolor_dataset
IMAGE_RESOLUTION = 256
def interpolate_number(number_min, number_max, interpolate_weight):
    return number_min + (number_max - number_min) * interpolate_weight

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "diffusion"                    # Model family
    hf_token: str = Path(".hf_token")                     
    # Pretrained checkpoint path
    pretrained_checkpoint: Union[str, Path] = "/mnt/hdd3/xingyouguang/projects/robotics/lerobot/outputs/train/2025-03-26/21-24-06_diffusion/checkpoints/030000/pretrained_model"
    
    
    # no use for next 5 lines
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite.
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task 50
    num_tasks_in_suite: int = 10
    
    viewpoint_rotate_min_interpolate_weight: float = 0.25
    viewpoint_rotate_max_interpolate_weight: float = 0.25
    color_scale_min_interpolate_weight: float = 0.25
    color_scale_max_interpolate_weight: float = 0.25
    
    viewpoint_rotate_upper_bound: float = 90.0
    viewpoint_rotate_lower_bound: float = -10.0
    need_color_change: bool = True
    color_light_a = [1.0, 0.0, 0.0]
    color_light_b = [1.0, 1.0, 0.0]
    color_scale_upper_bound = 1.0
    color_scale_lower_bound = 0.0

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    re_eval: bool = False
    change_light: bool = False
    base_num: float = 0.05
    
    specific_task_id: int = None
    
    server_host: str = "localhost"
    server_port: int = 8000
    replan_steps: int = 5


def check_eval_finish(local_log_filepath):
    base_path = os.path.dirname(local_log_filepath)
    all_txt_files = glob.glob(os.path.join(base_path, "*.txt"))
    
    skip_len = len(f"{DATE_TIME}.txt")
    
    similar_txt_files = [f_path for f_path in all_txt_files if f_path[:-skip_len] == local_log_filepath[:-skip_len]]
    if len(similar_txt_files) == 0:
        return False, local_log_filepath
    
    assert len(similar_txt_files) == 1, f"Found {len(similar_txt_files)} similar txt files, expected 1"
    
    # 有现成的 log file，就不要换来换去了
    similar_txt_file = similar_txt_files[0]
    with open(similar_txt_file, "r") as f:
        lines = f.readlines()
        if len(lines) > 0 and "Total time taken: " in lines[-1]:
            return True, similar_txt_file
    return False, similar_txt_file

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    
    already_check_flag, ori_log_path = check_eval_finish(local_log_filepath)
    if already_check_flag and not cfg.re_eval:
        print('#' * 40)
        print(f"Evaluation finished, skipping {local_log_filepath}")
        print('#' * 40)
        return

    if not already_check_flag and ori_log_path != local_log_filepath:
        # 这个是之前log没写完的，应该将 ori_log_path 删除
        os.remove(ori_log_path)
        video_dir_path = ori_log_path[:-4]
        if os.path.exists(video_dir_path):
            shutil.rmtree(video_dir_path)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    global memory_x
    del memory_x
    if cfg.model_family == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        pretrained_policy_path = cfg.pretrained_checkpoint
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    elif cfg.model_family in ["pi0", "pi0_fast"]:
        policy = websocket_client_policy.WebsocketClientPolicy(host=cfg.server_host, port=cfg.server_port)
        model = policy
    elif cfg.model_family in ["openvla", "prismatic"]:
        policy = get_model(cfg)
        model = policy
    else:
        raise ValueError(f"Unexpected `model_family` found in config ({cfg.model_family}).")
    
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if len(model.norm_stats) == 2 and "libero_spatial" in list(model.norm_stats.keys())[0] and "libero_spatial" in list(model.norm_stats.keys())[1]:
            cfg.unnorm_key = sorted(list(model.norm_stats.keys()))[0]
        else:
            if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
        
    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        
    log_file = open(local_log_filepath, "w", buffering=1)  # Enable line buffering
    print(f"Logging to local log file: {local_log_filepath}")
    
    # 将 cfg 中的参数写入 log_file, 包括 cfg 的每个属性
    for attr, value in cfg.__dict__.items():
        log_file.write(f"{attr}: {value}\n")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    #################################################################################################################
    # environment parameters
    #################################################################################################################
    num_tasks_in_suite = cfg.num_tasks_in_suite
    print(f"testing {cfg.task_suite_name} dataset!")
    # transparent object
    transparent_alpha = 0.0
    transparent_object_name = "akita_black_bowl_2"

    # uniform distribution
    robot_base_name = 'robot0_link0'
    camera_name = "agentview"
    viewpoint_rotate_upper_bound = cfg.viewpoint_rotate_upper_bound
    viewpoint_rotate_lower_bound = cfg.viewpoint_rotate_lower_bound
    need_color_change = cfg.need_color_change
    color_light_a = np.array(cfg.color_light_a)
    color_light_b = np.array(cfg.color_light_b)
    color_scale_upper_bound = cfg.color_scale_upper_bound
    color_scale_lower_bound = cfg.color_scale_lower_bound
    
    viewpoint_rotate_min_interpolate_weight = cfg.viewpoint_rotate_min_interpolate_weight
    viewpoint_rotate_max_interpolate_weight = cfg.viewpoint_rotate_max_interpolate_weight
    color_scale_min_interpolate_weight = cfg.color_scale_min_interpolate_weight
    color_scale_max_interpolate_weight = cfg.color_scale_max_interpolate_weight
    
    change_light = cfg.change_light
    base_num = cfg.base_num
    
    viewpoint_rotate_min = interpolate_number(viewpoint_rotate_lower_bound, viewpoint_rotate_upper_bound, viewpoint_rotate_min_interpolate_weight)
    viewpoint_rotate_max = interpolate_number(viewpoint_rotate_lower_bound, viewpoint_rotate_upper_bound, viewpoint_rotate_max_interpolate_weight)
    color_scale_min = interpolate_number(color_scale_lower_bound, color_scale_upper_bound, color_scale_min_interpolate_weight)
    color_scale_max = interpolate_number(color_scale_lower_bound, color_scale_upper_bound, color_scale_max_interpolate_weight)
    print(f"v: {viewpoint_rotate_min_interpolate_weight}->{viewpoint_rotate_min}, {viewpoint_rotate_max_interpolate_weight}->{viewpoint_rotate_max}, "
          f"c: {color_scale_min_interpolate_weight}->{color_scale_min}, {color_scale_max_interpolate_weight}->{color_scale_max}")
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = min(num_tasks_in_suite, task_suite.n_tasks)
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    if cfg.model_family == "diffusion":
        resize_size = get_image_resize_size(cfg) if IMAGE_RESOLUTION is None else IMAGE_RESOLUTION
    elif cfg.model_family in ["pi0", "pi0_fast", "openvla", "prismatic"]:
        resize_size = get_image_resize_size(cfg)
    else:
        raise ValueError(f"Unexpected `model_family` found in config ({cfg.model_family}).")

    # Start evaluation. 测试的时候，为了并行化处理，这里写的比较简单，但是生成数据集这里比较复杂，因为并行化操作卸载 python multi_process 中
    start_time = time.time()
    total_episodes, total_successes = 0, 0
    if cfg.specific_task_id is not None:
        assert num_tasks_in_suite == 1, f"num_tasks_in_suite must be 1 when specific_task_id is not None, but got {num_tasks_in_suite}"
        task_ids = [cfg.specific_task_id]
    else:
        task_ids = list(range(num_tasks_in_suite))

    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            if cfg.model_family == "diffusion":  # 因为 diffusion policy 数据 horizon 写在模型部分，需要reset, minivla 数据是自己处理，pi0则是没有history image
                policy.reset()
            elif cfg.model_family in ["pi0", "pi0_fast"]:
                action_plan = collections.deque()

            viewpoint_rotate = np.random.uniform(viewpoint_rotate_min, viewpoint_rotate_max)
            color_scale = np.random.uniform(color_scale_min, color_scale_max)
            
            # recolor and rotate scene
            camera_id = env.sim.model.camera_name2id(camera_name)
            if need_color_change:
                env = rotate_recolor_dataset.recolor_and_rotate_scene(env, alpha=color_scale, color_light_a=color_light_a, color_light_b=color_light_b, 
                                                                     camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                                                     theta=viewpoint_rotate, debug=False, need_change_light=change_light, base_num=base_num)
            else:
                env = rotate_recolor_dataset.rotate_camera(env=env, camera_id=camera_id, camera_name=camera_name, 
                                                           robot_base_name=robot_base_name, theta=viewpoint_rotate, debug=False)
            
            # change the transparency of the transparent object
            env = rotate_recolor_dataset.change_object_transparency(env, object_name=transparent_object_name, alpha=transparent_alpha, debug=False)
            
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    
                    if cfg.model_family == "diffusion":
                        # Get preprocessed image, 这里image已经是flip了，他和咱们训练的 xyg/v-0.xx-0.xx-0.xx-0.xx-flip 一致
                        replay_images.append(np.flipud(obs["agentview_image"]).copy())
                        image = torch.from_numpy(np.flipud(obs["agentview_image"]).copy())
                        state = torch.from_numpy(np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]))
                        state = state.to(torch.float32)
                        image = image.to(torch.float32) / 255
                        image = image.permute(2, 0, 1)  # H, W, C -> C, H, W
                        # Send data tensors from CPU to GPU
                        state = state.to('cuda', non_blocking=True)
                        image = image.to('cuda', non_blocking=True)

                        # Add extra (empty) batch dimension, required to forward the policy
                        state = state.unsqueeze(0)
                        image = image.unsqueeze(0)

                        # Create the policy input dictionary
                        observation = {
                            "observation.state": state,
                            "observation.image": image,
                        }

                        with torch.inference_mode():
                            action = policy.select_action(observation)

                        # Prepare the action for the environment
                        numpy_action = action.squeeze(0).to("cpu").numpy()

                        # Execute action in environment
                        obs, reward, done, info = env.step(numpy_action)
                    elif cfg.model_family in ["openvla", "prismatic"]:
                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        # use_wrist_image
                        if cfg.use_wrist_image:
                            wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image")
                            replay_wrist_images.append(wrist_img)

                        # buffering #obs_history images, optionally
                        image_history = replay_images[-cfg.obs_history :]
                        if len(image_history) < cfg.obs_history:
                            image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                        # same but for optional wrist images
                        if cfg.use_wrist_image:
                            wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                            if len(wrist_image_history) < cfg.obs_history:
                                wrist_image_history.extend(
                                    [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                                )
                            # interleaved images [... image_t, wrist_t ...]
                            image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": image_history,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        # Query model to get action
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )

                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                        if cfg.model_family in ["openvla", "prismatic"]:
                            action = invert_gripper_action(action)

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                    elif cfg.model_family in ["pi0", "pi0_fast"]:
                        img = np.ascontiguousarray(obs["agentview_image"][::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])

                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, resize_size, resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
                        )
                        state = np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        )
                        replay_images.append(img)
                        
                        if not action_plan:
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": state,
                                "prompt": str(task_description),
                            }
                            
                            action_chunk = model.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= cfg.replan_steps
                            ), f"We want to replan every {cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: cfg.replan_steps])
                        action = action_plan.popleft()
                        obs, reward, done, info = env.step(action.tolist())
                        
                    else:
                        raise ValueError("123, 123")
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video_dir(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file,
                task_id=task_id, episode_idx=episode_idx, task_suite_name=cfg.task_suite_name, local_log_filepath=local_log_filepath
            )

            # Save the videos to wandb
            if cfg.use_wandb and (task_successes < 10 or task_episodes - task_successes < 10):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60 
    print(f"Total time taken: {hours} hours, {minutes} minutes, {seconds} seconds")
    log_file.write(f"Total time taken: {hours} hours, {minutes} minutes, {seconds} seconds\n")
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
