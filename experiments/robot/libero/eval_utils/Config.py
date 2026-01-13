from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
    use_plucker: bool = False
    use_dynamics_basis: bool = False
    apply_basis_scale: bool = False
    camera_scale: float = 1.0
    for_dp: bool = True
    vla_mode: str = "vanilla"
