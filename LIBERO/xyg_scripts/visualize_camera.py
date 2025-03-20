"""
LIBERO环境摄像机控制工具

功能:
- 加载LIBERO环境
- 使用键盘直接在仿真窗口控制摄像机位置
- 使用欧拉角控制相机旋转（更直观）
- 按q退出程序
- 按g保存当前摄像机位置
- 按r重置相机到初始位置

使用方法:
    python libero_camera_control.py --libero_task_suite libero_spatial
"""

import argparse
import json
import os
import math
import numpy as np
import time
import cv2  # 使用OpenCV处理键盘输入
from rich import print

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import ControlEnv
import robosuite.utils.transform_utils as T  # 用于旋转变换

IMAGE_RESOLUTION = 256
CAMERA_NAME = "agentview"


def get_libero_env(task, model_family, resolution=256):
    """初始化并返回LIBERO环境及任务描述"""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution, 
        "hard_reset": False,
        "has_renderer": True, 
        "has_offscreen_renderer": False, 
        "use_camera_obs": False, 
        "render_camera": CAMERA_NAME
    }
    env = ControlEnv(**env_args)
    env.seed(0)
    return env, task_description


def quat_to_euler(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    alpha = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    beta = math.asin(2 * (qw * qy - qz * qx))
    gamma = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    
    euler = np.array([alpha, beta, gamma])
    
    return euler


def euler_to_quat(euler):
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    quat = np.array([qx, qy, qz, qw])
    return quat


def camera_change(env, camera_id):
    cur_camera_pos = env.sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = env.sim.model.cam_quat[camera_id].copy()
    cur_camera_euler = quat_to_euler(cur_camera_quat)

    delta_camera_pos = np.array([0.5, 0.5, 0.5])
    delta_camera_euler = np.array([0.0, 0.0, 0.0])
    
    tgt_camera_pos = cur_camera_pos + delta_camera_pos
    tgt_camera_quat = euler_to_quat(cur_camera_euler + delta_camera_euler)
    
    env.sim.model.cam_pos[camera_id] = tgt_camera_pos
    env.sim.model.cam_quat[camera_id] = tgt_camera_quat
    env.sim.forward()
    
    camera_img = env.sim.render(
        camera_name=CAMERA_NAME,  # 指定相机名称
        width=IMAGE_RESOLUTION,                  # 图像宽度
        height=IMAGE_RESOLUTION,                 # 图像高度
        depth=False,                # 是否需要深度图
        mode='offscreen'            # 离屏渲染模式
    )
    

def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    
    task = task_suite.get_task(0)
    env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
    
    env.reset()

    camera_id = env.sim.model.camera_name2id(CAMERA_NAME)
    
    camera_change(env, camera_id)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO任务套件。例如: libero_spatial",
        required=False,
        default="libero_spatial",
    )
    args = parser.parse_args()

    # 启动程序
    main(args)