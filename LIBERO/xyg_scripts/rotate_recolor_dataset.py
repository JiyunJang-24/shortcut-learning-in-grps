"""_summary_
API differences need to be noted when converting rotations between the simulator and the controller. For example,

MuJoCo API uses an [x, y, z, w] quaternion representation as shown here
Whereas, Drake uses the [w, x, y, z] ordering as described here

Returns:
    _type_: _description_

现在这个代码不能动了，因为我后面会调用它修改环境
"""
import argparse
import json
import os
import math
import numpy as np
import time
import cv2  # 使用OpenCV处理键盘输入
from rich import print
from PIL import Image
from itertools import product

import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import transforms3d.axangles as axangles

from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import ControlEnv
import robosuite.utils.transform_utils as T  # 用于旋转变换


IMAGE_RESOLUTION = 512
CAMERA_NAME = "agentview"
# IMAGE_SAVE_PATH = "LIBERO/xyg_scripts/image_transparency_example"
IMAGE_SAVE_PATH = "tmp_dir2"
MUJOCO_WXYZ = True
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)


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
    return euler.quat2euler(quat, "sxyz")


def euler_to_quat(euler_angles):
    return euler.euler2quat(euler_angles[0], euler_angles[1], euler_angles[2], "sxyz")


class Pose:
    # must be wxyz quaterion format
    def __init__(self, position=None, orientation=None):
        if position is None:
            position = np.zeros(3)  # 默认位置为原点
        if orientation is None:
            orientation = np.array([1.0, 0, 0, 0])

        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def get_position(self):
        return self.position

    def get_orientation(self):
        return self.orientation

    def set_position(self, position):
        self.position = np.array(position)

    def set_orientation(self, orientation):
        self.orientation = np.array(orientation)

    def transform(self, pose):
        """将当前Pose与另一个Pose相乘，返回组合后的Pose"""
        new_position = self.position + quaternions.quat2mat(self.orientation) @ pose.position
        new_orientation = quaternions.qmult(self.orientation, pose.orientation)
        return Pose(new_position, new_orientation)

    def orientation_inv(self):
        orientation = self.orientation
        orientation[1:] *= -1   # wxyz, 虚部求负数
        return orientation

    def inverse(self):
        """返回当前Pose的逆"""
        return Pose(-1 * quaternions.quat2mat(self.orientation_inv()) @ self.position, self.orientation_inv())


def rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, robot_base_pos, robot_base_quat, theta):
    """
        given wxyz quaterion format; camera pose rotate around robot base;
    """
    # camera pose and robot base pose in world frame
    camera_world_pose = Pose(cur_camera_pos, cur_camera_quat)
    robot_world_pose = Pose(robot_base_pos, robot_base_quat)
    
    robot_robot_pose = Pose(np.array([0, 0, 0]), np.array([1, 0, 0, 0]))
    
    world_to_robot_transform = robot_robot_pose.transform(robot_world_pose.inverse())
    
    # camera pose in robot frame
    camera_robot_pose = world_to_robot_transform.transform(camera_world_pose)
    
    # 沿着 z 轴旋转 theta 度
    new_camera_robot_pose = Pose(np.array([0, 0, 0]), euler.euler2quat(0, 0, theta / 180 * np.pi)).transform(camera_robot_pose)
    
    # camera pose in world frame
    new_camera_world_pose = world_to_robot_transform.inverse().transform(new_camera_robot_pose)
    
    return new_camera_world_pose.get_position(), new_camera_world_pose.get_orientation()
    

def rotate_camera(env, camera_id, camera_name, robot_base_name="robot0_base", theta=0.0, debug=False):
    cur_camera_pos = env.sim.model.cam_pos[camera_id].copy()
    cur_camera_quat = env.sim.model.cam_quat[camera_id].copy()
    
    robot_base_id = env.sim.model.body_name2id(robot_base_name)
    robot_base_pos = env.sim.model.body_pos[robot_base_id].copy()
    robot_base_quat = env.sim.model.body_quat[robot_base_id].copy()
    
    if not MUJOCO_WXYZ: # xyzw -> wxyz
        cur_camera_quat = np.array([cur_camera_quat[3], cur_camera_quat[0], cur_camera_quat[1], cur_camera_quat[2]])
        robot_base_quat = np.array([robot_base_quat[3], robot_base_quat[0], robot_base_quat[1], robot_base_quat[2]])
    
    tgt_camera_pos, tgt_camera_quat = rotate_camera_based_on_robot_base(cur_camera_pos, cur_camera_quat, 
                                                                        robot_base_pos, robot_base_quat, theta)
    
    if not MUJOCO_WXYZ: # wxyz -> xyzw
        tgt_camera_quat = np.array([tgt_camera_quat[1], tgt_camera_quat[2], tgt_camera_quat[3], tgt_camera_quat[0]])
    
    env.sim.model.cam_pos[camera_id] = tgt_camera_pos
    env.sim.model.cam_quat[camera_id] = tgt_camera_quat
    env.sim.forward()
    
    if debug:
        camera_img = env.sim.render(
            camera_name=camera_name,  # 指定相机名称
            width=IMAGE_RESOLUTION,                  # 图像宽度
            height=IMAGE_RESOLUTION,                 # 图像高度
            depth=False,                # 是否需要深度图
            mode='offscreen'            # 离屏渲染模式
        )
        
        Image.fromarray(camera_img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"rotate_{theta:.2f}.png"))
    return env


def color_interpolation(color_a, color_b, alpha):
    return color_a * alpha + color_b * (1 - alpha)


def change_env_light(env, light_id, color, specular, ambient, diffuse, active):
    if light_id is None:
        if env.sim.model.nlight > 0:
            light_id = 0
            print(f"使用默认光源 (ID: {light_id})")
        else:
            print("环境中没有光源")
            return False
    
    # 修改光源颜色
    if color is not None:
        env.sim.model.light_specular[light_id] = color
        env.sim.model.light_diffuse[light_id] = color
        env.sim.model.light_ambient[light_id] = [c * 0.1 for c in color]  # 环境光通常较弱
        # print(f"光源颜色设置为 {color}")
        
    # 单独修改各光照分量
    if specular is not None:
        if isinstance(specular, float):
            env.sim.model.light_specular[light_id] *= specular
        else:
            env.sim.model.light_specular[light_id] = specular
        # print(f"镜面反射设置为 {specular}")
    
    if ambient is not None:
        if isinstance(ambient, float):
            env.sim.model.light_ambient[light_id] *= ambient
        else:
            env.sim.model.light_ambient[light_id] = ambient
        # print(f"环境光设置为 {ambient}")
    
    if diffuse is not None:
        if isinstance(diffuse, float):
            env.sim.model.light_diffuse[light_id] *= diffuse
        else:
            env.sim.model.light_diffuse[light_id] = diffuse
        # print(f"漫反射设置为 {diffuse}")
    
    # 开启/关闭光源
    if active is not None:
        env.sim.model.light_active[light_id] = 1 if active else 0
        # print(f"光源已{'激活' if active else '关闭'}")


def recolor_scene(env, alpha, color_light_a, color_light_b, need_print_all_light=False, debug=False, need_change_light=False, base_num=0.1):    
    light_name_list = ["light1", "light2"]
    light_id_list = [env.sim.model.light_name2id(light_name) for light_name in light_name_list]
    
    light_id = None
    color = color_interpolation(color_light_a, color_light_b, alpha)
    
    
    if need_change_light:
        factor = 1.0
        specular, ambient, diffuse = base_num + alpha * factor, base_num + alpha * factor, base_num + alpha * factor
        active = None
    else:
        specular, ambient, diffuse, active = None, None, None, None
    
    # 获取光源ID
    if need_print_all_light and env.sim.model.nlight > 0:
        print("可用光源:")
        for i in range(env.sim.model.nlight):
            name = env.sim.model.light_id2name(i) if hasattr(env.sim.model, "light_id2name") else f"light_{i}"
            print(f"  ID {i}: {name}")
    
    # 如果没有提供ID或名称，默认使用第一个光源
    for light_id in light_id_list:
        change_env_light(env, light_id, color, specular, ambient, diffuse, active)
    
    # 更新物理状态
    env.sim.forward()
    
    # 渲染一张图像验证效果
    if debug:
        img = env.sim.render(
            width=IMAGE_RESOLUTION, 
            height=IMAGE_RESOLUTION, 
            camera_name=CAMERA_NAME
        )
        
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"basenum_{base_num:.2f}_color_{alpha:.2f}_light_{need_change_light}.png"))
    
    return env
    

def recolor_and_rotate_scene(env, alpha, color_light_a, color_light_b, camera_id, 
                             camera_name, robot_base_name, theta, debug=True, need_change_light=False, base_num=0.5):
    env = recolor_scene(env, alpha, color_light_a, color_light_b, need_change_light=need_change_light, base_num=base_num)
    env = rotate_camera(env, camera_id, camera_name, robot_base_name, theta)
    
    if debug:
        env.sim.forward()
        img = env.sim.render(
            width=IMAGE_RESOLUTION, 
            height=IMAGE_RESOLUTION, 
            camera_name=CAMERA_NAME
        )
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"light_{alpha:.2f}_rotate_{theta:.2f}.png"))
    return env


def change_object_transparency(env, object_name, alpha=1.0, debug=False):
    object_ids = []
    for i, name in enumerate(env.sim.model.body_names):
        if name and object_name in name:
            object_ids.append(i)
    
    if not object_ids:
        raise ValueError(f"找不到名称包含 '{object_name}' 的物体!")

    if alpha is not None:
        for object_id in object_ids:
            # 获取该物体的所有几何体
            for i in range(env.sim.model.ngeom):
                if env.sim.model.geom_bodyid[i] == object_id:
                    geom_id = i
                    current_rgba = env.sim.model.geom_rgba[geom_id].copy()
                    # 修改透明度
                    if alpha is not None:
                        current_rgba[3] = alpha
                    
                    env.sim.model.geom_rgba[geom_id] = current_rgba
    
    env.sim.forward()
    
    img = env.sim.render(
        width=IMAGE_RESOLUTION, 
        height=IMAGE_RESOLUTION, 
        camera_name=CAMERA_NAME
    )
    
    if debug:
        Image.fromarray(img[::-1]).save(os.path.join(IMAGE_SAVE_PATH, f"object_observation_transparency_{alpha:.2f}.png"))
    
    return env
    
    

def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task = task_suite.get_task(0)
    
    robot_base_name = 'robot0_link0'
    camera_name = CAMERA_NAME
    theta_list = [-180, -150, -120, -90, -60, -30, -10, 0, 10, 30, 60, 90, 120, 150, 180]
    color_light_a = np.array([1.0, 0.0, 0.0])
    color_light_b = np.array([1.0, 1.0, 0.0])
    alpha_list = np.linspace(0, 1, 5)
    
    theta_list = [-10, 0, 10]
    alpha_list = np.linspace(0, 1, 3)
    
    object_name = "akita_black_bowl_2"
    transparency_list = np.linspace(0, 1, 5)
    
    # rotate the camera around the robot base;
    need_rotate_camera = False
    if need_rotate_camera:
        for theta in theta_list:
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = rotate_camera(env, camera_id, camera_name, robot_base_name, theta=theta)

    # recolor the scene;
    need_recolor_scene = True
    need_change_light = True
    if need_recolor_scene:
        for base_num in np.linspace(0.05, 0.95, 8):
            for transparency in alpha_list:
                env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
                env.reset()
                env = recolor_scene(env, alpha=transparency, color_light_a=color_light_a, color_light_b=color_light_b, debug=True, 
                                    need_change_light=need_change_light, base_num=base_num)
    
    need_recolor_and_rotate = False
    if need_recolor_and_rotate:
        for transparency, theta in product(alpha_list, theta_list):
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            camera_id = env.sim.model.camera_name2id(camera_name)
            env = recolor_and_rotate_scene(env, alpha=transparency, color_light_a=color_light_a, color_light_b=color_light_b, 
                                     camera_id=camera_id, camera_name=camera_name, robot_base_name=robot_base_name, 
                                     theta=theta, debug=True)
    
    # 将场景中的部分无色设置为透明，不可见
    need_change_object_transparency = False
    if need_change_object_transparency:
        for transparency in transparency_list:
            env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
            env.reset()
            env = change_object_transparency(env, object_name=object_name, alpha=transparency, debug=True)
    

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