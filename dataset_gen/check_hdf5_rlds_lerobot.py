# 加载 hdf5, rlds, lerobot 文件
import os
import h5py
from glob import glob
import cv2
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
)

def main():
    step_num=20
    
    repo_name = "xyg_15.0_65.0_test/v-0.250-0.450_num1"
    # hdf5    
    hdf5_data_path = r"/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island_1_hdf5"
    hdf5_data_file = h5py.File(glob(os.path.join(hdf5_data_path, repo_name, "*.hdf5"))[0], "r")
    hdf5_data = hdf5_data_file["data"]
    hdf5_image1 = hdf5_data["demo_0_1"]["obs"]["agentview_rgb"][()][step_num][::-1]
    hdf5_image2 = hdf5_data["demo_0_1"]["obs"]["eye_in_hand_rgb"][()][step_num][::-1]
    hdf5_action = hdf5_data["demo_0_1"]["actions"][()][step_num]
    hdf5_state = hdf5_data["demo_0_1"]["states"][()][step_num]
    Image.fromarray(hdf5_image1).save('tmp_dir/hdf5_image1.png')
    Image.fromarray(hdf5_image2).save('tmp_dir/hdf5_image2.png')
    
    # rlds
    rlds_data_path = r"/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island_1_rlds"
    rlds_builder = tfds.builder("libero_spatial", data_dir=os.path.join(rlds_data_path, repo_name))
    rlds_dataset = dl.DLataset.from_rlds(rlds_builder, split="train", shuffle=False)
    rlds_trajectory = next(rlds_dataset.iterator())
    rlds_image1 = cv2.imdecode(np.frombuffer(rlds_trajectory["observation"]["image"][step_num], dtype=np.uint8), cv2.IMREAD_COLOR_RGB)
    rlds_image2 = cv2.imdecode(np.frombuffer(rlds_trajectory["observation"]["wrist_image"][step_num], dtype=np.uint8), cv2.IMREAD_COLOR_RGB)
    rlds_state = rlds_trajectory["observation"]["state"][step_num]
    rlds_action = rlds_trajectory["action"][step_num]
    Image.fromarray(rlds_image1).save('tmp_dir/rlds_image1.png')
    Image.fromarray(rlds_image2).save('tmp_dir/rlds_image2.png')
    
    # lerobot
    lerobot_data_path = r"/mnt/hdd3/xingyouguang/datasets/robotics/libero/libero_spatial_no_noops_island_1_lerobot"
    lerobot_dataset = LeRobotDataset(
        repo_name,
        root=os.path.join(lerobot_data_path, repo_name),
    )
    lerobot_dataset.episode_data_index
    lerobot_image1 = (lerobot_dataset[step_num]["observation.image"].cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)    # c, h, w -> h, w, c
    lerobot_image2 = (lerobot_dataset[step_num]["observation.wrist_image"].cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
    lerobot_state = lerobot_dataset[step_num]["observation.state"]
    lerobot_action = lerobot_dataset[step_num]["action"]
    Image.fromarray(lerobot_image1).save('tmp_dir/lerobot_image1.png')
    Image.fromarray(lerobot_image2).save('tmp_dir/lerobot_image2.png')
    
    import ipdb; ipdb.set_trace()
    print('this is a test')

if __name__ == "__main__":
    main()
