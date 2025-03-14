import os
import sys
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from PIL import Image
import matplotlib.pyplot as plt
import wandb


@dataclass
class Config:
    task_suite_name: str = 'libero_90'


@draccus.wrap()
def main(cfg: Config) -> None:
    task_suite_name = cfg.task_suite_name
    
    base_save_img_dir = f'datasets_vis/{task_suite_name}'
    os.makedirs(base_save_img_dir, exist_ok=True)
    
    # 遍历base_save_img_dir中的所有文件夹
    image_list = []
    task_description_list = []
    for task_id in os.listdir(base_save_img_dir):
        task_id = int(task_id)
        task_img_dir = f'{base_save_img_dir}/{task_id}'
        
        task_description_path = f'{task_img_dir}/task_description.txt'
        with open(task_description_path, 'r') as f:
            task_description = f.read()
        task_description_list.append(task_description)
        
        # 遍历task_img_dir中的所有图片
        img_path_list = glob.glob(f'{task_img_dir}/*.png')
        tmp_image_list = []
        for img_path in img_path_list:
            img = Image.open(img_path)
            tmp_image_list.append(img)
        image_list.append(tmp_image_list)
        
    # len(image_list) x 2大小的figure
    fig, axs = plt.subplots(len(image_list), 2, figsize=(20, 40))
    for i in range(len(image_list)):
        axs[i, 0].imshow(image_list[i][0])
        axs[i, 1].imshow(image_list[i][1])
        # 读取task_img_dir中的task_description.txt
        axs[i, 0].set_title(f'Task: {task_description_list[i]}')
        # 关闭坐标轴
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
    
    fig.savefig(f'show_{task_suite_name}.png')
        
        
if __name__ == "__main__":
    main()


