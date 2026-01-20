#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import h5py
import numpy as np
import argparse
from pathlib import Path
import cv2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ================= 설정 (Configuration) =================
FPS = 10
IMG_H, IMG_W = 256, 256
# ========================================================


def is_noop(action, prev_action=None, threshold=1e-4):
    """action: (7,) = [x,y,z,rx,ry,rz,gripper]"""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def preprocess_image(img, out_hw=(IMG_H, IMG_W)):
    """
    - (H,W,3) 형태로 맞추고
    - 256x256으로 리사이즈
    - uint8 유지
    """
    if img is None:
        raise ValueError("Image is None")

    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {img.shape}")

    if (img.shape[0], img.shape[1]) != out_hw:
        img = cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)

    if img.dtype != np.uint8:
        # 데이터가 float [0,1]인 경우 등을 대비
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    return img


import re
import numpy as np
from pathlib import Path

def load_wrist_intrinsic_matrix(path: Path) -> np.ndarray:
    txt = path.read_text()

    def find_float(keys):
        # keys 중 하나를 포함한 줄에서 float 찾기
        for k in keys:
            m = re.search(rf"{k}\s*=\s*['\"]?([-+]?\d*\.?\d+)", txt)
            if m:
                return float(m.group(1))
            m = re.search(rf"<{k}>\s*([-+]?\d*\.?\d+)\s*</{k}>", txt)
            if m:
                return float(m.group(1))
        return None

    fx = find_float(["px", "fx", "focal_x"])
    fy = find_float(["py", "fy", "focal_y"])
    cx = find_float(["u0", "cx", "principal_x"])
    cy = find_float(["v0", "cy", "principal_y"])

    if None in (fx, fy, cx, cy):
        raise RuntimeError(f"Could not parse fx,fy,cx,cy from {path}")

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return K

def get_scaled_intrinsic_matrix(K: np.ndarray, src_wh=(640, 480), dst_wh=(256, 256)) -> np.ndarray:
    """
    원본 해상도(src_wh) 기준 intrinsic K를, 리사이즈된 해상도(dst_wh) 기준으로 스케일링.

    src_wh: (width, height)
    dst_wh: (width, height)

    스케일링:
      fx' = fx * (dst_w/src_w)
      fy' = fy * (dst_h/src_h)
      cx' = cx * (dst_w/src_w)
      cy' = cy * (dst_h/src_h)
    """
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh

    sw = float(dst_w) / float(src_w)
    sh = float(dst_h) / float(src_h)

    K_scaled = K.copy()
    K_scaled[0, 0] *= sw  # fx
    K_scaled[1, 1] *= sh  # fy
    K_scaled[0, 2] *= sw  # cx (u0)
    K_scaled[1, 2] *= sh  # cy (v0)

    # K[0,1], K[1,0] (skew 등)도 일반적으로는 스케일링이 필요할 수 있지만
    # 보통 0이라 그대로 두며, 필요 시 아래 주석 해제:
    # K_scaled[0, 1] *= sw
    # K_scaled[1, 0] *= sh

    return K_scaled.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Convert PKL files to LeRobot and HDF5 dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing .pkl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Root path for output")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot Repo ID (e.g., yujin/ur5_plush_pickup)")
    parser.add_argument(
        "--wrist_intrinsic_path",
        type=str,
        default="/home/vai/Desktop/yujin/visp/build/apps/calibration/hand-eye/data-ur/ur_camera.xml",
        help="Path to wrist camera intrinsic (XML/TXT). Loaded once and reused for all frames.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    repo_id = args.repo_id

    if not input_dir.exists():
        print(f"Error: 입력 폴더를 찾을 수 없습니다 -> {input_dir}")
        return

    pkl_files = sorted(list(input_dir.glob("*.pkl")))
    if not pkl_files:
        print(f"Error: {input_dir} 내에 .pkl 파일이 없습니다.")
        return

    print(f"Found {len(pkl_files)} pickle files.")

    task_description = "Pick_up_the_purple_grape_and_place_it_on_the_orange_plate"
    dataset_dir = output_root / repo_id

    wrist_extrinsic_path = Path("/home/vai/Desktop/yujin/visp/build/apps/calibration/hand-eye/data-ur/ur_ePc.txt")
    wrist_intrinsic_path = Path(args.wrist_intrinsic_path)
    if not wrist_intrinsic_path.exists():
        raise FileNotFoundError(f"wrist_intrinsic_path not found: {wrist_intrinsic_path}")
    last_wrist_extrinsic = np.loadtxt(wrist_extrinsic_path).astype(np.float32).reshape(4, 4)
    last_wrist_intrinsic = load_wrist_intrinsic_matrix(wrist_intrinsic_path).astype(np.float32)
    last_wrist_intrinsic = get_scaled_intrinsic_matrix(last_wrist_intrinsic, src_wh=(640, 480), dst_wh=(256, 256))
    
    # LeRobot Dataset 초기화
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_dir,
        fps=FPS,
        robot_type="ur5",
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (IMG_H, IMG_W, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.wrist_image": {  
                "dtype": "image",
                "shape": (IMG_H, IMG_W, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "observation.tcp_pose": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["tcp_pose"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
            "intrinsic_matrix": {
                "dtype": "float32",
                "shape": (3, 3),
            },
            "extrinsic_matrix": {
                "dtype": "float32",
                "shape": (4, 4),
            },
            "wrist_intrinsic_matrix": { 
                "dtype": "float32",
                "shape": (3, 3),
            },
            "wrist_extrinsic_matrix": { 
                "dtype": "float32",
                "shape": (4, 4),
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # HDF5 (통합 파일 하나 생성)
    safe_repo_name = repo_id.replace("/", "_")
    hdf5_path = dataset_dir / f"{safe_repo_name}_demos.hdf5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(hdf5_path, "w")
    grp = h5_file.create_group("data")

    print(f"Start processing... Output HDF5: {hdf5_path}")

    episode_count = 0

    for pkl_path in pkl_files:
        print(f"\nProcessing file: {pkl_path.name}...")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 에피소드 버퍼
        ep_buffer = {
            "agentview": [],
            "wristview": [],
            "actions": [],
            "robot_states": [],
            "extrinsics": [],
            "intrinsics": [],
            "wrist_extrinsics": [],
            "wrist_intrinsics": [],
        }

        last_extrinsic = np.eye(4, dtype=np.float32)
        last_intrinsic = np.eye(3, dtype=np.float32)

        prev_action = None

        for i, frame in enumerate(data):
            # --- Action 처리 ---
            # frame['actions']가 (7,)이라고 가정: [x,y,z,rx,ry,rz,gripper]
            raw_actions = np.asarray(frame["actions"], dtype=np.float32)
            if raw_actions.shape[0] < 7:
                raise ValueError(f"frame['actions'] expected >=7 dims, got {raw_actions.shape}")

            xyzrpy = raw_actions[:6]
            gripper_action = raw_actions[6]
            action = np.concatenate([xyzrpy, np.array([gripper_action], dtype=np.float32)]).astype(np.float32)  # (7,)

            reward = float(frame.get("rewards", 0.0))
            is_done = bool(frame.get("dones", False))

            # reward=1인데 done이 아닌 프레임에서 noop면 스킵 (기존 로직 유지)
            if reward == 1.0 and not is_done:
                if is_noop(action, prev_action):
                    continue

            prev_action = action

            # --- Observation ---
            img_front = preprocess_image(frame["observations"]["front"])
            img_wrist = preprocess_image(frame["observations"]["wrist"])

            tcp_pose = np.asarray(frame["observations"]["state"]["tcp_pose"], dtype=np.float32)

            tip_pose = np.asarray(frame["observations"]["state"]["tip_pose"], dtype=np.float32)
            pos = tip_pose[:3]
            quat = tip_pose[3:]
            gripper_scalar = float(frame["observations"]["state"]["gripper_state"])

            robot_state_8d = np.concatenate([pos, quat, np.array([gripper_scalar], dtype=np.float32)]).astype(np.float32)

            # --- Matrices (PKL에 있는 것만 갱신) ---
            if "extrinsic_matrix" in frame:
                last_extrinsic = np.asarray(frame["extrinsic_matrix"], dtype=np.float32)
            if "intrinsic_matrix" in frame:
                last_intrinsic = np.asarray(frame["intrinsic_matrix"], dtype=np.float32)
            # wrist intrinsic은 PKL에 없으므로 last_wrist_intrinsic(파일에서 읽은 상수) 사용

            # --- LeRobot frame 추가 ---
            dataset.add_frame(
                {
                    "observation.image": img_front,
                    "observation.wrist_image": img_wrist,  # ✅ 키 통일
                    "observation.state": robot_state_8d,
                    "observation.tcp_pose": tcp_pose,
                    "action": action,
                    "intrinsic_matrix": last_intrinsic,
                    "extrinsic_matrix": last_extrinsic,
                    "wrist_extrinsic_matrix": last_wrist_extrinsic,
                    "wrist_intrinsic_matrix": last_wrist_intrinsic,
                    "task": task_description,
                }
            )

            # --- HDF5 buffer ---
            ep_buffer["agentview"].append(img_front)
            ep_buffer["wristview"].append(img_wrist)
            ep_buffer["actions"].append(action)
            ep_buffer["robot_states"].append(robot_state_8d)
            ep_buffer["extrinsics"].append(last_extrinsic)
            ep_buffer["intrinsics"].append(last_intrinsic)
            ep_buffer["wrist_extrinsics"].append(last_wrist_extrinsic)
            ep_buffer["wrist_intrinsics"].append(last_wrist_intrinsic)

            # --- Episode end condition ---
            if is_done or reward == 1.0:
                num_frames = len(ep_buffer["actions"])

                # 10프레임 이하 skip
                if num_frames <= 10:
                    print(f"Skipping short episode (Length: {num_frames})")
                    for key in ep_buffer:
                        ep_buffer[key] = []
                    prev_action = None
                    continue

                # 1) LeRobot 에피소드 저장
                dataset.save_episode()

                # 2) HDF5 저장
                demo_grp_name = f"demo_{episode_count}"
                ep_data_grp = grp.create_group(demo_grp_name)

                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("agentview_rgb", data=np.stack(ep_buffer["agentview"], axis=0))
                obs_grp.create_dataset("wristview_rgb", data=np.stack(ep_buffer["wristview"], axis=0))

                obs_grp.create_dataset("intrinsic_matrix", data=np.stack(ep_buffer["intrinsics"], axis=0))
                obs_grp.create_dataset("extrinsic_matrix", data=np.stack(ep_buffer["extrinsics"], axis=0))
                obs_grp.create_dataset("wrist_intrinsic_matrix", data=np.stack(ep_buffer["wrist_intrinsics"], axis=0))
                obs_grp.create_dataset("wrist_extrinsic_matrix", data=np.stack(ep_buffer["wrist_extrinsics"], axis=0))

                ep_data_grp.create_dataset("actions", data=np.stack(ep_buffer["actions"], axis=0))
                ep_data_grp.create_dataset("robot_states", data=np.stack(ep_buffer["robot_states"], axis=0))

                dones_arr = np.zeros(num_frames, dtype=np.uint8)
                dones_arr[-1] = 1
                rewards_arr = np.zeros(num_frames, dtype=np.uint8)
                rewards_arr[-1] = 1
                ep_data_grp.create_dataset("dones", data=dones_arr)
                ep_data_grp.create_dataset("rewards", data=rewards_arr)

                ep_data_grp.attrs["model_file"] = "xml_model_string_placeholder"

                print(f"  -> Saved Episode {episode_count} (Frames: {num_frames})")

                episode_count += 1
                prev_action = None

                # 버퍼 초기화
                for key in ep_buffer:
                    ep_buffer[key] = []

    h5_file.close()

    print("\n=== 전체 변환 완료 ===")
    print(f"1. LeRobot Dataset Path: {dataset_dir}")
    print(f"2. HDF5 File Path: {hdf5_path}")
    print(f"Total Episodes Processed: {episode_count}")


if __name__ == "__main__":
    main()
