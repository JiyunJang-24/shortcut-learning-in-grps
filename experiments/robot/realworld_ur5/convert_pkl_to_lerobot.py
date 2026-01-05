import pickle
import h5py
import torch
import numpy as np
import argparse
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R
import cv2

# ================= 설정 (Configuration) =================
FPS = 10
# ========================================================

def is_noop(action, prev_action=None, threshold=1e-4):
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action

def main():
    parser = argparse.ArgumentParser(description="Convert PKL files to LeRobot and HDF5 dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing .pkl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Root path for output")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot Repo ID (e.g., yujin/ur5_plush_pickup)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    repo_id = args.repo_id

    if not input_dir.exists():
        print(f"Error: 입력 폴더를 찾을 수 없습니다 -> {input_dir}")
        return

    # 폴더 내 모든 pkl 파일 찾기 및 정렬
    pkl_files = sorted(list(input_dir.glob("*.pkl")))
    if not pkl_files:
        print(f"Error: {input_dir} 내에 .pkl 파일이 없습니다.")
        return

    print(f"Found {len(pkl_files)} pickle files.")

    task_description = "Pick up the gray plush from the bin and place it on the brown plate"
    dataset_dir = output_root / repo_id
    
    # LeRobot Dataset 초기화 (한 번만 생성)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_dir,
        fps=FPS,
        robot_type="ur5",
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,), 
                "names": ["state"],
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
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # HDF5 파일 생성 (통합 파일 하나 생성)
    safe_repo_name = repo_id.replace('/', '_')
    hdf5_path = data_dir / f"{safe_repo_name}_demos.hdf5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(hdf5_path, "w")
    grp = h5_file.create_group("data")

    print(f"Start processing... Output HDF5: {hdf5_path}")
    
    episode_count = 0  # 전체 파일에 걸쳐 에피소드 카운트 유지

    for pkl_path in pkl_files:
        print(f"\nProcessing file: {pkl_path.name}...")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 에피소드별 임시 저장소 (파일이 바뀔 때 초기화해도 되지만, 안전하게 에피소드 단위로 관리)
        ep_buffer = {
            "agentview": [],
            "actions": [],
            "robot_states": [],
            "extrinsics": [],
            "intrinsics": []
        }

        last_extrinsic = np.eye(4, dtype=np.float32)
        last_intrinsic = np.eye(3, dtype=np.float32)
        prev_action = None
        
        # 파일 내 프레임 순회
        for i, frame in enumerate(data):
            
            # 1. Action 처리 및 No-op 체크
            xyz = frame['actions'][:3]
            gripper_action = frame['actions'][3:]
            rpy_zeros = np.zeros(3, dtype=np.float32)
            action = np.concatenate([xyz, rpy_zeros, gripper_action]).astype(np.float32)  # (7,)

            reward = frame.get('rewards', 0.0)
            is_done = frame.get('dones', False)

            if reward == 1.0 and not is_done:
                if is_noop(action, prev_action):
                    continue

            prev_action = action 

            img_front = frame['observations']['front']
            
            tip_pose = frame['observations']['state']['tip_pose']
            pos = tip_pose[:3]
            quat = tip_pose[3:]
            gripper_scalar = frame['observations']['state']['gripper_state']
            
            robot_state_8d = np.concatenate([pos, quat, np.array([gripper_scalar])]).astype(np.float32)

            if 'extrinsic_matrix' in frame:
                last_extrinsic = frame['extrinsic_matrix'].astype(np.float32)
            if 'intrinsic_matrix' in frame:
                last_intrinsic = frame['intrinsic_matrix'].astype(np.float32)

            # LeRobot Dataset에 프레임 추가
            dataset.add_frame({
                "observation.image": img_front,
                "observation.state": robot_state_8d,
                "action": action,
                "intrinsic_matrix": last_intrinsic,
                "extrinsic_matrix": last_extrinsic,
                "task": task_description
            })

            # HDF5 버퍼에 추가
            ep_buffer["agentview"].append(img_front)
            ep_buffer["actions"].append(action)
            ep_buffer["robot_states"].append(robot_state_8d)
            ep_buffer["extrinsics"].append(last_extrinsic)
            ep_buffer["intrinsics"].append(last_intrinsic)

            # 에피소드 종료 조건
            if np.any(is_done) or reward == 1.0: 
                dataset.save_episode()

                demo_grp_name = f"demo_{episode_count}"
                ep_data_grp = grp.create_group(demo_grp_name)
                
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("agentview_rgb", data=np.stack(ep_buffer["agentview"], axis=0))
                obs_grp.create_dataset("intrinsic_matrix", data=np.stack(ep_buffer["intrinsics"], axis=0))
                obs_grp.create_dataset("extrinsic_matrix", data=np.stack(ep_buffer["extrinsics"], axis=0))
                
                ep_data_grp.create_dataset("actions", data=np.stack(ep_buffer["actions"], axis=0))
                ep_data_grp.create_dataset("robot_states", data=np.stack(ep_buffer["robot_states"], axis=0))
                
                num_frames = len(ep_buffer["actions"])
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

        # 파일 끝 처리: 만약 에피소드가 안 끝난 상태로 파일이 끝나면 저장 (Warning)
        if len(ep_buffer["actions"]) > 0:
            print(f"Warning: File {pkl_path.name} ended but episode not marked done. Saving anyway...")
            dataset.save_episode()
            
            demo_grp_name = f"demo_{episode_count}"
            ep_data_grp = grp.create_group(demo_grp_name)
            
            obs_grp = ep_data_grp.create_group("obs")
            obs_grp.create_dataset("agentview_rgb", data=np.stack(ep_buffer["agentview"], axis=0))
            obs_grp.create_dataset("intrinsic_matrix", data=np.stack(ep_buffer["intrinsics"], axis=0))
            obs_grp.create_dataset("extrinsic_matrix", data=np.stack(ep_buffer["extrinsics"], axis=0))

            ep_data_grp.create_dataset("actions", data=np.stack(ep_buffer["actions"], axis=0))
            ep_data_grp.create_dataset("robot_states", data=np.stack(ep_buffer["robot_states"], axis=0))
            
            # 강제로 done 처리
            num_frames = len(ep_buffer["actions"])
            dones_arr = np.zeros(num_frames, dtype=np.uint8)
            dones_arr[-1] = 1
            ep_data_grp.create_dataset("dones", data=dones_arr)

            episode_count += 1

    h5_file.close()
    
    print("\n=== 전체 변환 완료 ===")
    print(f"1. LeRobot Dataset Path: {dataset_dir}")
    print(f"2. HDF5 File Path: {hdf5_path}")
    print(f"Total Episodes Processed: {episode_count}")

if __name__ == "__main__":
    main()