import os
from glob import glob
import numpy as np
import pandas as pd


def get_success_rate_in_txt_file(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    if 'Total time taken: ' not in lines[-1]:
        return None
    else:
        # Current total success rate: 0.76
        return float(lines[-2].strip().split(': ')[-1])


def get_single_log_dir_success_rate(
    log_dir=r'/mnt/hdd3/xingyouguang/projects/robotics/openvla-mini/experiments/logs-0.25-0.25-0.75-0.75',
    need_A=True,
    need_B=True,
    need_C=True,
    need_D=True,
):
    txt_paths = sorted(glob(os.path.join(log_dir, '*.txt')))
    
    ckpt_ids = sorted(set([txt_path.split('/')[-1].split('_')[4] for txt_path in txt_paths]))
    success_rates = {}
    for ckpt_id in ckpt_ids:
        success_rates[ckpt_id] = {}
        ckpt_txt_paths = sorted(glob(os.path.join(log_dir, f'*{ckpt_id}*.txt')))
        try:
            assert len(ckpt_txt_paths) == (need_A + need_B + need_C + need_D)
            if need_A:
                assert f"{ckpt_id}_AEVAL" in ckpt_txt_paths[0]
            if need_B:
                assert f"{ckpt_id}_BEVAL" in ckpt_txt_paths[1]
            if need_C:
                assert f"{ckpt_id}_CEVAL" in ckpt_txt_paths[2]
            if need_D:
                assert f"{ckpt_id}_DEVAL" in ckpt_txt_paths[3]
        except Exception as e:
            print(f"Error in {ckpt_id}: {e}")
            import ipdb; ipdb.set_trace()
            print('this is a test')
        
        if need_A:
            success_rates[ckpt_id]['A'] = get_success_rate_in_txt_file(ckpt_txt_paths[0])
        if need_B:
            success_rates[ckpt_id]['B'] = get_success_rate_in_txt_file(ckpt_txt_paths[1])
        if need_C:
            success_rates[ckpt_id]['C'] = get_success_rate_in_txt_file(ckpt_txt_paths[2])
        if need_D:
            success_rates[ckpt_id]['D'] = get_success_rate_in_txt_file(ckpt_txt_paths[3])

    return success_rates


def main():
    log_dir_list = [
        r'/mnt/hdd3/xingyouguang/projects/robotics/openvla-mini/experiments/logs-0.25-0.25-0.25-0.25',
    ]
    
    df_dict = {}
    for log_dir in log_dir_list:
        success_rates = get_single_log_dir_success_rate(log_dir, need_A=True, need_B=False, need_C=False, need_D=False)
        df_dict[log_dir.split('/')[-1]] = pd.DataFrame(success_rates)
    
    for k, v in df_dict.items():
        print('#' * 5, k, '#' * 5)
        print(v)
        print('\n')
    
    import ipdb; ipdb.set_trace()
    print('this is a test')


if __name__ == "__main__":
    main()
