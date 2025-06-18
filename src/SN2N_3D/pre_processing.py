import torch
from src.utils.utils_dataprocessing import get_all_files, split_and_save_tensor



# 数据预处理
# raw_path = "/data1/ryi/paddle_SN2N/raw_data"
# save_path = "/data1/ryi/paddle_SN2N/datasets"
raw_path = "/home/tyche/paddle_SN2N/data/raw_data"
save_path = "/home/tyche/paddle_SN2N/data/datasets"

raw_map_list = get_all_files(raw_path)
print(raw_map_list)


n_chunks = 0
for i, mapfile in enumerate(raw_map_list):
    result = split_and_save_tensor(mapfile, save_path, map_index=i, minPercent=0, maxPercent=99.999, box_size=48, stride=12)
    n_chunks += result[0]
    print(f'processing: {i + 1}/{len(raw_map_list)}')

print(f"num_chunks: {n_chunks}")