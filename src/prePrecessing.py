import numpy as np
from utils import *



# 数据预处理
# raw_path = "/data1/ryi/paddle_SN2N/raw_data"
# save_path = "/data1/ryi/paddle_SN2N/datasets"
raw_path = "/home/tyche/paddle_SN2N/raw_data"
save_path = "/home/tyche/paddle_SN2N/datasets"

raw_map_list, n_maps = get_all_files(raw_path)
print(raw_map_list)


n_chunks, i = 0, 0
for i, mapfile in enumerate(raw_map_list):
    n_chunks += split_and_save_tensor(mapfile, save_path, map_index=i)
    print(f'processing: {i}/{n_maps}')