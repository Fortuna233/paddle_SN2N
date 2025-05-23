import numpy as np
from utils import *



# 数据预处理
# raw_path = "data1/ryi/paddle_SN2N/raw_data"
# save_path = "data1/ryi/paddle_SN2N/datasets"
# paramsFolder = "data1/ryi/paddle_SN2N/params"
raw_path = "/home/tyche/paddle_SN2N/raw_data"
save_path = "/home/tyche/paddle_SN2N/datasets"
paramsFolder = "/home/tyche/paddle_SN2N/params"

_, current_epochs = get_all_files(paramsFolder)
raw_map_list, n_maps = get_all_files(raw_path)
print(raw_map_list)


n_chunks, i = 0, 0
for mapfile in raw_map_list:
    n_chunks += split_and_save_tensor(mapfile, save_path, file_type='.tif')
    i += 1 
    print(f'processing: {i}/{n_maps}')