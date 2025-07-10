import os
from src.utils.utils_dataprocessing import get_all_files, split_and_save_tensor
from src.SN2N_2D.constants_2d import rawdataFolder, datasetsFolder, box_size, stride


raw_map_list = get_all_files(rawdataFolder)
print(raw_map_list)


for i, mapfile in enumerate(raw_map_list):
    result = split_and_save_tensor(mapfile, datasetsFolder, map_index=i, minPercent=0, maxPercent=99.999, box_size=box_size, stride=stride, mode='2d')

    print(f'processing: {i + 1}/{len(raw_map_list)}')

with os.scandir(datasetsFolder) as entries:
    count = sum(1 for _ in entries)
print(f"total number of chunks: {count}")

