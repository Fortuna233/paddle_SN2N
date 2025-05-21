import os
import time
import random
import torch
import mrcfile
import numpy as np
from torch import nn
from math import ceil
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader


def get_all_files(directory):
    file_list = []
    n_files = 0
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
        n_files += 1
    return file_list.sort(), n_files


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def split_and_save_tensor(map_file, save_dir, minPercent=0, maxPercent=99.999, box_size=60, stride=30):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("mapFile:", map_file)
    mrc = mrcfile.open(map_file, mode='r')
    map = np.asarray(mrc.data.copy(), dtype=np.float32)
    mrc.close()
    min = np.percentile(map, minPercent)
    max = np.percentile(map, maxPercent)
    map = map.clip(min=min, max=max) / max
    map_shape = map.shape
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), 0.0, dtype=np.float32)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
    padded_map = torch.from_numpy(padded_map)
    n_chunks = 0
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    while (cur_z + stride < map_shape[2] + box_size):
        next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        print(f'{map_file}_{cur_x}_{cur_y}_{cur_z}.npz')
        filename = os.path.join(save_dir, f'{map_file}_{cur_x}_{cur_y}_{cur_z}')      
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point # Reset X
            if (cur_y + stride  >= map_shape[1] + box_size):
                cur_z += stride
                cur_y = start_point # Reset Y
                cur_x = start_point # Reset X
        np.savez_compressed(filename, next_chunk)
        print(f"successfully save {filename}.npz")
        n_chunks += 1
    print(n_chunks)
    return n_chunks


class CompressedDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    

    def __len__(self):
        return len(self.file_list)
    

    def __getitem__(self, index):
        return torch.from_numpy(np.load(self.file_list[index])['arr_0'])


def diagonal_resample(batch_chunks, kernel):
    conv_layer = nn.Conv2d(in_channels=1, out_channels=kernel.shape[0], kernel_size=2, stride=2, padding=0, bias=False)

    conv_layer.weight.data = kernel
    batch_chunks = batch_chunks.unsqueeze(1)
    batch_chunks = conv_layer(batch_chunks)
    return batch_chunks


def get_augs(boxsize):
    train_augs = T.Compose([T.Resize(size=boxsize), T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
    test_augs = T.Compose([T.Resize(boxsize)])
    return train_augs, test_augs


 


    