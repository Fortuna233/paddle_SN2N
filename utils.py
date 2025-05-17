import os
import time
import random
import torch
import mrcfile
import numpy as np
from torch import nn
from math import ceil
import matplotlib.pyplot as plt
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


def fourier_inter(batch_chunks, scale_factor):
    """
    SN2N tool: Fourier re-scale
    ------
    image_stack
        image TO Fourier interpolation
    
    Returns
    -------
    imgf1: image with 2x size 
    """
    chunk_shape = batch_chunks.shape
    print(chunk_shape)

    [t, x, y] = image_stack.shape
    imgf1 = np.zeros((t, imsize[0], imsize[1]))
    
    for slice in range(t):
        img = image_stack[slice, :, :]
        imgsz = np.array([x, y])
        tem1 = np.divide(imgsz, 2)
        tem2 = np.multiply(tem1, 2)
        tem3 = np.subtract(imgsz, tem2)
        b = (tem3 == np.array([0, 0]))
        if b[0] == True:
            sz = imgsz - 1
        else:
            sz = imgsz            
        n = np.array([2, 2])
        ttem1 = np.add(np.ceil(np.divide(sz, 2)), 1)
        ttem2 = np.multiply(np.floor(np.divide(sz, 2)), np.subtract(n, 1))
        idx = np.add(ttem1, ttem2)
        padsize = np.array([x/2, y/2], dtype = 'int')
        pad_wid = np.ceil(padsize[0]).astype('int')
        img = np.pad(img, ((pad_wid, 0), (pad_wid, 0)), 'symmetric')
        img = np.pad(img, ((0, pad_wid), (0, pad_wid)),  'symmetric')
        imgsz1 = np.array(img.shape)
        tttem1 = np.multiply(n, imgsz1)
        tttem2 = np.subtract(n, 1)
        newsz = np.round(np.subtract(tttem1, tttem2))
        img1 = self.interpft(img, newsz[0], 0)
        img1 = self.interpft(img1, newsz[1], 1)
        idx = idx.astype('int')
        ttttem1 = np.subtract(np.multiply(n[0], imgsz[0]), 1).astype('int')
        ttttem2 = np.subtract(np.multiply(n[1], imgsz[1]), 1).astype('int')
        imgf1[slice, :, :] = img1[idx[0] - 1:idx[0] + ttttem1, idx[1] - 1:idx[1] + ttttem2]
        imgf1[imgf1 < 0] = 0
    return imgf1

def interpft(self, x, ny, dim = 0):
    '''
    Function to interpolate using FT method, based on matlab interpft()
    ------
    x 
        array for interpolation
    ny 
        length of returned vector post-interpolation
    dim
        performs interpolation along dimension DIM
        {default: 0}
    Returns
    -------
    y: interpolated data
    '''

    if dim >= 1: 
    #if interpolating along columns, dim = 1
        x = np.swapaxes(x,0,dim)
    #temporarily swap axes so calculations are universal regardless of dim
    if len(x.shape) == 1:            
    #interpolation should always happen along same axis ultimately
        x = np.expand_dims(x,axis=1)

    siz = x.shape
    [m, n] = x.shape

    a = np.fft.fft(x,m,0)
    nyqst = int(np.ceil((m+1)/2))
    b = np.concatenate((a[0:nyqst,:], np.zeros(shape=(ny-m,n)), a[nyqst:m, :]),0)

    if np.remainder(m,2)==0:
        b[nyqst,:] = b[nyqst,:]/2
        b[nyqst+ny-m,:] = b[nyqst,:]

    y = np.fft.irfft(b,b.shape[0],0)
    y = y * ny / m
    y = np.reshape(y, [y.shape[0],siz[1]])
    y = np.squeeze(y)

    if dim >= 1:  
    #switches dimensions back here to get desired form
        y = np.swapaxes(y,0,dim)

    return y





 


    