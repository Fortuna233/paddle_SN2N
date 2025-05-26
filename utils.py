import os
import torch
import imageio
import mrcfile
import tifffile
import numpy as np
from torch import nn
import torch.fft as fft
import multiprocessing
from functools import partial
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
from typing import Dict, Optional
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed




def get_all_files(directory):
    file_list = []
    n_files = 0
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
        n_files += 1
    return sorted(file_list), n_files


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# def split_and_save_tensor(map_file, save_dir, minPercent=0, maxPercent=99.999, box_size=48, stride=12, file_type='.mrc'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     if file_type == '.mrc':
#         print("mapFile:", map_file)
#         mrc = mrcfile.open(map_file, mode='r')
#         map = np.asarray(mrc.data.copy(), dtype=np.float32)
#         mrc.close()
#     if file_type == '.tif':
#         print("mapFile:", map_file)
#         map = np.array(tifffile.imread(map_file))
    
#     min = np.percentile(map, minPercent)
#     max = np.percentile(map, maxPercent)
#     map = map.clip(min=min, max=max) / max
#     map_shape = map.shape
#     padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), 0.0, dtype=np.float32)
#     padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map
#     padded_map = torch.from_numpy(padded_map)
#     n_chunks = 0
#     start_point = box_size - stride
#     cur_x, cur_y, cur_z = start_point, start_point, start_point
#     while (cur_z + stride < map_shape[2] + box_size):
#         next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
#         filepath = os.path.join(save_dir, f'{map_file[-21:-4]}_{cur_x}_{cur_y}_{cur_z}.npz')
#         print(f"filename: {filepath}, chunk_shape: {next_chunk.shape}")  
#         cur_x += stride
#         if (cur_x + stride >= map_shape[0] + box_size):
#             cur_y += stride
#             cur_x = start_point # Reset X
#             if (cur_y + stride  >= map_shape[1] + box_size):
#                 cur_z += stride
#                 cur_y = start_point # Reset Y
#                 cur_x = start_point # Reset X
#         np.savez_compressed(filepath, next_chunk)
#         n_chunks += 1
#     print(n_chunks)
#     return n_chunks


def process_chunk(padded_map, map_file, save_dir, chunk_coords, box_size):
    """处理单个数据块并保存"""
    cur_x, cur_y, cur_z = chunk_coords
    next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size].numpy()
    filepath = os.path.join(save_dir, f'{map_file[-21:-4]}_{cur_x}_{cur_y}_{cur_z}.npz')
    np.savez_compressed(filepath, next_chunk)
    return filepath, next_chunk.shape


def split_and_save_tensor(map_file, save_dir, minPercent=0, maxPercent=99.999, box_size=48, stride=12, file_type='.mrc', num_workers=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取数据
    if file_type == '.mrc':
        print("mapFile:", map_file)
        mrc = mrcfile.open(map_file, mode='r')
        map_data = np.asarray(mrc.data.copy(), dtype=np.float32)
        mrc.close()
    elif file_type == '.tif':
        print("mapFile:", map_file)
        map_data = np.array(tifffile.imread(map_file))
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # 归一化和填充
    min_val = np.percentile(map_data, minPercent)
    max_val = np.percentile(map_data, maxPercent)
    map_data = map_data.clip(min=min_val, max=max_val) / max_val
    map_shape = map_data.shape
    
    # 创建填充后的张量
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), 0.0, dtype=np.float32)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map_data
    
    # 转换为PyTorch张量并设置为共享内存，避免每个进程复制
    padded_map = torch.from_numpy(padded_map)
    padded_map.share_memory_()
    
    # 生成所有块的坐标
    start_point = box_size - stride
    cur_x, cur_y, cur_z = start_point, start_point, start_point
    chunk_coords_list = []
    
    while (cur_z + stride < map_shape[2] + box_size):
        chunk_coords_list.append((cur_x, cur_y, cur_z))
        
        cur_x += stride
        if (cur_x + stride >= map_shape[0] + box_size):
            cur_y += stride
            cur_x = start_point  # Reset X
            if (cur_y + stride >= map_shape[1] + box_size):
                cur_z += stride
                cur_y = start_point  # Reset Y
                cur_x = start_point  # Reset X
    
    # 设置并行工作进程数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # 使用进程池并行处理
    process_func = partial(process_chunk, padded_map, map_file, save_dir, box_size=box_size)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, coords) for coords in chunk_coords_list]
        
        # 收集结果
        for future in as_completed(futures):
            try:
                filepath, chunk_shape = future.result()
                print(f"filename: {filepath}, chunk_shape: {chunk_shape}")
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    print(f"Total chunks processed: {len(chunk_coords_list)}")
    return len(chunk_coords_list)


class myDataset(Dataset):
    def __init__(self, file_list, is_train):
        self.file_list = file_list
        self.is_train = is_train


    def __len__(self):
        return len(self.file_list)
    

    def __getitem__(self, index):
        if self.is_train:
            return train_augs(torch.from_numpy(np.load(self.file_list[index])['arr_0']))
        return torch.from_numpy(np.load(self.file_list[index])['arr_0'])


def fourier_interpolate(tensor, scale_factor=2):
    assert tensor.ndim == 5, "[B, C, D, H, W]"
    
    # 3D FFT
    tensor_fft = fft.rfftn(tensor, dim=(-3, -2, -1))
    batch_size, channels, d, h, w = tensor.shape
    
    new_d = d * scale_factor
    new_h = h * scale_factor
    new_w = w * scale_factor
    new_w_rfft = new_w // 2 + 1  # rfft输出的最后一维大小
    
    # 创建零张量用于填充
    tensor_fft_padded = torch.zeros(
        (batch_size, channels, new_d, new_h, new_w_rfft),
        dtype=tensor_fft.dtype,
        device=tensor_fft.device
    )
    d_start = (new_d - d) // 2
    h_start = (new_h - h) // 2
    w_start = 0  # rfft的低频部分从0开始
    
    tensor_fft_padded[
        :, :, 
        d_start:d_start+d, 
        h_start:h_start+h, 
        w_start:w_start+(w//2+1)
    ] = tensor_fft
    
    tensor_interpolated = fft.irfftn(tensor_fft_padded, s=(new_d, new_h, new_w), dim=(-3, -2, -1))
    
    # tensor_interpolated = tensor_interpolated * (scale_factor ** 3)
    return tensor_interpolated


def resample(batch_chunks, kernel):
    kernel = kernel.view(kernel.shape[0], 1, 2, 2, 2)
    chunks_shape = batch_chunks.shape
    batch_chunks = batch_chunks.view(chunks_shape[0], 1, chunks_shape[1], chunks_shape[2], chunks_shape[3])
    conv_layer = nn.Conv3d(in_channels=1, out_channels=kernel.shape[0], kernel_size=2, stride=2, padding=0, bias=False)
    conv_layer.weight.data = kernel
    batch_chunks = conv_layer(batch_chunks)
    return fourier_interpolate(batch_chunks)


def train_augs(tensor):
    augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
    return augs(tensor) 


def combine_tensors_to_gif(
    tensors: Dict[str, np.ndarray],
    output_path: str = "combined_tensors.gif",
    fps: int = 2,
    cmap: str = "viridis",
    figsize: tuple = (12, 8),
    dpi: int = 100,
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:

    # Check all tensors have the same depth
    depths = [tensor.shape[0] for tensor in tensors.values()]   
    depth = max(depths)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate frames for GIF
    frames = []
    
    for i in range(depth):
        # Create figure and axes
        fig, axes = plt.subplots(1, len(tensors), figsize=figsize, dpi=dpi, sharey=True)
        if len(tensors) == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Plot each tensor's current layer
        ims = []
        for j, (name, tensor) in enumerate(tensors.items()):
            im = axes[j].imshow(tensor[i % tensor.shape[0]], cmap=cmap, vmin=vmin, vmax=vmax)
            ims.append(im)
            axes[j].set_title(f"{name} - Layer {i % tensor.shape[0]}")
        
        # Add colorbar if specified
        if show_colorbar:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(ims[0], cax=cbar_ax)
        
        # Render figure to numpy array
        plt.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        # Close figure to free memory
        plt.close(fig)
    
    # Save frames as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")


# 定义损失函数
def loss(pred, target):
    smooth_l1_loss = nn.SmoothL1Loss(beta=1.0, reduction='mean')
    return smooth_l1_loss(pred, target) + 1 - ssim(pred, target, data_range=1.0, size_average=True)


def loss_channels(X, Y, loss_fun=loss):
    l = 0
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i > j:
                l += loss_fun(X[:, i, :, :, :], Y[:, j, :, : ,:]) 
                l += loss_fun(X[:, j, :, :, :], Y[:, i, :, : ,:])
                l += loss_fun(Y[:, i, :, :, :], Y[:, j, :, : ,:])
    return l