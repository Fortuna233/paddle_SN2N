import os
import mrcfile
import tifffile
import numpy as np
import multiprocessing
from pathlib import Path
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import imageio
import matplotlib.pyplot as plt
from typing import Dict, Optional

import torch
from torch import nn
import torch.fft as fft

from utils_ddp import *


def get_all_files(directory):
    file_list = []
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return sorted(file_list)


def process_chunk(padded_map, save_dir, chunk_coords, box_size, map_index):
    cur_x, cur_y, cur_z = chunk_coords
    next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size].numpy()
    filepath = os.path.join(save_dir, f'{map_index}_{cur_x}_{cur_y}_{cur_z}.npz')
    np.savez_compressed(filepath, next_chunk)
    return filepath, next_chunk.shape


def normalize(map_data, minPercent=0, maxPercent=99.999, mode='3d'):
    map_data = np.array(map_data)
    
    if mode.lower() == '3d':
        min_val = np.percentile(map_data, minPercent)
        max_val = np.percentile(map_data, maxPercent)
        normalized_data = (map_data - min_val) / (max_val - min_val)
    elif mode.lower() == '2d':
        normalized_data = np.zeros_like(map_data, dtype=np.float32)
        for z in range(map_data.shape[0]):
            slice_data = map_data[z]
            min_val = np.percentile(slice_data, minPercent)
            max_val = np.percentile(slice_data, maxPercent)
            normalized_data[z] = (slice_data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported mode: {mode}, only 2d or 3d mode supported")
    normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=1.0, neginf=0.0)
    return normalized_data


def generate_chunk_coords(map_shape, box_size, stride):
    start_point = box_size - stride
    z_coords = np.arange(start_point, map_shape[0] + box_size, stride)
    x_coords = np.arange(start_point, map_shape[1] + box_size, stride)
    y_coords = np.arange(start_point, map_shape[2] + box_size, stride)
    chunk_coords_list = list(product(z_coords, x_coords, y_coords))
    return chunk_coords_list


def split_and_save_tensor(map_file, save_dir, map_index, minPercent=0, maxPercent=99.999, box_size=48, stride=12):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    map_path = Path(map_file)
    if map_path.suffix == '.mrc':
        print("mapFile:", map_file)
        mrc = mrcfile.open(map_file, mode='r')
        map_data = np.asarray(mrc.data.copy(), dtype=np.float32)
        mrc.close()
    elif map_path.suffix == '.tif':
        print("mapFile:", map_file)
        map_data = np.array(tifffile.imread(map_file))
    else:
        print(f"unsupported filetype: {map_file.suffix}, only mrcfile and tifffile supported.")

    map_data = normalize(map_data, minPercent=minPercent, maxPercent=maxPercent, mode='3d')
    map_shape = map_data.shape
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), 0.0, dtype=np.float32)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map_data
    padded_map = torch.from_numpy(padded_map)
    padded_map.share_memory_()
    chunk_coords_list = generate_chunk_coords(map_shape=map_shape, box_size=box_size, stride=stride)
    num_workers = multiprocessing.cpu_count()
    process_func = partial(process_chunk, padded_map, map_file, save_dir, box_size=box_size, map_index=map_index)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, coords) for coords in chunk_coords_list]
        for future in as_completed(futures):
            try:
                filepath, chunk_shape = future.result()
                print(f"filename: {filepath}, chunk_shape: {chunk_shape}")
            except Exception as e:
                print(f"Error processing chunk: {e}")
    print(f"Total chunks processed: {len(chunk_coords_list)}")
    return len(chunk_coords_list), map_shape


def fourier_interpolate(tensor, scale_factor=2):
    assert tensor.ndim == 5, "[B, C, D, H, W]"
    
    # 3D FFT
    tensor_fft = fft.rfftn(tensor, dim=(-3, -2, -1))
    batch_size, channels, d, h, w = tensor.shape
    
    new_d = d * scale_factor
    new_h = h * scale_factor
    new_w = w * scale_factor
    
    tensor_fft_padded = torch.zeros(
        (batch_size, channels, new_d, new_h, new_w),
        dtype=tensor_fft.dtype,
        device=tensor_fft.device
    )
    d_start = (new_d - d) // 2
    h_start = (new_h - h) // 2
    w_start = (new_w - w) // 2 
    
    tensor_fft_padded[
        :, :, 
        d_start:d_start+d, 
        h_start:h_start+h, 
        w_start:w_start+w] = tensor_fft
    tensor_interpolated = fft.irfftn(tensor_fft_padded, s=(new_d, new_h, new_w), dim=(-3, -2, -1))
    tensor_interpolated = tensor_interpolated * (scale_factor ** 3)
    return tensor_interpolated


def resample(batch_chunks, kernel):
    kernel = kernel.view(-1, 1, 2, 2, 2)
    chunks_shape = batch_chunks.shape
    batch_chunks = batch_chunks.view(chunks_shape[0], 1, chunks_shape[1], chunks_shape[2], chunks_shape[3])
    batch_chunks = fourier_interpolate(batch_chunks, scale_factor=2)
    # return F.interpolate(batch_chunks, scale_factor=2, mode='trilinear')
    conv_layer = nn.Conv3d(in_channels=1, out_channels=kernel.shape[0], kernel_size=2, stride=2, padding=0, bias=False)
    conv_layer.weight.data = kernel
    batch_chunks = conv_layer(batch_chunks)
    return batch_chunks


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
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        # Close figure to free memory
        plt.close(fig)
    
    # Save frames as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")