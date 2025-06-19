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
# import torch.fft as fft
from src.utils.utils_ddp import *


def get_all_files(directory):
    file_list = []
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return sorted(file_list)


def normalize(map_data, minPercent=0, maxPercent=99.999, mode='3d'):
    map_data = np.array(map_data)
    
    if mode.lower() == '3d':
        min_val = np.percentile(map_data, minPercent)
        max_val = np.percentile(map_data, maxPercent)
        map_data = np.clip(map_data, min_val, max_val)
        normalized_data = (map_data - min_val) / (max_val - min_val)
    elif mode.lower() == '2d':
        if map_data.ndim == 2:
            map_data = map_data.reshape(-1, map_data.shape[0], map_data.shape[1])
        normalized_data = np.zeros_like(map_data, dtype=np.float32)
        for z in range(map_data.shape[0]):
            slice_data = map_data[z]
            min_val = np.percentile(slice_data, minPercent)
            max_val = np.percentile(slice_data, maxPercent)
            slice_data = np.clip(slice_data, min_val, max_val)
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
    chunk_coords_generator = product(z_coords, x_coords, y_coords)
    return chunk_coords_generator


def process_chunk(padded_map, datasetsFolder, chunk_coords, box_size, map_index):
    cur_x, cur_y, cur_z = chunk_coords
    next_chunk = padded_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size].numpy()
    filepath = os.path.join(datasetsFolder, f'{map_index}_{cur_x}_{cur_y}_{cur_z}.npz')
    np.savez_compressed(filepath, next_chunk)
    return filepath, next_chunk.shape


def split_and_save_tensor(map_file, datasetsFolder, map_index, minPercent=0, maxPercent=99.999, box_size=48, stride=12):
    if not os.path.exists(datasetsFolder):
        os.makedirs(datasetsFolder)
    
    map_path = Path(map_file)
    try:
        if map_path.suffix == '.mrc':
            print(f"Loading mrc file: {map_file}")
            with mrcfile.open(map_file, mode='r') as mrc:
                map_data = np.asarray(mrc.data.copy(), dtype=np.float32)
        elif map_path.suffix == '.tif':
            print(f"Loading tif file: {map_file}")
            map_data = np.array(tifffile.imread(map_file))
        else:
            print.warning(f"Unsupported filetype: {map_path.suffix}, only .mrc and .tif are supported.")
    except Exception as e:
        print(f"Error loading file {map_file}: {str(e)}")
        map_data = None
        return map_data

    map_data = normalize(map_data, minPercent=minPercent, maxPercent=maxPercent, mode='3d')
    map_shape = map_data.shape
    padded_map = np.full((map_shape[0] + 2 * box_size, map_shape[1] + 2 * box_size, map_shape[2] + 2 * box_size), 0.0, dtype=np.float32)
    padded_map[box_size : box_size + map_shape[0], box_size : box_size + map_shape[1], box_size : box_size + map_shape[2]] = map_data
    del map_data

    padded_map = torch.from_numpy(padded_map)
    padded_map.share_memory_()
    chunk_coords_generator = generate_chunk_coords(map_shape=map_shape, box_size=box_size, stride=stride)
    num_workers = multiprocessing.cpu_count()
    process_func = partial(process_chunk, padded_map=padded_map, datasetsFolder=datasetsFolder, box_size=box_size, map_index=map_index)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, chunk_coords=coords) for coords in chunk_coords_generator]
        for future in as_completed(futures):
            try:
                filepath, chunk_shape = future.result()
                print(f"filename: {filepath}, chunk_shape: {chunk_shape}")
            except Exception as e:
                print(f"Error processing chunk: {e}")
    return map_shape


# def fourier_interpolate(tensor, scale_factor=2.0):
#     dim = tuple(range(2, tensor.ndim))
#     tensor_fft = fft.rfftn(tensor, dim=dim)
#     tensor_shape = tensor.shape
#     tensor_shape[2:] *= scale_factor
    
#     tensor_fft_padded = torch.zeros(
#         tuple(tensor_shape),
#         dtype=tensor_fft.dtype,
#         device=tensor_fft.device
#     )
#     starts = (tensor_shape[2:] - tensor.shape[2:]) // 2
    
#     tensor_fft_padded[:, :, starts:starts+tensor.shape[2:]] = tensor_fft
#     tensor_interpolated = fft.irfftn(tensor_fft_padded, s=tuple(tensor_shape[2:]), dim=dim)
#     return tensor_interpolated

# interpolate first and resample
# can process both 2d and 3d tensors
def resample(batch_chunks, kernel):
    out_channels, *spatial_dims = kernel.shape
    kernel = kernel.view(out_channels, 1, *spatial_dims)
    batch_size, *chunks_shape = batch_chunks.shape
    batch_chunks = batch_chunks.view(batch_size, 1, *chunks_shape)
    
    if len(spatial_dims) == 3:
        conv_layer = nn.Conv3d(in_channels=1, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        conv_layer.weight.data = kernel
        batch_chunks = conv_layer(batch_chunks)
        batch_chunks = torch.nn.functional.interpolate(batch_chunks, scale_factor=2, mode='trilinear')
    elif len(spatial_dims) == 2:
        conv_layer = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        conv_layer.weight.data = kernel
        batch_chunks = conv_layer(batch_chunks)
        batch_chunks = torch.nn.functional.interpolate(batch_chunks, scale_factor=2, mode='bilinear')
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