import os
import mrcfile
import tifffile
import numpy as np
import multiprocessing
from pathlib import Path
from functools import partial
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    if mode == '3d':
        min_val = np.percentile(map_data, minPercent)
        max_val = np.percentile(map_data, maxPercent)
        map_data = np.clip(map_data, min_val, max_val)
        normalized_data = (map_data - min_val) / (max_val - min_val)
    elif mode == '2d':
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


def generate_chunk_coords(map_shape, box_size, stride, mode='3d'):
    if mode == '3d':
        z_coords = np.arange(0, map_shape[0] - box_size, stride)
        x_coords = np.arange(stride, map_shape[1] - box_size, stride)
        y_coords = np.arange(stride, map_shape[2] - box_size, stride)
        chunk_coords_generator = product(z_coords, x_coords, y_coords)
    elif mode == '2d':
        z_coords = np.arange(0, map_shape[0], 1)
        x_coords = np.arange(0, map_shape[1] - box_size, stride)
        y_coords = np.arange(0, map_shape[2] - box_size, stride)
        chunk_coords_generator = product(z_coords, x_coords, y_coords)
    return chunk_coords_generator


def process_chunk(map_data, datasetsFolder, chunk_coords, box_size, map_index, mode='3d'):
    cur_z, cur_x, cur_y = chunk_coords
    if mode == '3d':
        next_chunk = map_data[cur_z:cur_z + box_size, cur_x:cur_x + box_size, cur_y:cur_y + box_size].numpy()
    elif mode == '2d':
        next_chunk = map_data[cur_z, cur_x:cur_x + box_size, cur_y:cur_y + box_size].numpy()
    filepath = os.path.join(datasetsFolder, f'{map_index}_{cur_z}_{cur_x}_{cur_y}.npz')
    np.savez_compressed(filepath, next_chunk)
    return filepath, next_chunk.shape


def split_and_save_tensor(map_file, datasetsFolder, map_index, minPercent=0, maxPercent=99.999, box_size=48, stride=12, mode='3d'):
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

    map_data = normalize(map_data, minPercent=minPercent, maxPercent=maxPercent, mode=mode)
    map_shape = map_data.shape
    print(f"map_shape: {map_shape}")
    if len(map_shape) == 2:
        map_data = map_data.reshape(-1, *map_shape)


    map_data = torch.from_numpy(map_data)
    map_data.share_memory_()
    chunk_coords_generator = generate_chunk_coords(map_shape=map_shape, box_size=box_size, stride=stride, mode=mode)
    num_workers = multiprocessing.cpu_count()
    process_func = partial(process_chunk, map_data=map_data, datasetsFolder=datasetsFolder, box_size=box_size, map_index=map_index, mode=mode)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_func, chunk_coords=coords) for coords in chunk_coords_generator]
        for future in as_completed(futures):
            try:
                filepath, chunk_shape = future.result()
                print(f"filename: {filepath}, chunk_shape: {chunk_shape}")
            except Exception as e:
                print(f"Error processing chunk: {e}")
    return map_shape


def resample(batch_chunks, kernel, interpolate_mode='fourier'):
    out_channels, *spatial_dims = kernel.shape
    kernel = kernel.view(out_channels, 1, *spatial_dims)
    batch_size, *chunks_shape = batch_chunks.shape
    batch_chunks = batch_chunks.view(batch_size, 1, *chunks_shape)
    if len(spatial_dims) == 3:
        conv_layer = nn.Conv3d(in_channels=1, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        conv_layer.weight.data = kernel
        batch_chunks = conv_layer(batch_chunks)
        if interpolate_mode == 'fourier':
            batch_chunks = fourier_interpolate(batch_chunks) * 16
        else:
            batch_chunks = torch.nn.functional.interpolate(batch_chunks, scale_factor=2, mode='trilinear')
    elif len(spatial_dims) == 2:
        conv_layer = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        conv_layer.weight.data = kernel
        batch_chunks = conv_layer(batch_chunks)
        if interpolate_mode == 'fourier':
            batch_chunks = fourier_interpolate(batch_chunks) * 16
        else:
            batch_chunks = torch.nn.functional.interpolate(batch_chunks, scale_factor=2, mode='bilinear')
    return batch_chunks


def fourier_interpolate(image: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Upsample image or volumetric data using Fourier interpolation
    
    Args:
        image: Input tensor, supports 4D (B, C, H, W) 2D images or 5D (B, C, D, H, W) 3D volumes
        factor: Upsampling factor, default is 2
    
    Returns:
        Upsampled tensor
    """
    # Check input dimensions
    if image.dim() not in [4, 5]:
        raise ValueError("Input tensor must be 4D (B, C, H, W) or 5D (B, C, D, H, W)")
    
    is_3d = (image.dim() == 5)
    batch_size, channels = image.shape[:2]
    
    # Calculate target dimensions
    if is_3d:
        depth, height, width = image.shape[2:]
        target_depth = depth * factor
        target_height = height * factor
        target_width = width * factor
    else:
        height, width = image.shape[2:]
        target_height = height * factor
        target_width = width * factor
    
    # Symmetric padding
    if is_3d:
        padding = (width//2, width//2, height//2, height//2, depth//2, depth//2)
    else:
        padding = (width//2, width//2, height//2, height//2)
    
    img_padded = torch.nn.functional.pad(image, padding, mode='reflect')
    
    # Perform Fourier transform
    if is_3d:
        img_fft = torch.fft.fftn(img_padded, dim=(-3, -2, -1))
    else:
        img_fft = torch.fft.fft2(img_padded, dim=(-2, -1))
    
    # Get original and target frequency domain shapes
    fft_shape = list(img_fft.shape[-3:]) if is_3d else list(img_fft.shape[-2:])
    new_fft_shape = [s * factor for s in fft_shape]
    
    # Create padded frequency domain tensor
    fft_padded_shape = list(img_fft.shape[:-len(fft_shape)]) + new_fft_shape
    fft_padded = torch.zeros(fft_padded_shape, dtype=img_fft.dtype, device=img_fft.device)
    
    # Calculate center indices
    if is_3d:
        d_half, h_half, w_half = fft_shape[0]//2, fft_shape[1]//2, fft_shape[2]//2
        new_d_half, new_h_half, new_w_half = new_fft_shape[0]//2, new_fft_shape[1]//2, new_fft_shape[2]//2
        
        # Place original frequency components in new frequency space
        # Quadrant 1: top-left-front
        fft_padded[..., :d_half, :h_half, :w_half] = img_fft[..., :d_half, :h_half, :w_half]
        # Quadrant 2: top-right-front
        fft_padded[..., :d_half, :h_half, -w_half:] = img_fft[..., :d_half, :h_half, -w_half:]
        # Quadrant 3: bottom-left-front
        fft_padded[..., :d_half, -h_half:, :w_half] = img_fft[..., :d_half, -h_half:, :w_half]
        # Quadrant 4: bottom-right-front
        fft_padded[..., :d_half, -h_half:, -w_half:] = img_fft[..., :d_half, -h_half:, -w_half:]
        # Quadrant 5: top-left-back
        fft_padded[..., -d_half:, :h_half, :w_half] = img_fft[..., -d_half:, :h_half, :w_half]
        # Quadrant 6: top-right-back
        fft_padded[..., -d_half:, :h_half, -w_half:] = img_fft[..., -d_half:, :h_half, -w_half:]
        # Quadrant 7: bottom-left-back
        fft_padded[..., -d_half:, -h_half:, :w_half] = img_fft[..., -d_half:, -h_half:, :w_half]
        # Quadrant 8: bottom-right-back
        fft_padded[..., -d_half:, -h_half:, -w_half:] = img_fft[..., -d_half:, -h_half:, -w_half:]
    else:
        h_half, w_half = fft_shape[0]//2, fft_shape[1]//2
        new_h_half, new_w_half = new_fft_shape[0]//2, new_fft_shape[1]//2
        
        # Place original frequency components in new frequency space
        # Quadrant 1: top-left
        fft_padded[..., :h_half, :w_half] = img_fft[..., :h_half, :w_half]
        # Quadrant 2: top-right
        fft_padded[..., :h_half, -w_half:] = img_fft[..., :h_half, -w_half:]
        # Quadrant 3: bottom-left
        fft_padded[..., -h_half:, :w_half] = img_fft[..., -h_half:, :w_half]
        # Quadrant 4: bottom-right
        fft_padded[..., -h_half:, -w_half:] = img_fft[..., -h_half:, -w_half:]
    
    # Perform inverse Fourier transform
    if is_3d:
        img_ifft = torch.fft.ifftn(fft_padded, dim=(-3, -2, -1))
    else:
        img_ifft = torch.fft.ifft2(fft_padded, dim=(-2, -1))
    
    # Take real part
    img_real = img_ifft.real
    
    # Calculate crop indices
    if is_3d:
        start_d = (new_fft_shape[0] - target_depth) // 2
        start_h = (new_fft_shape[1] - target_height) // 2
        start_w = (new_fft_shape[2] - target_width) // 2
        upsampled_images = img_real[..., start_d:start_d+target_depth, 
                                   start_h:start_h+target_height, 
                                   start_w:start_w+target_width]
        # Normalization factor for 3D
        upsampled_images = upsampled_images / (factor ** 3)
    else:
        start_h = (new_fft_shape[0] - target_height) // 2
        start_w = (new_fft_shape[1] - target_width) // 2
        upsampled_images = img_real[..., start_h:start_h+target_height, 
                                   start_w:start_w+target_width]
        # Normalization factor for 2D
        upsampled_images = upsampled_images / (factor ** 2)
    
    return upsampled_images