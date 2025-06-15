import os
import time
import imageio
import mrcfile
import tifffile
import numpy as np
import multiprocessing
from pathlib import Path
from functools import partial
from itertools import product
import matplotlib.pyplot as plt
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed


import torch
import torch.utils
from torch import nn
import torch.backends
import torch.fft as fft
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from utils_ddp import *




def get_all_files(directory):
    file_list = []
    for file in os.listdir(directory):
        file_list.append(f"{directory}/{file}")
    return sorted(file_list)


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


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


class myDataset(Dataset):
    def __init__(self, file_list, is_train):
        self.file_list = file_list
        self.is_train = is_train
        self.train_augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])

    def __len__(self):
        return len(self.file_list)
    

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = np.load(file_path)['arr_0']
        filename = os.path.basename(file_path)
        parts = os.path.splitext(filename)[0].split("_")
        chunk_positions = np.array(parts, dtype=int)[1:]  # 跳过map_index，取z, x, y坐标
        if self.is_train:
            # return self.train_augs(torch.from_numpy(data)), chunk_positions
            return torch.from_numpy(data), chunk_positions
        else:
            return torch.from_numpy(data), chunk_positions
        

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


def loss_channels(X, Y, gamma=2, beta=0.5):
    def loss(pred, target, gamma, eps=1e-8):
        diff = torch.abs(pred - target)
        loss = (diff + eps) ** gamma
        return loss.mean() 
    num_channels = X.shape[1]
    i_indices, j_indices = torch.triu_indices(num_channels, num_channels, offset=1)
    l_xy = loss(X[:, i_indices].reshape(-1, *X.shape[2:]), Y[:, j_indices].reshape(-1, *Y.shape[2:]), gamma) + loss(X[:, j_indices].reshape(-1, *X.shape[2:]), Y[:, i_indices].reshape(-1, *Y.shape[2:]), gamma)
    l_yy = loss(Y[:, i_indices].reshape(-1, *Y.shape[2:]), Y[:, j_indices].reshape(-1, *Y.shape[2:]), gamma)
    return (l_xy + beta * l_yy) / (2 + beta)


def get_dataset(save_path, batch_size):
    chunks_file = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.npz')]
    trainData, valiData = train_test_split(chunks_file, test_size=0.25, random_state=42)
    trainSet = myDataset(trainData, is_train=True)
    valiSet = myDataset(valiData, is_train=False)
    train_iter = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    vali_iter = DataLoader(valiSet, batch_size=batch_size, shuffle=False, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    return train_iter, vali_iter


def train(rank, world_size, model, num_epochs=20, batch_size=16, accumulation_steps=6):
    setup(rank, world_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"num_parameters: {total_params}")
    
    model = create_ddp_model(rank=rank, model=model)  
    model = torch.compile(model)
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv3d:  
            nn.init.xavier_uniform_(m.weight)
    
    paramsFolder = "./params"
    
    current_epochs = len(get_all_files(paramsFolder))
    if current_epochs != 0:
        model.load_state_dict(torch.load(f'params/checkPoint_{current_epochs - 1}'))
    else:
        model.apply(init_weights)

    save_path='./datasets'
    chunks_file = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.npz')]
    trainData, valiData = train_test_split(chunks_file, test_size=0.25, random_state=42)
    trainSet = myDataset(trainData, is_train=True)
    valiSet = myDataset(valiData, is_train=False)
    train_iter = prepare_dataloader(trainSet, batch_size=batch_size, is_train=True)
    vali_iter = prepare_dataloader(valiSet, batch_size=batch_size, is_train=False)

    trainer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    scheduler = OneCycleLR(trainer, max_lr=0.01, total_steps=batch_size*len(train_iter), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,div_factor=25,final_div_factor=1e5)
    scaler = GradScaler()

    train_Loss = []
    vali_Loss = []
    kernel = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], 
                           [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]]).float()

    starttime = time.time()
    for epoch in range(current_epochs, num_epochs):
        gamma =  2.0 - 2.0 / num_epochs * epoch
        train_iter.sampler.set_epoch(epoch)
        train_loss = 0
        vali_loss = 0
        loss_batch = 0
        model.train()
        for i, (X, _) in enumerate(train_iter):
            X = resample(X, kernel=kernel)
            batch_size, channels, *spatial_dims = X.shape
            X = X.reshape(batch_size * channels, 1, *spatial_dims).to(rank)
            with autocast(device_type='cuda'):
                Y = model(X)
                X = X.reshape(batch_size, channels, *spatial_dims)
                Y = Y.reshape(batch_size, channels, *spatial_dims)
                l = loss_channels(X, Y, gamma)
                train_loss += l.item()
                loss_batch += l.item() / accumulation_steps
                del X, Y
            
            scaler.scale(l).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (i + 1) % accumulation_steps == 0:
                scaler.step(trainer)
                scaler.update()
                scheduler.step()
                trainer.zero_grad(set_to_none=True)
                Time = time.time() - starttime
                log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [loss: {loss_batch}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
                if rank == 0:
                    print(log_msg)
                    loss_batch = 0
                    with open("output.log", "a") as file:
                        file.write(log_msg + "\n")
        epoch_loss = train_loss / len(train_iter)
        train_Loss.append(epoch_loss)

        with torch.no_grad():
            for (X, _) in vali_iter:
                X = resample(X, kernel=kernel)
                batch_size, channels, *spatial_dims = X.shape
                X = X.reshape(batch_size * channels, 1, *spatial_dims).to(rank)
                Y = model(X)
                X = X.reshape(batch_size, channels, *spatial_dims)
                Y = Y.reshape(batch_size, channels, *spatial_dims)
                l = loss_channels(X, Y, gamma)
                vali_loss += l.item()
        vali_Loss.append(vali_loss / len(vali_iter))

        
        log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [train_loss: {loss_batch}] [vali_loss: {vali_Loss[-1]}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
        if rank == 0:
            print(log_msg)
            print(f"checkPoint_{epoch}")
            print("=================================================================================================================")
            with open("output.log", "a") as file:
                file.write(log_msg + "\n")
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f"params/checkPoint_{epoch}")
            else:
                torch.save(model.state_dict(), f"params/checkPoint_{epoch}")
    train_Loss = np.array(train_Loss)
    vali_Loss = np.array(vali_Loss)
    np.savez_compressed('Loss.npz', train_Loss)
    plt.plot(range(num_epochs), train_Loss, label='train loss')
    plt.plot(range(num_epochs), vali_Loss, label='vali loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log(loss)')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')  # 保存图像
    plt.show()


# # input raw_map and output prediction file
def predict(model, map_shape, map_index, box_size=48):
    predicFolder = './predictions'
    chunk_files = [os.path.join(predicFolder, f) for f in os.listdir(predicFolder) if f.endswith('.npz') and int(os.path.splitext(f)[0].split("_")[0]) == map_index]

    predSet = myDataset(chunk_files, is_train=False)
    pred_iter = DataLoader(predSet, batch_size=48, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    devices = try_all_gpus()
    model = model.to(device=devices[0])
    model = torch.compile(model)
    paramsFolder = "./params"
    _, current_epochs = get_all_files(paramsFolder)
    if current_epochs != 0:
        state_dict = torch.load(f'params/checkPoint_{current_epochs - 1}')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f'load params/checkPoint_{current_epochs - 1}')
    devices = try_all_gpus()

    # model = DataParallel(model, device_ids=[0, 1, 2])
    

    map = np.zeros(tuple(dim + 2 * box_size for dim in map_shape), dtype=np.float32)
    denominator = np.zeros_like(map)

    with torch.no_grad():
        for i, (X, chunk_positions) in enumerate(pred_iter):
            X = X.reshape(X.shape[0], 1, box_size, box_size, box_size).to(devices[0])
            X = model(X).reshape(-1, box_size, box_size, box_size).cpu()
            for index, (chunk, chunk_position) in enumerate(zip(X.numpy(), chunk_positions)):
                map[chunk_position[0]:chunk_position[0] + box_size,
                    chunk_position[1]:chunk_position[1] + box_size,
                    chunk_position[2]:chunk_position[2] + box_size] += chunk
                denominator[chunk_position[0]:chunk_position[0] + box_size,
                    chunk_position[1]:chunk_position[1] + box_size,
                    chunk_position[2]:chunk_position[2] + box_size] += 1
                filepath = os.path.join('./predictions', f"{map_index}_{chunk_position[0]}_{chunk_position[1]}_{chunk_position[2]}")
                np.savez_compressed(filepath, X[index, :, :, :].numpy())
                print(f"[{i}/{len(pred_iter)}] save {filepath}")

    return (map / denominator.clip(min=1))[box_size : map_shape[0] + box_size, box_size : map_shape[1] + box_size, box_size : map_shape[2] + box_size]





