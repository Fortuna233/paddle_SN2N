import os
import time
import tifffile
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



import torch
import torch.utils
from torch import nn
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from src.utils.utils_ddp import *
from src.utils.utils_dataprocessing import get_all_files, resample, normalize


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class myDataset(Dataset):
    def __init__(self, file_list, is_train):
        self.file_list = file_list
        self.is_train = is_train
        # self.train_augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])

    def __len__(self):
        return len(self.file_list)
    

    def __getitem__(self, index):
        file_path = self.file_list[index]
        data = np.load(file_path)['arr_0']
        filename = os.path.basename(file_path)
        parts = os.path.splitext(filename)[0].split("_")
        chunk_positions = np.array(parts, dtype=int)[1:]  # 跳过map_index，取x,y,z坐标
        if self.is_train:
            # return self.train_augs(torch.from_numpy(data)), chunk_positions
            return torch.from_numpy(data), chunk_positions
        return torch.from_numpy(data), chunk_positions
        

def get_dataset(save_path, batch_size):
    chunks_file = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.npz')]
    trainData, valiData = train_test_split(chunks_file, test_size=0.1, random_state=42)
    trainSet = myDataset(trainData, is_train=True)
    valiSet = myDataset(valiData, is_train=False)
    train_iter = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    vali_iter = DataLoader(valiSet, batch_size=batch_size, shuffle=False, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    return train_iter, vali_iter


def loss_channels(X, Y, beta=1):
    L1loss = torch.nn.SmoothL1Loss(reduction='mean')
    num_channels = X.shape[1]
    i_indices, j_indices = torch.triu_indices(num_channels, num_channels, offset=1)
    l_xy = L1loss(X[:, i_indices], Y[:, j_indices]) + L1loss(X[:, j_indices], Y[:, i_indices])
    l_yy = L1loss(Y[:, i_indices], Y[:, j_indices])
    return (l_xy + beta * l_yy) / (2 + beta)


def train(rank, world_size, model, kernel, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, lr, accumulation_steps):
    setup(rank, world_size)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"num_parameters: {total_params}")
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv3d or type(m) == torch.nn.Conv2d:  
            torch.nn.init.xavier_uniform_(m.weight)
    
    
    current_epoch = len(get_all_files(paramsFolder))
    print(f"current_epoch:{current_epoch}")
    if current_epoch != 0:
        state_dict = torch.load(f'{paramsFolder}/checkPoint_{current_epoch - 1}.pth')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if missing_keys:
            print(f"missing_keys: {missing_keys}")
        if unexpected_keys:
            print(f"unused_keys: {unexpected_keys}")
    
        print(f'load {paramsFolder}/checkPoint_{current_epoch - 1}')
    else:
        model.apply(init_weights)
        print(f'no params found, randomly init model')
    model = create_ddp_model(rank=rank, model=model)  

    
    chunks_file = [os.path.join(datasetsFolder, f) for f in os.listdir(datasetsFolder) if f.endswith('.npz')]
    trainData, valiData = train_test_split(chunks_file, test_size=0.1, random_state=42)
    trainSet = myDataset(trainData, is_train=True)
    valiSet = myDataset(valiData, is_train=False)
    train_iter = prepare_dataloader(trainSet, batch_size=batch_size, is_train=True)
    vali_iter = prepare_dataloader(valiSet, batch_size=batch_size, is_train=False)

    trainer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    scheduler = OneCycleLR(trainer, max_lr=lr, total_steps=int(len(train_iter)*(num_epochs - current_epoch)/accumulation_steps), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,div_factor=25,final_div_factor=1e5)
    scaler = GradScaler()

    train_Loss = []
    vali_Loss = []


    starttime = time.time()
    local_now = datetime.now()
    if rank==0:
        print(local_now)
    for epoch in range(current_epoch, num_epochs):
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
                l = loss_channels(X, Y)
                train_loss += l.item()
                loss_batch += l.item() / accumulation_steps
                del X, Y
            
            scaler.scale(l).backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (i + 1) % accumulation_steps == 0:
                scaler.step(trainer)
                scaler.update()
                scheduler.step()
                trainer.zero_grad()
                Time = time.time() - starttime
                log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [loss: {loss_batch}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
                if rank == 0:
                    print(log_msg)
                    loss_batch = 0
                    with open(f"{logsFolder}/output_{local_now}.log", "a") as file:
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
                l = loss_channels(X, Y)
                vali_loss += l.item()
        vali_Loss.append(vali_loss / len(vali_iter))

        
        log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [train_loss: {train_Loss[-1]}] [vali_loss: {vali_Loss[-1]}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
        if rank == 0:
            print(log_msg)
            print(f"checkPoint_{epoch}")
            print("=================================================================================================================")
            with open(f"{logsFolder}/output_{local_now}.log", "a") as file:
                file.write(log_msg + "\n")  
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f"{paramsFolder}/checkPoint_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"{paramsFolder}/checkPoint_{epoch}.pth")

    cleanup()


# # input raw_map and output prediction file
def predict2d(model, rawdataFolder, paramsFolder, resultFolder):
    devices = try_all_gpus()
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv3d or type(m) == torch.nn.Conv2d:  
            torch.nn.init.xavier_uniform_(m.weight)
    current_epoch = len(get_all_files(paramsFolder))
    print(f"current_epoch:{current_epoch}")
    if current_epoch != 0:
        state_dict = torch.load(f'{paramsFolder}/checkPoint_{current_epoch - 1}.pth')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if missing_keys:
            print(f"missing_keys: {missing_keys}")
        if unexpected_keys:
            print(f"unused_keys: {unexpected_keys}")
    
        print(f'load {paramsFolder}/checkPoint_{current_epoch - 1}')
    else:
        model.apply(init_weights)
        print(f'no params found, randomly init model')
    
    model = model.to(devices[0])
    map_files = get_all_files(rawdataFolder)
    model.eval()
    with torch.no_grad():
        for map_index, map_file in enumerate(map_files): 
            print(f"processing: {map_file}")
            map_data = np.asarray(tifffile.imread(map_file))
            map_data = torch.from_numpy(normalize(map_data, mode='2d'))
            
            predicted_map = model(raw_map.to(device=devices[0])).cpu().numpy()
            raw_map = raw_map.cpu().numpy()
            tifffile.imwrite(f'{resultFolder}/processed_maps/{map_index}_denoised.tif', predicted_map, imagej=True, metadata={'axes': 'ZYX'}, compression=None)
            tifffile.imwrite(f'{resultFolder}/combined_maps/{map_index}_combined.tif', np.concatenate((raw_map, predicted_map), axis=1), imagej=True, metadata={'axes': 'ZYX'}, compression=None)


def predict3d(model, rawdataFolder, paramsFolder, resultFolder):
    devices = try_all_gpus()
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv3d or type(m) == torch.nn.Conv2d:  
            torch.nn.init.xavier_uniform_(m.weight)
    current_epoch = len(get_all_files(paramsFolder))
    print(f"current_epoch:{current_epoch}")
    if current_epoch != 0:
        state_dict = torch.load(f'{paramsFolder}/checkPoint_{current_epoch - 1}.pth')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if missing_keys:
            print(f"missing_keys: {missing_keys}")
        if unexpected_keys:
            print(f"unused_keys: {unexpected_keys}")
    
        print(f'load {paramsFolder}/checkPoint_{current_epoch - 1}')
    else:
        model.apply(init_weights)
        print(f'no params found, randomly init model')
    
    model = model.to(devices[0])
    map_files = get_all_files(rawdataFolder)
    model.eval()
    with torch.no_grad():
        for map_index, map_file in enumerate(map_files): 
            print(f"processing: {map_file}")
            raw_map = np.asarray(tifffile.imread(map_file))
            normalized_map = torch.from_numpy(normalize(raw_map, mode='3d'))
            predicted_map = model(raw_map.to(device=devices[0])).cpu().numpy()
            raw_map = raw_map.cpu().numpy()
            tifffile.imwrite(f'{resultFolder}/processed_maps/{map_index}_denoised.tif', predicted_map, imagej=True, metadata={'axes': 'ZYX'}, compression=None)
            tifffile.imwrite(f'{resultFolder}/combined_maps/{map_index}_combined.tif', np.concatenate((raw_map, predicted_map), axis=1), imagej=True, metadata={'axes': 'ZYX'}, compression=None)


