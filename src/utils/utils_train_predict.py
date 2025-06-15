import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sympy import im
import torch
import torch.utils
from torch import nn
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from utils_ddp import *
from utils_dataprocessing import get_all_files, resample


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


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
            return self.train_augs(torch.from_numpy(data)), chunk_positions
        else:
            return torch.from_numpy(data), chunk_positions
        

def get_dataset(save_path, batch_size):
    chunks_file = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.npz')]
    trainData, valiData = train_test_split(chunks_file, test_size=0.25, random_state=42)
    trainSet = myDataset(trainData, is_train=True)
    valiSet = myDataset(valiData, is_train=False)
    train_iter = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    vali_iter = DataLoader(valiSet, batch_size=batch_size, shuffle=False, num_workers=4 * torch.cuda.device_count(), pin_memory=True, prefetch_factor=2)
    return train_iter, vali_iter


def loss_channels(X, Y, gamma=2, beta=0.5):
    def L0loss(pred, target, gamma, eps=1e-8):
        diff = torch.abs(pred - target)
        loss = (diff + eps) ** gamma
        return loss.mean() 
    num_channels = X.shape[1]
    i_indices, j_indices = torch.triu_indices(num_channels, num_channels, offset=1)
    l_xy = L0loss(X[:, i_indices].reshape(-1, *X.shape[2:]), Y[:, j_indices].reshape(-1, *Y.shape[2:]), gamma) + L0loss(X[:, j_indices].reshape(-1, *X.shape[2:]), Y[:, i_indices].reshape(-1, *Y.shape[2:]), gamma)
    l_yy = L0loss(Y[:, i_indices].reshape(-1, *Y.shape[2:]), Y[:, j_indices].reshape(-1, *Y.shape[2:]), gamma)
    return (l_xy + beta * l_yy) / (2 + beta)


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
                # filepath = os.path.join('./predictions', f"{map_index}_{chunk_position[0]}_{chunk_position[1]}_{chunk_position[2]}")
                # np.savez_compressed(filepath, X[index, :, :, :].numpy())
                print(f"[predicting: {i}/{len(pred_iter)}]")

    return (map / denominator.clip(min=1))[box_size : map_shape[0] + box_size, box_size : map_shape[1] + box_size, box_size : map_shape[2] + box_size]