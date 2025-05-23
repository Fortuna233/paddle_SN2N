import os
import time
import torch
import numpy as np
from torch import nn
from math import ceil
from scunet import SCUNet
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch import FloatTensor as FT
from torch.autograd import Variable as V
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Load data
# save_path = "data1/ryi/paddle_SN2N/datasets"
# paramsFolder = "data1/ryi/paddle_SN2N/params"
save_path = "/home/tyche/paddle_SN2N/datasets"
paramsFolder = "/home/tyche/paddle_SN2N/params"

_, current_epochs = get_all_files(paramsFolder)
n_chunks, i = 0, 0
chunkList, n_chunks = get_all_files(save_path)
chunks_file = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.npz')]

trainData, valiData = train_test_split(chunks_file, test_size=0.25, random_state=42)
trainSet = myDataset(trainData, is_train=True)
valiSet = myDataset(valiData, is_train=False)

train_iter = DataLoader(trainData, batch_size=48, shuffle=True, num_workers=12, pin_memory=True)
vali_iter = DataLoader(trainData, batch_size=48, shuffle=False, num_workers=12, pin_memory=True)


# 输入为torch张量batch_size*60*60*60
model = SCUNet(
    in_nc=1,
    config=[2,2,2,2,2,2,2],
    dim=32,
    drop_path_rate=0.0,
    input_resolution=64,
    head_dim=16,
    window_size=3,
)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)
if current_epochs != 0:
    model.load_state_dict(torch.load(f'params/checkPoint_{current_epochs - 1}'))
devices = try_all_gpus()
model = model.to(devices[0])
#model = torch.nn.DataParallel(model)
# model = torch.nn.DataParallel(model, [0, 1, 2])


# 定义trainer
trainer = torch.optim.Adam(
    model.parameters(),
    lr=0.00001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5,
    amsgrad=True
)


batch_size = 48
num_epochs = 300
train_Loss = []
vali_Loss = []
for epoch in range(current_epochs, num_epochs):
    train_loss = 0
    vali_loss = 0
    cur_steps = 0
    model.train()
    for X in train_iter:
        print(X) 
        Y = []
        print(f"X.shape: {X}")
        for i in range(X.shape[1]):
            x = X[:, i, :, :, :]
            print(x.shape)
            x = V(FT(x), requires_grad=True).view(-1, 1, 64, 64, 64)
            x = x.to(devices[0])
            y = model(x).view(-1, 64, 64, 64).to(devices[0])
            Y.append(y)
        Y_tensor = torch.as_tensor(Y).view(X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        del Y
        l = 0
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i != j:
                    l += loss(X[:, i, :, :, :], Y[:, j, :, : ,:])
                    l += loss(X[:, j, :, :, :], Y[:, i, :, : ,:])
                    l += loss(Y[:, i, :, :, :], Y[:, j, :, : ,:])

        
        trainer.zero_grad()
        l.backward()
        trainer.step()
        train_loss += l
        cur_steps += len(X)
        with open("output.log", "a") as file:
            file.write(f"[epoch: {epoch}] [processing: {cur_steps} / {len(train_iter)}] [loss: {l}]\n")
        print(f"[epoch: {epoch}] [processing: {cur_steps} / {len(train_iter)}] [loss: {l}]")
    train_Loss.append(train_loss / len(train_iter))
    
    with torch.no_grad():
        for X, y in vali_iter:
            X = V(FT(X), requires_grad=True).view(-1, 1, 48, 48, 48)
            X = X.to(devices[0])
            y = y.to(devices[0])        
            l = loss(model(X).view(-1, 48, 48, 48), y)
            vali_loss += l
        vali_Loss.append(vali_loss / len(vali_iter))
    
    with open("output.log", "a") as file:
        file.write(f"epoch:{epoch} train_loss:{train_loss / len(train_iter)} vali_loss:{vali_loss  / len(vali_iter)}\n")
    print(f"epoch:{epoch} train_loss:{train_loss  / len(train_iter)} vali_loss:{vali_loss  / len(vali_iter)}")
    print(f"checkPoint_{epoch}") 
    print("=================================================================================================================")
    torch.save(model.module.state_dict(), f"params/checkPoint_{epoch}")



train_Loss = np.array(train_Loss)
vali_Loss = np.array(vali_Loss)
np.savez_compressed('Loss.npz', train_Loss, vali_Loss)

plt.plot(range(num_epochs), train_Loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
