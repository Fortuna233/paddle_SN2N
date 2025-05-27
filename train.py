import os
import time
import torch
import argparse
import torch.backends
import torch.utils
import numpy as np
from utils import *
from torch import nn
from monai.networks.nets import UNet
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.nn import DataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environp['RANK'] = str(local_rank) 
    dist.init_process_group(backend='nccl', init_method='env://')


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


def get_ddp_generator(seed=42):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True
batch_size = 24
num_epochs = 300
accumulation_steps = 5
save_path = "/home/tyche/paddle_SN2N/datasets"
# save_path = "/data1/ryi/paddle_SN2N/datasets"


train_iter = get_dataset(save_path=save_path, batch_size=batch_size)


# model = SCUNet(
#     in_nc=1,
#     config=[2,2,2,2,2,2,2],
#     dim=32,
#     drop_path_rate=0.1,
#     input_resolution=48,
#     head_dim=16,
#     window_size=3
# )


total_params = sum(p.numel() for p in model.parameters())
print(f"num_parameters: {total_params}")  # 输出: 526,186
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv3d:  
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)


# paramsFolder = "/data1/ryi/paddle_SN2N/params"
paramsFolder = "/home/tyche/paddle_SN2N/params"
_, current_epochs = get_all_files(paramsFolder)
if current_epochs != 0:
    model.load_state_dict(torch.load(f'params/checkPoint_{current_epochs - 1}'))
devices = try_all_gpus()
model = model.to(device=devices[0])
# model = DataParallel(model, device_ids=[0, 1, 2])
model = torch.compile(model)

trainer = torch.optim.Adam(model.parameters(), 
                           lr=0.00001, 
                           betas=(0.9, 0.999), 
                           eps=1e-8, 
                           weight_decay=1e-4, 
                           amsgrad=True)

scheduler = OneCycleLR(trainer, max_lr=0.01, total_steps=batch_size*len(train_iter),
                       pct_start=0.3,
                       anneal_strategy='cos',
                       cycle_momentum=True,
                       base_momentum=0.85,
                       max_momentum=0.95,
                       div_factor=25,
                       final_div_factor=1e5)


scaler = GradScaler()
# 主训练流程
Time = 0
model.train()
train_Loss = []
kernel = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], 
                       [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]]).float()
starttime = time.time()
for epoch in range(current_epochs, num_epochs):
    # train_iter.sampler.set_epoch(epoch)
    train_loss = 0
    loss_batch = 0
    for i, X in enumerate(train_iter):
        X = resample(X, kernel=kernel)
        batch_size, channels, *spatial_dims = X.shape
        X = X.reshape(batch_size * channels, 1, *spatial_dims).to(devices[0])
        with autocast(device_type='cuda'):
            Y = model(X)
            X = X.reshape(batch_size, channels, *spatial_dims)
            Y = Y.reshape(batch_size, channels, *spatial_dims)
            t1 = time.time()
            l = loss_channels(X, Y)
            t2 = time.time()
            print(f"损失函数计算:{t2 - t1}")
            loss_batch += l / accumulation_steps
            del X, Y
        t1 = time.time()
        scaler.scale(l).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        t2 = time.time()
        print(f"反向传播{t2 - t1}")

        if (i + 1) % accumulation_steps == 0:
            t1 = time.time()
            scaler.step(trainer)
            scaler.update()
            scheduler.step()
            trainer.zero_grad(set_to_none=True)
            t2 = time.time()
            print(f"梯度下降{t2 - t1}")
            train_loss += l.item()
            endtime = time.time()
            Time = endtime - starttime  
            log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [loss: {loss_batch.item():.6f}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
            print(log_msg)
            loss_batch = 0
            with open("output.log", "a") as file:
                file.write(log_msg + "\n")
        
    # 记录本轮损失
    epoch_loss = train_loss / len(train_iter)
    train_Loss.append(epoch_loss)
        
    # 保存模型和日志

    log_msg = f"[epoch: {epoch}] [processing: {i + 1}/{len(train_iter)}] [loss: {loss_batch.item():.6f}] [lr: {trainer.param_groups[0]['lr']}] [Time: {Time:.2f}s]"
    print(log_msg)
    print(f"checkPoint_{epoch}")
    print("=================================================================================================================")
    
    with open("output.log", "a") as file:
        file.write(log_msg + "\n")

    torch.save(model.module.state_dict(), f"params/checkPoint_{epoch}")
    
# 保存损失曲线并绘制
train_Loss = np.array(train_Loss)
np.savez_compressed('Loss.npz', train_Loss)

plt.plot(range(num_epochs), train_Loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')  # 保存图像
plt.show()


def main(local_rank, args):
    init_ddp(local_rank)
    best_macro = 0
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, 
                 channels=(16, 32, 64, 128), 
                 strides=(2, 2, 2), 
                 num_res_units=2)
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f'use {num_gpus} gpus!')
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    num_training_steps = args['num_epochs'] * args['num_samples'] // args['batch_size']

    if args['requires_grad']:
        param_optimizer =list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(args['learning_rate']))
    
    else:
        optimizer = AdamW(model.parameters(), lr=float(args['learning_rate']))
    
    scheduler = OneCycleLR(trainer, max_lr=0.01, total_steps=batch_size*len(train_iter),
                           pct_start=0.3,
                           anneal_strategy='cos',
                           cycle_momentum=True,
                           base_momentum=0.85,
                           max_momentum=0.95,
                           div_factor=25,
                           final_div_factor=1e5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-args', help="priority", type=bool, required=False, default=True)
    parser.add_argument('-gpu', default='0.1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-mode', help='train || test', type=str, required=False, default='train')
    parser.add_argument('requires_grad', help='whether to weight_decay', type=bool, required=False, default=True)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '19198'  
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']  
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  

    if args['mode'] == 'train':
        time_start = time.time()
        mp.spawn(fn=main, args=(args, ), nprocs=world_size)
        time_elapsed = time.time() - time_start
        print(f'\ntime elapsed: {time_elapsed:.2f}s.')

    elif args['mode'] == 'test':
        time_start = time.time()
        mp.spawn(fn=test, args=(args, ), nprocs=world_size)
        time_elapsed = time.time() - time_start
        print(f'\ntime elapsed: {time_elapsed:.2f}s.')