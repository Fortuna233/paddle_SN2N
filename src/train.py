import os
import time
import torch
import argparse
import torch.backends
import torch.utils
import numpy as np
from utils import *
from monai.networks.nets import UNet
from scunet import SCUNet
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True
save_path = "/home/tyche/paddle_SN2N/datasets"
# save_path = "/data1/ryi/paddle_SN2N/datasets"

model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
               dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
# model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[4, 8, 16, 32],
#              strides=(2, 2, 2), num_res_units=2)

# train(model=model, num_epochs=20, batch_size=4, accumulation_steps=24)


if __name__ == "__main__":
    world_size = 3
    mp.spawn(train, args=(world_size, model), nprocs=world_size, join=True)