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
from scunet import SCUNet

torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True
batch_size = 24
num_epochs = 20
accumulation_steps = 5
save_path = "/home/tyche/paddle_SN2N/datasets"
# save_path = "/data1/ryi/paddle_SN2N/datasets"

model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[4, 8, 16, 32],
             strides=(2, 2, 2),
             num_res_units=2)


train(model=model)


