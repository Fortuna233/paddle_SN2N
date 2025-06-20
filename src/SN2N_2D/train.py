import torch
import torch.backends
import torch.utils
from src.utils.utils_train_predict import *
from monai.networks.nets import UNet
from src.models.scunet import SCUNet

import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True
from src.SN2N_2D.constants_2d import paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, accumulations_steps, kernel

if __name__ == "__main__":
    world_size = 1
    # model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
    #            dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
    model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=[64, 128, 256, 512],
                strides=(2, 2, 2), num_res_units=2)

    mp.spawn(train, args=(world_size, model, kernel, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, accumulations_steps), nprocs=world_size, join=True)