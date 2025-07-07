import torch
from monai.networks.nets import UNet
from src.models.scunet import SCUNet

mode = '2d'
world_size = 1
num_epochs = 10
batch_size = 128
lr = 1e-4
vali_ratio = 0.1
box_size = 128
stride = 64
stride_inference = 32
accumulations_steps = 1


rawdataFolder = '/home/tyche/paddle_SN2N/data/data_2d/raw_data'
datasetsFolder = '/home/tyche/paddle_SN2N/data/data_2d/datasets'
paramsFolder = '/home/tyche/paddle_SN2N/data/data_2d/params'
predictionsFolder = '/home/tyche/paddle_SN2N/data/data_2d/predictions'
logsFolder = '/home/tyche/paddle_SN2N/data/data_2d/logs'
resultFolder = '/home/tyche/paddle_SN2N/data/data_2d/results'

# model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
#            dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=[16, 32, 64, 128],
            strides=(2, 2, 2), num_res_units=2)
kernel = torch.tensor([[[1, 0], [0, 1]],
                      [[0, 1], [1, 0]]]).float() / 2

# kernel = torch.tensor([[[1, 1], [1, 0]],
#                       [[1, 1], [0, 1]],
#                       [[1, 0], [1, 1]],
#                       [[0, 1], [1, 1]]]).float() / 3