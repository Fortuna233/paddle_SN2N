import torch
from monai.networks.nets import UNet
from src.models.scunet import SCUNet

mode = '2d'
world_size = 1
num_epochs = 20
batch_size = 128
lr = 1e-4
vali_ratio = 0.1
box_size = 128
stride = 32
stride_inference = 120
accumulations_steps = 1


rawdataFolder = '/home/tyche/paddle_SN2N/data/data_2d/raw_data'
datasetsFolder = '/home/tyche/paddle_SN2N/data/data_2d/datasets'
paramsFolder = '/home/tyche/paddle_SN2N/data/data_2d/params'
logsFolder = '/home/tyche/paddle_SN2N/data/data_2d/logs'
visualizationFolder = '/home/tyche/paddle_SN2N/data/data_2d/process_visualization'
resultFolder = '/home/tyche/paddle_SN2N/data/data_2d/result'

# model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
#            dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
model = UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=[64, 128, 256, 512],
            strides=(2, 2, 2), num_res_units=2)
kernel = torch.tensor([[[1, 0], [0, 1]],
                      [[0, 1], [1, 0]]]).float() / 2

# kernel = torch.tensor([[[1, 1], [1, 0]],
#                       [[1, 1], [0, 1]],
#                       [[1, 0], [1, 1]],
#                       [[0, 1], [1, 1]]]).float() / 3