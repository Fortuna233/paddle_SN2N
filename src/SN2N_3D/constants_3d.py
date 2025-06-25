import torch
from monai.networks.nets import UNet
from src.models.scunet import SCUNet

num_epochs = 50
batch_size = 64
lr = 1e-4
vali_ratio = 0.1
box_size = 256
stride = 64
stride_inference = 64
accumulations_steps = 1


rawdataFolder = '/home/tyche/paddle_SN2N/data/data_3d/raw_data'
datasetsFolder = '/home/tyche/paddle_SN2N/data/data_3d/datasets'
paramsFolder = '/home/tyche/paddle_SN2N/data/data_3d/params'
predictionsFolder = '/home/tyche/paddle_SN2N/data/data_3d/predictions'
logsFolder = '/home/tyche/paddle_SN2N/data/data_3d/logs'
resultFolder = '/home/tyche/paddle_SN2N/data/data_3d/results'

# model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
#            dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[64, 128, 256, 512],
            strides=(2, 2, 2), num_res_units=2)

# kernel = torch.tensor([[[[0, 1], [1, 1]], [[1, 1], [1, 1]]],
#                        [[[1, 0], [1, 1]], [[1, 1], [1, 1]]],
#                        [[[1, 1], [0, 1]], [[1, 1], [1, 1]]],
#                        [[[1, 1], [1, 0]], [[1, 1], [1, 1]]],
#                        [[[1, 1], [1, 1]], [[0, 1], [1, 1]]],
#                        [[[1, 1], [1, 1]], [[1, 0], [1, 1]]],
#                        [[[1, 1], [1, 1]], [[1, 1], [0, 1]]],
#                        [[[1, 1], [1, 1]], [[1, 1], [1, 0]]]]).float() / 7
kernel = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], 
                       [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]]).float() / 4   
if __name__ == '__main__':
    print(kernel)


