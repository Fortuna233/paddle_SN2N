import os
import time
import torch
import tifffile
import torch.backends
import torch.utils
import numpy as np
from src.utils.utils import *
from monai.networks.nets import UNet
from scunet import SCUNet

torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True
# save_path = "/home/tyche/paddle_SN2N/datasets"
save_path = "./raw_data"
predict_path = "./predictions"
map_files, _ = get_all_files(save_path)
# model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
#                dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
# model = UNet(spatial_dims=3,
#              in_channels=1,
#              out_channels=1,
#              channels=[8, 16, 32, 64, 128],
#              strides=(2, 2, 2, 2),
#              num_res_units=2)
model = UNet(spatial_dims=3,
             in_channels=1,
             out_channels=1,
             channels=[16, 32, 64, 128, 256],
             strides=(2, 2, 2, 2, 2),
             num_res_units=2)

for map_index, map_file in enumerate(map_files):
    # _, map_shape = split_and_save_tensor(map_file=map_file,
    #                           save_dir=predict_path,
    #                           map_index=map_index)
    # print(map_file)
    
    raw_map = np.asarray(tifffile.imread(map_file))
    denoised_map = predict(model, map_shape=raw_map.shape, map_index=map_index)
    
   
    raw_map = raw_map.clip(min=0.0, max=np.percentile(raw_map[raw_map > 0], 99.999)) / np.percentile(raw_map[raw_map > 0], 99.999)
    denoised_map = np.array(tifffile.imread('./c12_SR_w1L-561_t1_denoised.tif'))
    for i in range(denoised_map.shape[0]):
        denoised_map[i] = denoised_map[i].clip(min=0, max=np.percentile(denoised_map[i][denoised_map[i] > 0], 99.999)) / np.percentile(denoised_map[i][denoised_map[i] > 0], 99.999)
    # denoised_map = denoised_map.clip(min=0, max=np.percentile(denoised_map[denoised_map > 0], 99.999)) / np.percentile(denoised_map[denoised_map > 0], 99.999)
    combine_tensors_to_gif(
        tensors={
            "raw_map": raw_map,
            "denoised_map": denoised_map
        },
        output_path="combined_tensors.gif",
        fps=2,
        cmap="grey"
    )    

