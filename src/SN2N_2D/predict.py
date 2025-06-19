from locale import normalize
import torch
import tifffile
import torch.backends
import torch.utils
import numpy as np
import torch.multiprocessing as mp
from src.utils.utils_dataprocessing import get_all_files, combine_tensors_to_gif, normalize
from src.utils.utils_train_predict import predict
from src.constants import rawdataFolder, paramsFolder, datasetsFolder, resultFolder, box_size, stride
from src.models.scunet import SCUNet
from monai.networks.nets import UNet

torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True



if __name__ == "__main__":
    # model = SCUNet(in_nc=1, config=[2, 2, 2, 2, 2, 2, 2],
    #            dim=32, drop_path_rate=0.1, input_resolution=48, head_dim=16, window_size=3)
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[8, 16, 32, 64, 128, 256],
                strides=(2, 2, 2, 2, 2), num_res_units=2)

    
    map_files = get_all_files(rawdataFolder)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[8, 16, 32, 64, 128, 256], 
                 strides=(2, 2, 2, 2, 2), num_res_units=2)
    world_size = 3
    for map_index, map_file in enumerate(map_files):
        print(map_file)    
        raw_map = np.asarray(tifffile.imread(map_file))
        raw_map = normalize(raw_map)

        # denoised_map = mp.spawn(predict, args=(world_size, model, raw_map.shape, map_index, box_size, paramsFolder, datasetsFolder), nprocs=world_size, join=True)
        denoised_map = predict(model, raw_map, box_size=box_size, stride=16, paramsFolder=paramsFolder)
        denoised_map = normalize(denoised_map, mode='2d')
        tifffile.imwrite(f'{resultFolder}/{map_index}_denoised.tif', denoised_map, imagej=True, metadata={'axes': 'ZYX'}, compression='zlib')

        combine_tensors_to_gif(
            tensors={
                "raw_map": raw_map,
                "denoised_map": denoised_map
            },
            output_path=f"{resultFolder}/{map_index}.gif",
            fps=2,
            cmap="grey"
        )    
    