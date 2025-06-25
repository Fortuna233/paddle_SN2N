from locale import normalize
import torch
import tifffile
import torch.backends
import torch.utils
import numpy as np
import torch.multiprocessing as mp
from src.utils.utils_dataprocessing import get_all_files, normalize
from src.utils.utils_train_predict import predict, try_all_gpus, try_all_gpus
from src.SN2N_2D.constants_2d import rawdataFolder, paramsFolder, resultFolder, box_size, stride_inference, model
from src.utils.utils_visualization import plot_multiple_tensors, combine_tensors_to_gif

torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True



if __name__ == "__main__":
    devices = try_all_gpus()
    model = model.to(devices[0])
    model = torch.nn.DataParallel(model, device_ids=[0])
    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv3d or type(m) == torch.nn.Conv2d:  
            torch.nn.init.xavier_uniform_(m.weight)

    current_epoch = len(get_all_files(paramsFolder))
    if current_epoch != 0:
        state_dict = torch.load(f'{paramsFolder}/checkPoint_{current_epoch - 1}')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)

        if missing_keys:
            print(f"missing_keys: {missing_keys}")
        if unexpected_keys:
            print(f"unused_keys: {unexpected_keys}")
        
        print(f'load {paramsFolder}/checkPoint_{current_epoch - 1}')
    else:
        model.apply(init_weights)
        print(f'no params found, randomly init model')

    map_files = get_all_files(rawdataFolder)
    for map_index, map_file in enumerate(map_files):
        print(map_file)    
        raw_map = np.asarray(tifffile.imread(map_file))
        raw_map = normalize(raw_map, mode='2d')
        if len(raw_map.shape) == 2:
            raw_map = raw_map.reshape(-1, *raw_map.shape)

        denoised_map = predict(model, raw_map, box_size=box_size, stride=stride_inference, paramsFolder=paramsFolder, mode='2d')
        denoised_map = normalize(denoised_map, minPercent=0, maxPercent=99.999, mode='2d')
        tifffile.imwrite(f'{resultFolder}/processed_maps/{map_index}_denoised.tif', denoised_map, imagej=True, metadata={'axes': 'ZYX'}, compression='zlib')

        combine_tensors_to_gif(
            tensors={
                "raw_map": raw_map,
                "denoised_map": denoised_map
            },
            output_path=f"{resultFolder}/combined_maps/{map_index}.gif",
            fps=2,
            cmap="grey"
        )
        plot_multiple_tensors(tensors=[raw_map, denoised_map],
                              titles=["raw_map", "denoised_map"], 
                              output_path=f"{resultFolder}/histograms/{map_index}.png")
    