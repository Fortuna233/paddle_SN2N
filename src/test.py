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


# input raw_map and output prediction file and a gif of to map
def predict(model, raw_map, map_index, box_size=48):
    predicFolder = './predictions'
    raw_map = raw_map.clip(min=0.0, max=np.percentile(raw_map[raw_map > 0], 99.999)) / np.percentile(raw_map[raw_map > 0], 99.999)
    threshold = np.percentile(raw_map, 30)
    map_shape = raw_map.shape

    chunk_files = [os.path.join(predicFolder, f) for f in os.listdir(predicFolder) if f.endswith('.npz') and int(os.path.splitext(f)[0].split("_")[0]) == map_index]
    predSet = myDataset(chunk_files, is_train=False)
    pred_iter = DataLoader(predSet, batch_size=48, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    devices = try_all_gpus()
    model = model.to(device=devices[0])
    model = torch.compile(model)
    paramsFolder = "./params"           
    _, current_epochs = get_all_files(paramsFolder)
    if current_epochs != 0:
        state_dict = torch.load(f'params/checkPoint_{current_epochs - 1}')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f'load params/checkPoint_{current_epochs - 1}')
    devices = try_all_gpus()
    map = np.zeros(tuple(dim + 2 * box_size for dim in map_shape), dtype=np.float32)
    denominator = np.zeros_like(map)

    with torch.no_grad():
        for i, (X, chunk_positions) in enumerate(pred_iter):
            if torch.max(X) >= threshold:
                X = X.reshape(X.shape[0], 1, box_size, box_size, box_size).to(devices[0])
                X = model(X).reshape(-1, box_size, box_size, box_size).cpu()
            else:
                X = 0

            for index, (chunk, chunk_position) in enumerate(zip(X.numpy(), chunk_positions)):
                map[chunk_position[0]:chunk_position[0] + box_size,
                    chunk_position[1]:chunk_position[1] + box_size,
                    chunk_position[2]:chunk_position[2] + box_size] += chunk
                denominator[chunk_position[0]:chunk_position[0] + box_size,
                    chunk_position[1]:chunk_position[1] + box_size,
                    chunk_position[2]:chunk_position[2] + box_size] += 1
                
            # for index, chunk_position in enumerate(chunk_positions):
                filepath = os.path.join('./predictions', f"{map_index}_{chunk_position[0]}_{chunk_position[1]}_{chunk_position[2]}")
                np.savez_compressed(filepath, X[index, :, :, :].numpy())
                print(f"[{i}/{len(pred_iter)}] save {filepath}")

    return (map / denominator.clip(min=1))[box_size : map_shape[0] + box_size, box_size : map_shape[1] + box_size, box_size : map_shape[2] + box_size]



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

