import torch
import torch.backends
import torch.utils
from src.utils.utils_train_predict import train


import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True
from src.SN2N_2D.constants_2d import world_size, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, lr, accumulations_steps, kernel, model

if __name__ == "__main__":
    mp.spawn(train, args=(world_size, model, kernel, paramsFolder, datasetsFolder, logsFolder, num_epochs, batch_size, lr, accumulations_steps), nprocs=world_size, join=True)