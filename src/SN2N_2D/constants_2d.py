import torch


num_epochs = 50
batch_size = 64
lr = 1e-4
vali_ratio = 0.1
box_size = 256
stride = 64
stride_inference = 64
accumulations_steps = 1


rawdataFolder = '/home/tyche/paddle_SN2N/data/raw_data'
datasetsFolder = '/home/tyche/paddle_SN2N/data/datasets'
paramsFolder = '/home/tyche/paddle_SN2N/data/params'
predictionsFolder = '/home/tyche/paddle_SN2N/data/predictions'
logsFolder = '/home/tyche/paddle_SN2N/data/logs'
resultFolder = '/home/tyche/paddle_SN2N/data/results'



# kernel = torch.tensor([[[1, 0], [0, 1]],
#                       [[0, 1], [1, 0]]]).float() / 2

kernel = torch.tensor([[[1, 1], [1, 0]],
                      [[1, 1], [0, 1]],
                      [[1, 0], [1, 1]],
                      [[0, 1], [1, 1]]]).float() / 3