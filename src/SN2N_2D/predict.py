import torch
import torch.backends
from src.utils.utils_train_predict import predict2d
from src.SN2N_2D.constants_2d import rawdataFolder, paramsFolder, resultFolder, model, box_size, batch_size, stride_inference


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True


if __name__ == "__main__":
    predict2d(model=model, box_size=box_size, stride=stride_inference, batch_size=batch_size, rawdataFolder=rawdataFolder, paramsFolder=paramsFolder, resultFolder=resultFolder)
    