
import torch
import torch.backends
from src.utils.utils_train_predict import predict2d
from src.SN2N_2D.constants_2d import rawdataFolder, paramsFolder, resultFolder, model


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True


if __name__ == "__main__":
    predict2d(model=model, rawdataFolder=rawdataFolder, paramsFolder=paramsFolder, resultFolder=resultFolder)
    