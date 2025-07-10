
import torch
import torch.backends
from src.utils.utils_dataprocessing import get_all_files
from src.utils.utils_train_predict import predict
from src.SN2N_2D.constants_2d import rawdataFolder, paramsFolder, resultFolder, model, mode


torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True


if __name__ == "__main__":
    predict(model=model, rawdataFolder=rawdataFolder, paramsFolder=paramsFolder, resultFolder=resultFolder, mode=mode)
    