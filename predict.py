from utils import *

save_path


paramsFolder = "./params"
_, current_epochs = get_all_files(paramsFolder)
if current_epochs != 0:
    model.load_state_dict(torch.load(f'params/checkPoint_{current_epochs - 1}'))
devices = try_all_gpus()
model = model.to(device=devices[0])
raw_maps = get_all_files()