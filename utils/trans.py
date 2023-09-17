import torch
import numpy as np
import copy

def model2array(model):
    array=np.array([])
    for key,var in model.state_dict().items():
        layer_array = var.clone().cpu().detach().numpy().reshape(-1)
        array = np.concatenate((array, layer_array), axis=0)
    return np.copy(array)

def array2model(Model, Param):
    model_dict = copy.deepcopy(Model.state_dict())
    idx = 0
    for key,var in Model.state_dict().items():
        length = var.reshape(-1).shape[0]
        model_dict[key].data.copy_(torch.from_numpy(Param[idx:idx+length].reshape(var.shape)))
        # model_dict[key] = (torch.from_numpy(Param[idx:idx+length].reshape(var.shape).copy()))
        idx += length
    
    Model.load_state_dict(model_dict, strict=True)
    return Model