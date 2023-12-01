import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# quantile point
# fraction_point = 0.8
# fractor_alpha = 4

# fedprox
prox_factor = 1e-4

# feddyn
feddyn_alpha = 1e-1
isDyn = False

eps = 1e-7

max_norm = 10

unswnb15_small_param = [202,256,64,16,8]
unswnb15_mid_param = [202,256,128,64,32]
unswnb15_big_param = [202,512,256,128,64]
nslkdd_small_param = [121,64,32,16,8]
nslkdd_mid_param = [121,128,64,32,16]
nslkdd_big_param = [121,256,128,64,32]

def choose_param(dataset_name, model_size):
    param_list = None
    if dataset_name=='UNSW-NB15':
        if model_size == 'SMALL':
            param_list = unswnb15_small_param
        elif model_size == 'MID':
            param_list = unswnb15_mid_param
        elif model_size == 'BIG':
            param_list = unswnb15_big_param
    elif dataset_name=='NSL-KDD':
        if model_size == 'SMALL':
            param_list = nslkdd_small_param
        elif model_size == 'MID':
            param_list = nslkdd_mid_param
        elif model_size == 'BIG':
            param_list = nslkdd_big_param
    else:
        raise Warning("no dataset")

    return param_list