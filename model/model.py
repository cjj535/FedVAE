from model.VAE import *
from model.WAE import *
from model.VQVAE import *
from model.DSVDD import *
from model.weightWAE import *
from model.DAGMM import *
from utils.Params import *

def choose_model(param_list, model_name, dataset_name):
    model = None
    if dataset_name == 'UNSW-NB15':
        if model_name == 'VAE':
            model = VAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=1.0).to(device)
        elif model_name == 'BetaVAE':
            model = VAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=0.25).to(device)
        elif model_name == 'WAE':
            model = WAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=0.25).to(device)
        elif model_name == 'weightWAE':
            model = weightWAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=0.25).to(device)
        else:
            raise Warning('others are not implemented.')
    elif dataset_name == 'NSL-KDD':
        if model_name == 'VAE':
            model = VAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=1.0).to(device)
        elif model_name == 'BetaVAE':
            model = VAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=1.0).to(device)
        elif model_name == 'WAE':
            model = WAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=1.0).to(device)
        elif model_name == 'weightWAE':
            model = weightWAE(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], weight=1.0).to(device)
        else:
            raise Warning('others are not implemented.')

    return model

def choose_DSVDD_model(param_list):
    model = DSVDD(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4]).to(device)
    return model

def choose_DAGMM_model(param_list):
    model = DAGMM(param_list[0], param_list[1], param_list[2], param_list[3], param_list[4]).to(device)

    return model