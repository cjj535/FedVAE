import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from scipy.interpolate import interp1d
from utils.LoadData import *
from utils.Params import *
from run.test import auc_value

def train_loss(model, dataset):
    model.eval()
    
    loss_list = None
    label_list = None
    
    dataloader = generate_data(dataset, batch_size=2048, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, z = model(inputs)
        _,_,_,loss = model.loss(inputs, recon, z)
        # recon, mu, logvar, _ = model(inputs)
        # _,_,_, loss = model.loss(inputs, recon, mu, logvar)

        if i==0:
            label_list = labels.detach().cpu().numpy().copy()
            loss_list = loss.detach().cpu().numpy().copy()
        else:
            label_list = np.concatenate((label_list, labels.detach().cpu().numpy().copy()), axis=0)
            loss_list = np.concatenate((loss_list, loss.detach().cpu().numpy().copy()), axis=0)
    
    norm_loss = loss_list[label_list==0]
    attk_loss = loss_list[label_list>0]
    
    norm_loss = np.mean(norm_loss)
    attk_loss = np.mean(attk_loss)
    
    return norm_loss, attk_loss

def distance(x, y):
    return np.sum((x-y)**2, axis=1)

def test_auc(model, dataset):
    model.eval()

    dis_list = None
    label_list = None

    dataloader = generate_data(dataset, batch_size=2048, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        recon, _ = model(inputs)
        # recon, _, _, _ = model(inputs)

        if i==0:
            label_list = labels.detach().cpu().numpy().copy()
            dis_list = distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
        else:
            label_list = np.concatenate((label_list, labels.detach().cpu().numpy().copy()), axis=0)
            dis_list = np.concatenate((dis_list, distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
        
    auc = auc_value(label_list, dis_list)
    
    return auc