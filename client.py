import torch
import numpy as np
import copy
import random

from utils.trans import *
from utils.Logger import *
from utils.Params import *
from run.train import *
from model.VAE import *
from model.WAE import *
from model.model import *

clt_pool = []

class Client:
    def __init__(self, model, lr=1e-3, epochs=2, batch_size=128):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_size = 0
        self.dataset = None
        self.batch_num = 0

        self.param_arr = np.zeros(model2array(model).shape[0], dtype=np.float64)

def InitPool(param_list, trainset, args):
    global clt_pool
    clt_pool=[None for _ in range(args.clt_num)]

    num_classes = 2
    
    train_list = [data for data in trainset]
    
    random.shuffle(train_list)
    
    classed_data = [[], []]
    for data in train_list:
        _, label = data
        classed_data[label].append(data)
    
    # dirichlet distribution or iid
    if args.distri == 'non-iid':
        cls_distri = np.random.dirichlet([args.alpha]*args.clt_num, num_classes)
        
        # 限制最小不可以是0
        min_value = 0.0001
        cls_distri = np.maximum(min_value, cls_distri)
        divisors = np.sum(cls_distri, axis=1)
        cls_distri = cls_distri / divisors[:, np.newaxis]
        
        distri_cumsum = np.cumsum(cls_distri, axis=1)
        distri = [ (distri_cumsum[y]*len(classed_data[y])).astype(int) for y in range(num_classes) ]
    else:
        cls_distri = np.ones((num_classes, args.clt_num))
        distri_cumsum = np.cumsum(cls_distri, axis=1)
        distri = [ (distri_cumsum[y]/args.clt_num*len(classed_data[y])).astype(int) for y in range(num_classes) ]

    data_split = [ [ classed_data[y][(distri[y][i-1] if i>0 else 0):
                            (distri[y][i] if i<args.clt_num else len(classed_data[y]))] 
                            for y in range(num_classes) ] for i in range(args.clt_num)]
    
    # init model
    model = choose_model(param_list, args.model, args.dataset)
    for i in range(args.clt_num):
        clt_pool[i] = Client(model, 
                            args.lr,  
                            args.epochs, 
                            args.batch_size)
        
        norm_num = len(data_split[i][0])
        attk_num = len(data_split[i][1])
        cltlogger.info(f'{i} dataset size: norm {norm_num}, attk {attk_num}')
        clt_pool[i].dataset = data_split[i][0]
        clt_pool[i].dataset.extend(data_split[i][1])
        
        clt_pool[i].dataset_size = len(clt_pool[i].dataset)
        clt_pool[i].batch_num = math.ceil(clt_pool[i].dataset_size / args.batch_size)


def FedAvg(globModel, clts_list, param_list, args, round):
    global clt_pool

    clt_model_list = [None for _ in range(args.clt_num)]

    for i in clts_list:
        cltlogger.info(f'----------------{i} cotrain-----------------')

        # 复制global model的参数
        clt_model_list[i] = choose_model(param_list, args.model, args.dataset)
        clt_model_list[i].load_state_dict(copy.deepcopy(dict(globModel.state_dict())), strict=True)
        optimizer = torch.optim.Adam(clt_model_list[i].parameters(), lr=clt_pool[i].lr)
        
        clt_model_list[i].train()
        for epoch in range(clt_pool[i].epochs):
            loss_rec = 0
            recon_loss_rec = 0
            reg_loss_rec = 0
            dataloader = generate_data(clt_pool[i].dataset, batch_size=clt_pool[i].batch_size, is_shuffle=True)
            for _,data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(device)

                optimizer.zero_grad()
                
                # if args.model == 'VAE':
                #     recon, mu, logvar, _ = clt_model_list[i](inputs)
                #     loss, recon_loss, reg_loss, _ = clt_model_list[i].loss(inputs, recon, mu, logvar)
                # else:
                recon, z = clt_model_list[i](inputs)
                loss, recon_loss, reg_loss, _ = clt_model_list[i].loss(inputs, recon, z)

                loss_rec += loss.item()
                recon_loss_rec += recon_loss.item()
                reg_loss_rec += reg_loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=clt_model_list[i].parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()

            loss_rec = loss_rec/clt_pool[i].batch_num
            recon_loss_rec = recon_loss_rec/clt_pool[i].batch_num
            reg_loss_rec = reg_loss_rec/clt_pool[i].batch_num
            cltlogger.info(f'R{round}-E{epoch} / Loss: {loss_rec:.3f} {recon_loss_rec:.3f} {reg_loss_rec:.3f}')
    
    clt_param_list = [model2array(clt_model_list[i]) for i in clts_list]
    clt_size_list = np.array([clt_pool[i].dataset_size for i in clts_list])
    return np.copy(clt_param_list), np.copy(clt_size_list)

def FedProx(globModel, clts_list, param_list, args, round):
    global clt_pool

    clt_model_list = [None for _ in range(args.clt_num)]

    pre_glob_model_param = torch.cat([var.clone().detach().view(-1) for var in globModel.parameters()])

    for i in clts_list:
        cltlogger.info(f'----------------{i} cotrain-----------------')

        # 复制global model的参数
        clt_model_list[i] = choose_model(param_list, args.model, args.dataset)
        clt_model_list[i].load_state_dict(copy.deepcopy(dict(globModel.state_dict())), strict=True)
        optimizer = torch.optim.Adam(clt_model_list[i].parameters(), lr=clt_pool[i].lr)
        
        clt_model_list[i].train()
        for epoch in range(clt_pool[i].epochs):
            loss_rec = 0
            recon_loss_rec = 0
            reg_loss_rec = 0
            dataloader = generate_data(clt_pool[i].dataset, batch_size=clt_pool[i].batch_size, is_shuffle=True)
            for _,data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(device)

                optimizer.zero_grad()

                # if args.model == 'VAE':
                #     recon, mu, logvar, _ = clt_model_list[i](inputs)
                #     loss, recon_loss, reg_loss, _ = clt_model_list[i].loss(inputs, recon, mu, logvar)
                # else:
                recon, z = clt_model_list[i](inputs)
                loss, recon_loss, reg_loss, _ = clt_model_list[i].loss(inputs, recon, z)

                loss_rec += loss.item()
                recon_loss_rec += recon_loss.item()
                reg_loss_rec += reg_loss.item()

                # fedprox 正则项
                pre_local_model_param = torch.cat([var.view(-1) for var in clt_model_list[i].parameters()])
                loss += (prox_factor/2 * torch.sum(torch.pow(pre_local_model_param - pre_glob_model_param, 2)))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=clt_model_list[i].parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()
        
            loss_rec = loss_rec/clt_pool[i].batch_num
            recon_loss_rec = recon_loss_rec/clt_pool[i].batch_num
            reg_loss_rec = reg_loss_rec/clt_pool[i].batch_num
            cltlogger.info(f'R{round}-E{epoch} / Loss: {loss_rec:.3f} {recon_loss_rec:.3f} {reg_loss_rec:.3f}')
    
    clt_param_list = [model2array(clt_model_list[i]) for i in clts_list]
    clt_size_list = np.array([clt_pool[i].dataset_size for i in clts_list])
    return np.copy(clt_param_list), np.copy(clt_size_list)

def FedDyn(globModel, clts_list, param_list, args, round):
    global clt_pool

    clt_model_list = [None for _ in range(args.clt_num)]

    pre_glob_model_param = torch.cat([var.clone().detach().view(-1) for var in globModel.parameters()])

    for i in clts_list:
        cltlogger.info(f'----------------{i} cotrain-----------------')

        # 复制global model的参数
        clt_model_list[i] = choose_model(param_list, args.model, args.dataset)
        clt_model_list[i].load_state_dict(copy.deepcopy(dict(globModel.state_dict())), strict=True)
        optimizer = torch.optim.Adam(clt_model_list[i].parameters(), lr=clt_pool[i].lr)
        
        clt_model_list[i].train()
        for epoch in range(clt_pool[i].epochs):
            loss_rec = 0
            recon_loss_rec = 0
            reg_loss_rec = 0
            dataloader = generate_data(clt_pool[i].dataset, batch_size=clt_pool[i].batch_size, is_shuffle=True)
            for _,data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(device)

                optimizer.zero_grad()

                # if args.model == 'VAE':
                #     recon, mu, logvar, _ = clt_model_list[i](inputs)
                #     loss, recon_loss, reg_loss, _ = clt_model_list[i].loss(inputs, recon, mu, logvar)
                # else:
                recon, z = clt_model_list[i](inputs)
                loss, recon_loss, reg_loss, _ = clt_model_list[i].loss(inputs, recon, z)
                
                loss_rec += loss.item()
                recon_loss_rec += recon_loss.item()
                reg_loss_rec += reg_loss.item()

                # fedprox 正则项
                pre_local_model_param = torch.cat([var.view(-1) for var in clt_model_list[i].parameters()])
                loss += (feddyn_alpha * (torch.sum(pre_local_model_param * torch.from_numpy(clt_pool[i].param_arr).to(device)) +
                        0.5*torch.sum(torch.pow(pre_local_model_param - pre_glob_model_param, 2))))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=clt_model_list[i].parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()
            
            loss_rec = loss_rec/clt_pool[i].batch_num
            recon_loss_rec = recon_loss_rec/clt_pool[i].batch_num
            reg_loss_rec = reg_loss_rec/clt_pool[i].batch_num
            cltlogger.info(f'R{round}-E{epoch} / Loss: {loss_rec:.3f} {recon_loss_rec:.3f} {reg_loss_rec:.3f}')

        clt_pool[i].param_arr += (model2array(clt_model_list[i])-pre_glob_model_param.cpu().numpy())
        
    clt_param_list = [model2array(clt_model_list[i]) for i in clts_list]
    clt_size_list = np.array([clt_pool[i].dataset_size for i in clts_list])
    clt_param_arr = [clt_pool[i].param_arr  for i in range(args.clt_num)]
    
    return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(clt_param_arr)