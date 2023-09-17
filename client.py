import torch
import numpy as np
import copy
import random

from utils.trans import *
from utils.Logger import *
from utils.Params import *
from run.train import *
from model.VAE import *
from model.VQVAE import *
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

    random.shuffle(trainset)
    
    # dirichlet distribution or iid
    if args.distri == 'non-iid':
        cls_distri = np.random.dirichlet([args.alpha]*args.clt_num)
        
        # 限制最小不可以是0
        min_value = 0.0001
        cls_distri = np.maximum(min_value, cls_distri)
        cls_distri /= sum(cls_distri)
        
        # print(cls_distri)
        distri_cumsum = np.cumsum(cls_distri)
        distri = (distri_cumsum*len(trainset)).astype(int)
    else:
        cls_distri = np.ones(args.clt_num)
        distri_cumsum = np.cumsum(cls_distri)
        distri = (distri_cumsum/args.clt_num*len(trainset)).astype(int)
    # cltlogger.info(f'data size: {distri}')

    data_split = [ trainset[(distri[i-1] if i>0 else 0):(distri[i] if i<args.clt_num else len(trainset))] for i in range(args.clt_num)]
    
    # init model
    model = choose_model(param_list, args.model, args.dataset)
    for i in range(args.clt_num):
        clt_pool[i] = Client(model, 
                            args.lr,  
                            args.epochs, 
                            args.batch_size)
        
        clt_pool[i].dataset = data_split[i]
        clt_pool[i].dataset_size = len(clt_pool[i].dataset)
        clt_pool[i].batch_num = math.ceil(clt_pool[i].dataset_size / args.batch_size)
        cltlogger.info(f'{i} dataset size: {clt_pool[i].dataset_size}')

def SelfTrain(testset, clt_model_list, clts_list, args, round):
    global clt_pool

    for i in clts_list:
        cltlogger.info(f'---------------{i} selftrain----------------')
        optimizer = torch.optim.Adam(clt_model_list[i].parameters(), lr=clt_pool[i].lr)
        # choose_train(clt_model_list[i], optimizer, clt_pool[i].dataset, testset, cltlogger, clt_pool[i].epochs, clt_pool[i].batch_size, round, args.model, isTest=True)

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

                if args.model == 'VAE' or args.model == 'BetaVAE':
                    recon, mu, logvar, _ = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, mu, logvar)
                elif args.model == 'VQVAE':
                    _, _, _, loss, recon_loss, reg_loss = clt_model_list[i](inputs)
                elif args.model == 'WAE':
                    recon, z = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, z)
                else:
                    raise Warning('no this model.')
                
                loss_rec += loss.item()
                recon_loss_rec += recon_loss.item()
                reg_loss_rec += reg_loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=clt_model_list[i].parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()
            
            loss_rec = loss_rec/clt_pool[i].batch_num
            recon_loss_rec = recon_loss_rec/clt_pool[i].batch_num
            reg_loss_rec = reg_loss_rec/clt_pool[i].batch_num
            cltlogger.info(f'R{round}-E{epoch} / loss:{loss_rec}')
        
        dis, label, norm_loss, loss = choose_test(clt_model_list[i], testset, args.model)
        split = find_split(clt_model_list[i], clt_pool[i].dataset, model_name=args.model, split_rate=args.norm_rate)
        # auc_cur = auc_value(label, dis)
        auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
        cltlogger.info(f'R{round} / auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{norm_loss:.3f} / loss:{loss:.3f}')
        print(f'R{round} / auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{auc_cur:.3f} / precision:{precision_cur:.3f} / norm_loss: {norm_loss:.3f} / loss:{loss:.3f}')

        torch.save(clt_model_list[i].state_dict(), f'./checkpoint/{args.model}-{round}.pth')

def FedAvg(globModel, clts_list, param_list, args, round):
    global clt_pool

    clt_model_list = [None for _ in range(args.clt_num)]
    split_list = []
    emb_p_list = []

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

                if args.model == 'VAE' or args.model == 'BetaVAE':
                    recon, mu, logvar, _ = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, mu, logvar)
                elif args.model == 'VQVAE':
                    _, _, _, loss, recon_loss, reg_loss = clt_model_list[i](inputs)
                elif args.model == 'WAE':
                    recon, z = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, z)
                else:
                    raise Warning('no this model.')

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

        split = find_split(clt_model_list[i], clt_pool[i].dataset, model_name=args.model, split_rate=args.norm_rate)
        split_list.append(split)

        if args.model == 'VQVAE':
            clt_model_list[i].eval()
            emb_p = np.zeros(param_list[6], dtype=np.float64)
            dataloader = generate_data(clt_pool[i].dataset, batch_size=clt_pool[i].batch_size)
            for _,data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(device)
                emb_labels = clt_model_list[i].get_label(inputs)
                for index in emb_labels:
                    emb_p[index]+=1
            emb_p_list.append(emb_p)
    
    clt_param_list = [model2array(clt_model_list[i]) for i in clts_list]
    clt_size_list = np.array([clt_pool[i].dataset_size for i in clts_list])
    if isKmeans and args.model == 'VQVAE':
        # 单独拿出embeddings，用kmeans聚合
        embeddings_list = np.concatenate([ (clt_model_list[i].state_dict()['vq_layer.embedding.weight']).detach().cpu().numpy().copy() for i in clts_list ],axis=0)
        emb_p_list = np.concatenate(emb_p_list, axis=0)
        return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(np.array(split_list, dtype=np.float64)), np.copy(embeddings_list), np.copy(emb_p_list)
    else:
        return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(np.array(split_list, dtype=np.float64))

def FedProx(globModel, clts_list, param_list, args, round):
    global clt_pool

    clt_model_list = [None for _ in range(args.clt_num)]
    split_list = []
    emb_p_list = []

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

                if args.model == 'VAE' or args.model == 'BetaVAE':
                    recon, mu, logvar, _ = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, mu, logvar)
                elif args.model == 'VQVAE':
                    _, _, _, loss, recon_loss, reg_loss = clt_model_list[i](inputs)
                elif args.model == 'WAE':
                    recon, z = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, z)
                else:
                    raise Warning('no this model.')

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

        split = find_split(clt_model_list[i], clt_pool[i].dataset, model_name=args.model, split_rate=args.norm_rate)
        split_list.append(split)

        if args.model == 'VQVAE':
            clt_model_list[i].eval()
            emb_p = np.zeros(param_list[6], dtype=np.float64)
            dataloader = generate_data(clt_pool[i].dataset, batch_size=clt_pool[i].batch_size)
            for _,data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(device)
                emb_labels = clt_model_list[i].get_label(inputs)
                for index in emb_labels:
                    emb_p[index]+=1
            emb_p_list.append(emb_p)
    
    clt_param_list = [model2array(clt_model_list[i]) for i in clts_list]
    clt_size_list = np.array([clt_pool[i].dataset_size for i in clts_list])
    if isKmeans and args.model == 'VQVAE':
        # 单独拿出embeddings，用kmeans聚合
        embeddings_list = np.concatenate([ (clt_model_list[i].state_dict()['vq_layer.embedding.weight']).detach().cpu().numpy().copy() for i in clts_list ],axis=0)
        emb_p_list = np.concatenate(emb_p_list, axis=0)
        return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(np.array(split_list, dtype=np.float64)), np.copy(embeddings_list), np.copy(emb_p_list)
    else:
        return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(np.array(split_list, dtype=np.float64))

def FedDyn(globModel, clts_list, param_list, args, round):
    global clt_pool

    clt_model_list = [None for _ in range(args.clt_num)]
    split_list = []
    emb_p_list = []

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

                if args.model == 'VAE' or args.model == 'BetaVAE':
                    recon, mu, logvar, _ = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, mu, logvar)
                elif args.model == 'VQVAE':
                    _, _, _, loss, recon_loss, reg_loss = clt_model_list[i](inputs)
                elif args.model == 'WAE':
                    recon, z = clt_model_list[i](inputs)
                    loss, recon_loss, reg_loss = clt_model_list[i].loss(inputs, recon, z)
                else:
                    raise Warning('no this model.')
                
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

        split = find_split(clt_model_list[i], clt_pool[i].dataset, model_name=args.model, split_rate=args.norm_rate)
        split_list.append(split)

        if args.model == 'VQVAE':
            clt_model_list[i].eval()
            emb_p = np.zeros(param_list[6], dtype=np.float64)
            dataloader = generate_data(clt_pool[i].dataset, batch_size=clt_pool[i].batch_size)
            for _,data in enumerate(dataloader):
                inputs, _ = data
                inputs = inputs.to(device)
                emb_labels = clt_model_list[i].get_label(inputs)
                for index in emb_labels:
                    emb_p[index]+=1
            emb_p_list.append(emb_p)

        clt_pool[i].param_arr += (model2array(clt_model_list[i])-pre_glob_model_param.cpu().numpy())
        
    clt_param_list = [model2array(clt_model_list[i]) for i in clts_list]
    clt_size_list = np.array([clt_pool[i].dataset_size for i in clts_list])
    clt_param_arr = [clt_pool[i].param_arr  for i in range(args.clt_num)]
    if isKmeans and args.model == 'VQVAE':
        # 单独拿出embeddings，用kmeans聚合
        embeddings_list = np.concatenate([ (clt_model_list[i].state_dict()['vq_layer.embedding.weight']).detach().cpu().numpy().copy() for i in clts_list ],axis=0)
        emb_p_list = np.concatenate(emb_p_list, axis=0)
        return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(np.array(split_list, dtype=np.float64)), np.copy(clt_param_arr), np.copy(embeddings_list), np.copy(emb_p_list)
    else:
        return np.copy(clt_param_list), np.copy(clt_size_list), np.copy(np.array(split_list, dtype=np.float64)), np.copy(clt_param_arr)