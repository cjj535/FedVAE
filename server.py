# server对所有参数求加权平均值，返回给client
# client在server传来的模型基础上训练
import torch
import numpy as np
import random
from sklearn.cluster import KMeans

from utils.trans import *
from utils.Logger import *
from utils.Params import *

from model.VAE import *
from model.WAE import *
from model.model import *

from run.loss import *
from client import *

def serFedAvg(trainset, testset, logger, param_list, args):
    metrics = []
    
    globModel = choose_model(param_list, args.model, args.dataset)

    # download pre-trained model
    # if args.isload == 'Yes':
    #     globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    for round in range(args.comm_round):
        logger.info(f'-----------round {round}--------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        # 交给client去训练
        Param_list, size_list = FedAvg(globModel, sel_clt_list, param_list, args, round)

        # aggregation
        global_param = None
        total_size = sum(size_list)
        for i in range(len(Param_list)):
            if global_param is None:
                global_param = Param_list[i] * size_list[i] / total_size
            else:
                global_param += Param_list[i] * size_list[i] / total_size
        globModel = array2model(globModel, global_param)

        # 测试并记录
        norm_loss, attk_loss = train_loss(globModel, trainset)
        auc_cur = test_auc(globModel, testset)
        
        metrics.append([auc_cur,norm_loss,attk_loss])
        logger.info(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')
        print(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')

        # save globmodel
        if (1+round)%5==0 or round<5:
            torch.save(globModel.state_dict(),f'./FL_checkpoint/{args.dataset}-{args.fl_method}-{args.alpha}-{round}.pth')

    return metrics

def serFedProx(trainset, testset, logger, param_list, args):
    metrics = []
    
    globModel = choose_model(param_list, args.model, args.dataset)
    
    # download pre-trained model
    # if args.isload == 'Yes':
    #     globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    for round in range(args.comm_round):
        logger.info(f'------------round {round}-------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        Param_list, size_list = FedProx(globModel, sel_clt_list, param_list, args, round)

        # aggregation
        global_param = None
        total_size = sum(size_list)
        for i in range(len(Param_list)):
            if global_param is None:
                global_param = Param_list[i] * size_list[i] / total_size
            else:
                global_param += Param_list[i] * size_list[i] / total_size
        globModel = array2model(globModel, global_param)

        # 测试并记录
        norm_loss, attk_loss = train_loss(globModel, trainset)
        auc_cur = test_auc(globModel, testset)
        
        metrics.append([auc_cur,norm_loss,attk_loss])
        logger.info(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')
        print(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')

        # save globmodel
        if (1+round)%5==0 or round<5:
            torch.save(globModel.state_dict(),f'./FL_checkpoint/{args.dataset}-{args.fl_method}-{args.alpha}-{round}.pth')

    return metrics

def serFedDyn(trainset, testset, logger, param_list, args):
    metrics = []
    
    globModel = choose_model(param_list, args.model, args.dataset)
    
    # download pre-trained model
    # if args.isload == 'Yes':
    #     globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    for round in range(args.comm_round):
        logger.info(f'------------round {round}-------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        Param_list, size_list, param_arr = FedDyn(globModel, sel_clt_list, param_list, args, round)

        # aggregation
        global_param = None
        total_size = sum(size_list)
        for i in range(len(Param_list)):
            if global_param is None:
                global_param = Param_list[i] * size_list[i] / total_size
            else:
                global_param += Param_list[i] * size_list[i] / total_size
        if isDyn:
            # h = mean of all
            clt_param_mean = param_arr[0]
            for i in range(1, args.clt_num):
                clt_param_mean += param_arr[i]
            clt_param_mean /= args.clt_num
            # add param mean
            global_param += clt_param_mean
        globModel = array2model(globModel, global_param)

        # 测试并记录
        norm_loss, attk_loss = train_loss(globModel, trainset)
        auc_cur = test_auc(globModel, testset)
        
        metrics.append([auc_cur,norm_loss,attk_loss])
        logger.info(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')
        print(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')

        # save globmodel
        if (1+round)%5==0 or round<5:
            torch.save(globModel.state_dict(),f'./FL_checkpoint/{args.dataset}-{args.fl_method}-{args.alpha}-{round}.pth')

    return metrics

def serFedFT(trainset, testset, logger, param_list, args):
    metrics_before = []
    metrics_after = []
    
    globModel = choose_model(param_list, args.model, args.dataset)

    # download pre-trained model
    # if args.isload == 'Yes':
    #     globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    for round in range(args.comm_round):
        logger.info(f'-----------round {round}--------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        Param_list, size_list = FedAvg(globModel, sel_clt_list, param_list, args, round)

        # aggregation
        global_param = None
        total_size = sum(size_list)
        for i in range(len(Param_list)):
            if global_param is None:
                global_param = Param_list[i] * size_list[i] / total_size
            else:
                global_param += Param_list[i] * size_list[i] / total_size
        globModel = array2model(globModel, global_param)

        # 测试并记录
        norm_loss, attk_loss = train_loss(globModel, trainset)
        auc_cur = test_auc(globModel, testset)
        
        metrics_before.append([auc_cur,norm_loss,attk_loss])
        logger.info(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')
        print(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')

        # save globmodel
        if (1+round)%5==0 or round<5:
            torch.save(globModel.state_dict(),f'./FL_checkpoint/{args.dataset}-{args.fl_method}-{args.alpha}-{round}.pth')


        # FT using knowledge distillation
        clt_model_list = []
        for i,param in enumerate(Param_list):
            model = choose_model(param_list, args.model, args.dataset)
            clt_model_list.append(array2model(model, param))

        FineTune(globModel, clt_model_list, size_list, args)
    
        # 测试并记录
        norm_loss, attk_loss = train_loss(globModel, trainset)
        auc_cur = test_auc(globModel, testset)
        
        metrics_after.append([auc_cur,norm_loss,attk_loss])
        logger.info(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')
        print(f'R{round} auc:{auc_cur:.3f} / norm:{norm_loss:.3f} / attk:{attk_loss:.3f}')

        # save globmodel
        if (1+round)%5==0 or round<5:
            torch.save(globModel.state_dict(),f'./FL_checkpoint/FT-{args.dataset}-{args.fl_method}-{args.alpha}-{round}.pth')

    return metrics_before, metrics_after


def FineTune(model, clt_model_list, size_list, args):
    # parameters
    lambda_weight = 1
    epochs = 5
    total_pseudo_samples_num = 2048
    total_size = np.sum(size_list)
    
    # generate samples
    trainset = []
    for i in range(len(clt_model_list)):
        pseodu_samples_num = int(total_pseudo_samples_num*(size_list[i]/total_size))
        pseudo_samples, recon_pseudo_samples = clt_model_list[i].sample_pair(pseodu_samples_num, device)
        
        for sample, recon in zip(pseudo_samples,recon_pseudo_samples):
            trainset.append([sample,recon])
    
    # normal fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        
        dataloader = generate_pseudo_data(trainset, batch_size=2048, is_shuffle=True)
        for _,data in enumerate(dataloader):
            inputs, t_recons = data
            inputs = inputs.to(device)
            t_recons = t_recons.to(device)
            
            s_recons, z = model(inputs)
            
            optimizer.zero_grad()

            loss_md = F.mse_loss(s_recons, t_recons, reduction='none').sum(dim=-1).mean(dim=0)
            loss_rec,_,_,_ = model.loss(inputs, s_recons, z)
            loss = loss_md + lambda_weight * loss_rec

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

            running_loss += loss.item()
        