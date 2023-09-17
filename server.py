# server对所有参数求加权平均值，返回给client
# client在server传来的模型基础上训练
import torch
import numpy as np
import random
from sklearn.cluster import KMeans

from utils.trans import *
from utils.Logger import *
from utils.Params import *
from oldfile.kmeans import *

from model.VAE import *
from model.WAE import *
from model.VQVAE import *
from model.model import *

from client import *

# 忽略指定warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=2")

def split_mean(split):
    sorted_split = np.sort(split)
    remove_count = int(0.2 * split.shape[0])
    trimmed_split = sorted_split[remove_count:-remove_count]
    mean_split = np.mean(trimmed_split)
    return mean_split

def serFedAvg(testset, param_list, args):
    
    globModel = choose_model(param_list, args.model, args.dataset)
    
    if args.isclt == 'Yes':
        cltModel = [choose_model(param_list, args.model, args.dataset) for _ in range(args.clt_num) ]
    
    # download pre-trained model
    if args.isload == 'Yes':
        globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    auc_rec = []
    for round in range(args.comm_round):
        serlogger.info(f'-----------round {round}--------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        # client自己用自己的数据集单独训练
        if args.isclt == 'Yes':
            SelfTrain(testset, cltModel, sel_clt_list, args, round)

        # local training and aggregation with above method
        if isKmeans and args.model == 'VQVAE':
            Param_list, size_list, split_list, embeddings, emb_weight = FedAvg(globModel, sel_clt_list, param_list, args, round)
            
            # kmeans聚类，k是embedding的数量
            # kmeans = KMeans(n_clusters=param_list[6],n_init='auto')
            # kmeans.fit(embeddings)
            # centers = kmeans.cluster_centers_
            centers,centers_weight,_ = my_kmeans(embeddings, emb_weight, param_list[6])
        else:
            Param_list, size_list, split_list = FedAvg(globModel, sel_clt_list, param_list, args, round)

        # aggregation
        global_param = None
        total_size = sum(size_list)
        for i in range(len(Param_list)):
            if global_param is None:
                global_param = Param_list[i] * size_list[i] / total_size
            else:
                global_param += Param_list[i] * size_list[i] / total_size
        globModel = array2model(globModel, global_param)

        # 把kmeans聚类中心赋值给聚合后的model
        if isKmeans and args.model == 'VQVAE':
            with torch.no_grad():
                globModel.vq_layer.embedding.weight.copy_(torch.tensor(centers, dtype=float).to(device))
        
        # 记录auc值
        dis, label, normal_loss, loss = choose_test(globModel, testset, args.model)
        # print(split_list)
        split = split_mean(split_list)
        # print(split)
        # split = find_split(globModel, trainset, model_name=args.model, split_rate=args.norm_rate)
        # auc_cur = auc_value(label, dis)
        auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
        auc_rec.append(auc_cur)
        serlogger.info(f'R{round} !FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
        print(f'R{round} !FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

        # save globmodel
        if (1+round)%1==0:
            torch.save(globModel.state_dict(),f'./checkpoint/FedAvg-{args.model}-{round+1}.pth')

        if args.isFT=='Yes':
            # 重置model参数开始训练
            if args.isRetrain=='Yes':
                globModel = choose_model(param_list, args.model, args.dataset)
            
            clt_model_list = []
            for i,param in enumerate(Param_list):
                model = choose_model(param_list, args.model, args.dataset)
                clt_model_list.append(array2model(model, param))
            if args.model == 'VQVAE':
                VQVAE_FineTune(globModel, clt_model_list, centers_weight, args, round)
            else:
                FineTune(globModel, clt_model_list, args, round)
        
            # 记录auc值
            dis, label, normal_loss, loss = choose_test(globModel, testset, args.model)
            split = split_mean(split_list)
            # split = find_split(globModel, trainset, model_name=args.model, split_rate=args.norm_rate)
            # auc_cur = auc_value(label, dis)
            auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
            auc_rec.append(auc_cur)
            serlogger.info(f'R{round} FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
            print(f'R{round} FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

            # save globmodel
            if (1+round)%1==0:
                torch.save(globModel.state_dict(),f'./checkpoint/FedAvg-FT-{args.model}-{round+1}.pth')

    return auc_rec

def serFedProx(testset, param_list, args):
    
    globModel = choose_model(param_list, args.model, args.dataset)
    
    if args.isclt == 'Yes':
        cltModel = [choose_model(param_list, args.model, args.dataset) for _ in range(args.clt_num) ]
    
    # download pre-trained model
    if args.isload == 'Yes':
        globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    auc_rec = []
    for round in range(args.comm_round):
        serlogger.info(f'------------round {round}-------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        # client自己用自己的数据集单独训练
        if args.isclt == 'Yes':
            SelfTrain(testset, cltModel, sel_clt_list, args, round)

        # local training and aggregation with above method
        if isKmeans and args.model == 'VQVAE':
            Param_list, size_list, split_list, embeddings, emb_weight = FedProx(globModel, sel_clt_list, param_list, args, round)
            
            # kmeans聚类，k是embedding的数量
            centers,centers_weight,_ = my_kmeans(embeddings, emb_weight, param_list[6])
        else:
            Param_list, size_list, split_list = FedProx(globModel, sel_clt_list, param_list, args, round)

        # aggregation
        global_param = None
        total_size = sum(size_list)
        for i in range(len(Param_list)):
            if global_param is None:
                global_param = Param_list[i] * size_list[i] / total_size
            else:
                global_param += Param_list[i] * size_list[i] / total_size
        globModel = array2model(globModel, global_param)

        # 把kmeans聚类中心赋值给聚合后的model
        if isKmeans and args.model == 'VQVAE':
            with torch.no_grad():
                globModel.vq_layer.embedding.weight.copy_(torch.tensor(centers, dtype=float).to(device))
        
        # 记录auc值
        dis, label, normal_loss, loss = choose_test(globModel, testset, args.model)
        split = split_mean(split_list)
        # split = find_split(globModel, trainset, model_name=args.model, split_rate=args.norm_rate)
        # auc_cur = auc_value(label, dis)
        auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
        auc_rec.append(auc_cur)
        serlogger.info(f'R{round} !FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
        print(f'R{round} !FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

        # save globmodel
        if (1+round)%1==0:
            torch.save(globModel.state_dict(),f'./checkpoint/FedProx-{args.model}-{round+1}.pth')

        if args.isFT=='Yes':
            clt_model_list = []
            for i,param in enumerate(Param_list):
                model = choose_model(param_list, args.model, args.dataset)
                clt_model_list.append(array2model(model, param))
            if args.model == 'VQVAE':
                VQVAE_FineTune(globModel, clt_model_list, centers_weight, args, round)
            else:
                FineTune(globModel, clt_model_list, args, round)
        
            # 记录auc值
            dis, label, normal_loss, loss = choose_test(globModel, testset, args.model)
            split = split_mean(split_list)
            # split = find_split(globModel, trainset, model_name=args.model, split_rate=args.norm_rate)
            # auc_cur = auc_value(label, dis)
            auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
            auc_rec.append(auc_cur)
            serlogger.info(f'R{round} FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
            print(f'R{round} FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

            # save globmodel
            if (1+round)%1==0:
                torch.save(globModel.state_dict(),f'./checkpoint/FedProx-FT-{args.model}-{round+1}.pth')

    return auc_rec

def serFedDyn(testset, param_list, args):
    
    globModel = choose_model(param_list, args.model, args.dataset)
    
    if args.isclt == 'Yes':
        cltModel = [choose_model(param_list, args.model, args.dataset) for _ in range(args.clt_num) ]
    
    # download pre-trained model
    if args.isload == 'Yes':
        globModel.load_state_dict(torch.load(args.ckpt))

    # fed learning
    auc_rec = []
    for round in range(args.comm_round):
        serlogger.info(f'------------round {round}-------------')

        # select clients
        if args.act_rate < 1.0:
            sel_clt_list = random.sample(range(args.clt_num),int(args.act_rate*args.clt_num))
        else:
            sel_clt_list = np.arange(args.clt_num)

        # client自己用自己的数据集单独训练
        if args.isclt == 'Yes':
            SelfTrain(testset, cltModel, sel_clt_list, args, round)

        # local training and aggregation with above method
        if isKmeans and args.model == 'VQVAE':
            Param_list, size_list, split_list, param_arr, embeddings, emb_weight = FedDyn(globModel, sel_clt_list, param_list, args, round)
            
            # kmeans聚类，k是embedding的数量
            centers,centers_weight,_ = my_kmeans(embeddings, emb_weight, param_list[6])
        else:
            Param_list, size_list, split_list, param_arr = FedDyn(globModel, sel_clt_list, param_list, args, round)

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

        # 把kmeans聚类中心赋值给聚合后的model
        if isKmeans and args.model == 'VQVAE':
            # print('before kmeans: ',globModel.state_dict()['vq_layer.embedding.weight'])
            with torch.no_grad():
                globModel.vq_layer.embedding.weight.copy_(torch.tensor(centers, dtype=float).to(device))
            # print('after kmeans: ',globModel.state_dict()['vq_layer.embedding.weight'])
        
        # 记录auc值
        dis, label, normal_loss, loss = choose_test(globModel, testset, args.model)
        split = split_mean(split_list)
        # split = find_split(globModel, trainset, model_name=args.model, split_rate=args.norm_rate)
        # auc_cur = auc_value(label, dis)
        auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
        auc_rec.append(auc_cur)
        serlogger.info(f'R{round} !FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
        print(f'R{round} !FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

        # save globmodel
        if (1+round)%1==0:
            torch.save(globModel.state_dict(),f'./checkpoint/FedDyn-{args.model}-{round+1}.pth')

        if args.isFT=='Yes':
            clt_model_list = []
            for i,param in enumerate(Param_list):
                model = choose_model(param_list, args.model, args.dataset)
                clt_model_list.append(array2model(model, param))
            if args.model == 'VQVAE':
                VQVAE_FineTune(globModel, clt_model_list, centers_weight, args, round)
            else:
                FineTune(globModel, clt_model_list, args, round)
        
            # 记录auc值
            dis, label, normal_loss, loss = choose_test(globModel, testset, args.model)
            split = split_mean(split_list)
            # split = find_split(globModel, trainset, model_name=args.model, split_rate=args.norm_rate)
            # auc_cur = auc_value(label, dis)
            auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, split)
            auc_rec.append(auc_cur)
            serlogger.info(f'R{round} FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
            print(f'R{round} FT | auc:{auc_cur:.3f} / f1:{f1_cur:.3f} / recall:{recall_cur:.3f} / precision:{precision_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

            # save globmodel
            if (1+round)%1==0:
                torch.save(globModel.state_dict(),f'./checkpoint/FedDyn-FT-{args.model}-{round+1}.pth')

    return auc_rec