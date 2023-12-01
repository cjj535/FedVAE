import argparse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA

from utils.LoadData import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from run.test import *
from run.train import *
from model.model import *




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018-Dos'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='WAE', choices=['VAE','BetaVAE','VQVAE','WAE','weightWAE'],
                        help='model name (default: WAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    parser.add_argument('--rate', type=str, default='1%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='anomaly rate (default: 0%)')
    parser.add_argument('--isall', type=str, default='Yes', choices=['Yes','No'],
                        help='anomaly rate (default: 0%)')
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)
    
    if args.isall=='Yes':
        is_all = True
    else:
        is_all = False
    # 加载数据集
    if is_all:
        trainset = load_csv(f'./dataset/{args.dataset}/trainset-{args.rate}')
        testset = load_csv(f'./dataset/{args.dataset}/testset-{args.rate}')
        dataset0 = [ x for x in trainset if x[1]==0]
        dataset1 = [ x for x in testset if x[1]>0]
        dataset = random.sample(dataset0,50000)+random.sample(dataset1,50000)
    else:
        trainset = load_csv(f'./dataset/{args.dataset}/trainset-{args.rate}')
        # testset = load_csv(f'./dataset/{args.dataset}/testset-{args.rate}')
        dataset0 = [ x for x in trainset if x[1]==0]
        # dataset1 = [ x for x in testset if x[1]>0]
        dataset = random.sample(dataset0,20000)#+random.sample(dataset1,10000)
    
    model = choose_model(param_list, args.model, args.dataset)
    model.load_state_dict(torch.load(f'./checkpoint/{args.dataset}-{args.rate}-{args.model}-{args.model_size}-19.pth'))

    # dataloader = generate_data(dataset, batch_size=2048, is_shuffle=False)
    # cum_dis = np.zeros((69))
    # for i, data in enumerate(dataloader):
    #     inputs, labels = data
    #     inputs = inputs.to(device)

    #     recon, _ = model(inputs)
    #     recon = recon.detach().cpu().numpy().copy()
    #     inputs = inputs.detach().cpu().numpy().copy()
    #     dis = np.sum((recon-inputs)**2,axis=0)
    #     cum_dis += dis
    
    # log10_cum_dis = np.log10(cum_dis)
    
    # # 获取排序后的索引
    # sorted_indices = np.argsort(log10_cum_dis)

    # # 输出最大的五个值及对应的原下标
    # top_five_values = log10_cum_dis[sorted_indices[-5:]][::-1]
    # top_five_indices = sorted_indices[-5:][::-1]

    # print("最大的五个值：", top_five_values)
    # print("对应的原下标：", top_five_indices)

    # # # 绘制ROC曲线
    # indices = np.arange(len(log10_cum_dis))

    # # 画方形
    # plt.bar(indices, log10_cum_dis, color='blue', edgecolor='black')
    # if is_all:
    #     plt.savefig(f"feature importance all {args.dataset}.png")
    # else:
    #     plt.savefig(f"feature importance norm {args.dataset}.png")
    
    dataloader = generate_data(dataset0, batch_size=2048, is_shuffle=False)
    cum_dis_norm = np.zeros((69))
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, _ = model(inputs)
        recon = recon.detach().cpu().numpy().copy()
        inputs = inputs.detach().cpu().numpy().copy()
        dis = np.sum((recon-inputs)**2,axis=0)
        cum_dis_norm += dis
    
    log10_cum_dis_norm = np.log10(cum_dis_norm)
    
    dataloader = generate_data(dataset1, batch_size=2048, is_shuffle=False)
    cum_dis_attk = np.zeros((69))
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, _ = model(inputs)
        recon = recon.detach().cpu().numpy().copy()
        inputs = inputs.detach().cpu().numpy().copy()
        dis = np.sum((recon-inputs)**2,axis=0)
        cum_dis_attk += dis
    
    log10_cum_dis_attk = np.log10(cum_dis_attk)
    
    
    dif_cum_dis = cum_dis_norm - cum_dis_attk
    dif_cum_dis[dif_cum_dis<=0] = 1e-7
    dif_cum_dis = np.log10(dif_cum_dis)
    
    # 获取排序后的索引
    sorted_indices = np.argsort(dif_cum_dis)

    # 输出最大的五个值及对应的原下标
    top_five_values = dif_cum_dis[sorted_indices[-10:]][::-1]
    top_five_indices = sorted_indices[-10:][::-1]

    print("最大的五个值：", top_five_values)
    print("对应的原下标：", top_five_indices)

    # 画方形
    indices = np.arange(len(log10_cum_dis_norm))
    plt.subplot(3,1,1)
    plt.bar(indices, log10_cum_dis_norm, color='blue', edgecolor='black')
    plt.subplot(3,1,2)
    plt.bar(indices, log10_cum_dis_attk, color='blue', edgecolor='black')
    plt.subplot(3,1,3)
    plt.bar(indices, dif_cum_dis, color='blue', edgecolor='black')
    plt.savefig(f"feature importance {args.dataset}.png")
    
if __name__ == '__main__':
    main()