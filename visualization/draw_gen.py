import argparse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.LoadData import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from run.test import *
from run.train import *
from model.model import *

def generate_attack_data(model, samples_num):
    # generate samples
    gen_dataset = []
    samples = model.sample_attack(samples_num*10, device)
    for sample in samples:
        gen_dataset.append([sample,-2])
    
    return gen_dataset

def generate_normal_data(model, samples_num):
    # generate samples
    gen_dataset = []
    samples = model.sample(samples_num*100, device)
    for sample in samples:
        gen_dataset.append([sample,-1])
    
    return gen_dataset

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CICIDS17'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='BetaVAE', choices=['VAE','BetaVAE','VQVAE','WAE'],
                        help='model name (default: VAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: SMALL)')
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)

    # 配置模型及优化器
    model = choose_model(param_list, args.model, args.dataset)
    model.load_state_dict(torch.load('./checkpoint/WAE-12.pth'))

    # 加载数据集
    testset = load_test_csv(f'./dataset/{args.dataset}/testset')
    testset0 = [ x for x in testset if x[1]==0]
    testset1 = [ x for x in testset if x[1]>0]
    testset0 = testset0[:len(testset1)]
    testset = testset0 + testset1
    print(len(testset))
    testset2 = generate_normal_data(model, 300)
    testset3 = generate_attack_data(model, 300)
    print(len(testset2),len(testset3))
    testset = testset+testset2+testset3

    # 求重构误差
    z, dis, cos, label, _,_ = choose_test_all(model, testset, args.model)
    label = label.astype(np.int32)

    z0 = z[label==0]
    z1 = z[label>0]
    z2 = z[label==-1]
    z3 = z[label==-2]
    dis0 = dis[label==0]
    dis1 = dis[label>0]
    dis2 = dis[label==-1]
    dis3 = dis[label==-2]
    cos0 = cos[label==0]
    cos1 = cos[label>0]
    cos2 = cos[label==-1]
    cos3 = cos[label==-2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for i in range(max_label):
    num=300
    ax.scatter(z0[:num,0], z0[:num,1], z0[:num,2], c='black')
    ax.scatter(z1[:num,0], z1[:num,1], z1[:num,2], c='red')
    ax.scatter(z2[:num,0], z2[:num,1], z2[:num,2], c='blue')
    ax.scatter(z3[:num,0], z3[:num,1], z3[:num,2], c='green')
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')
    ax.set_zlabel('z2')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dis0[:num], cos0[:num], c='black')
    ax.scatter(dis1[:num], cos1[:num], c='red')
    ax.scatter(dis2[:num], cos2[:num], c='blue')
    ax.scatter(dis3[:num], cos3[:num], c='green')
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')

    plt.show()

    # 计算ROC曲线和AUC面积
    row_mask1 = (label!=0) & (label!=-1)
    row_mask0 = (label==0) | (label==-1)
    label[row_mask1] = 1
    label[row_mask0] = 0
    normalized_dis = dis/np.max(dis)
    fpr, tpr, thresholds = roc_curve(label, normalized_dis)
    auc_score = roc_auc_score(label, normalized_dis)
    print(auc_score)
    
if __name__ == '__main__':
    main()