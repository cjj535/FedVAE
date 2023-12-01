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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SVDD')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CICIDS17'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='DSVDD', choices=['DSVDD','VAESVDD'],
                        help='model name (default: DSVDD)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: SMALL)')
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)

    # 配置模型及优化器
    if args.model=='DSVDD':
        model = choose_DSVDD_model(param_list, args.dataset)
    model.load_state_dict(torch.load(f'./checkpoint/{args.model}-MID-14-0%.pth'))

    # 加载数据集
    testset = load_test_csv(f'./dataset/{args.dataset}/testset')
    testset0 = [ x for x in testset if x[1]==0]
    testset1 = [ x for x in testset if x[1]>0]
    testset0 = testset0[:len(testset1)]
    testset = testset0 + testset1
    print(len(testset))

    # 求重构误差
    if args.model=='DSVDD':
        center = model.center.detach().clone()
        z, dis, label, _,_ = DSVDD_test(model, testset, center)
    elif args.model=='VAESVDD':
        center = model.center.detach().clone()
        z, dis, label, _,_ = VAESVDD_test(model, testset, center)
    label = label.astype(np.int32)
    print(center)

    z0 = z[label==0]
    z1 = z[label==1]
    z2 = z[label==2]
    z3 = z[label==3]
    z4 = z[label==4]
    z5 = z[label==5]
    z6 = z[label==6]
    z7 = z[label==7]
    z8 = z[label==8]
    z9 = z[label==9]
    dis0 = dis[label==0]
    dis1 = dis[label==1]
    dis2 = dis[label==2]
    dis3 = dis[label==3]
    dis4 = dis[label==4]
    dis5 = dis[label==5]
    dis6 = dis[label==6]
    dis7 = dis[label==7]
    dis8 = dis[label==8]
    dis9 = dis[label==9]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for i in range(max_label):
    num=10
    ax.scatter(z0[:num,0], z0[:num,1], z0[:num,2], c='black')
    ax.scatter(z1[:num,0], z1[:num,1], z1[:num,2], c='red')
    ax.scatter(z2[:num,0], z2[:num,1], z2[:num,2], c='blue')
    ax.scatter(z3[:num,0], z3[:num,1], z3[:num,2], c='grey')
    ax.scatter(z4[:num,0], z4[:num,1], z4[:num,2], c='green')
    ax.scatter(z5[:num,0], z5[:num,1], z5[:num,2], c='yellow')
    ax.scatter(z6[:num,0], z6[:num,1], z6[:num,2], c='orange')
    ax.scatter(z7[:num,0], z7[:num,1], z7[:num,2], c='violet')
    ax.scatter(z8[:num,0], z8[:num,1], z8[:num,2], c='cyan')
    ax.scatter(z9[:num,0], z9[:num,1], z9[:num,2], c='brown')
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')
    ax.set_zlabel('z2')
    plt.show()

    num=300
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dis0[:num], np.random.rand(min(num,dis0.shape[0])), c='black')
    ax.scatter(dis1[:num], np.random.rand(min(num,dis1.shape[0])), c='red')
    ax.scatter(dis2[:num], np.random.rand(min(num,dis2.shape[0])), c='blue')
    ax.scatter(dis3[:num], np.random.rand(min(num,dis3.shape[0])), c='grey')
    ax.scatter(dis4[:num], np.random.rand(min(num,dis4.shape[0])), c='green')
    ax.scatter(dis5[:num], np.random.rand(min(num,dis5.shape[0])), c='yellow')
    ax.scatter(dis6[:num], np.random.rand(min(num,dis6.shape[0])), c='orange')
    ax.scatter(dis7[:num], np.random.rand(min(num,dis7.shape[0])), c='violet')
    ax.scatter(dis8[:num], np.random.rand(min(num,dis8.shape[0])), c='cyan')
    ax.scatter(dis9[:num], np.random.rand(min(num,dis9.shape[0])), c='brown')
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')

    plt.show()

    # 计算ROC曲线和AUC面积
    label[label>0] = 1
    normalized_dis = dis/np.max(dis)
    fpr, tpr, thresholds = roc_curve(label, normalized_dis)
    auc_score = roc_auc_score(label, normalized_dis)
    print(auc_score)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    # {'Normal': 0, 'Exploits': 1, 'Reconnaissance': 2, 'DoS': 3, 'Generic': 4, 'Shellcode': 5, ' Fuzzers': 6, 'Worms': 7, 'Backdoors': 8, 'Analysis': 9}
    #      1             0                  0               0           1             0               1             0            1               1

    # log处理后模型越训练空间组织性越强，但是重构误差对样本可分性变差，最好的时候在10epoch
    # 不做log处理模型训练逐渐稳定，重构误差和空间组织性都不大变化，15epoch左右比较好，越训练会使得异常样本的重构误差也变小

if __name__ == '__main__':
    main()