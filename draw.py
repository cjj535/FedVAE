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
    parser = argparse.ArgumentParser(description='VAE')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CICIDS17'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='WAE', choices=['VAE','BetaVAE','VQVAE','WAE'],
                        help='model name (default: WAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    parser.add_argument('--rate', type=str, default='0%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='model size (default: SMALL)')
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)

    # 配置模型及优化器
    model = choose_model(param_list, args.model, args.dataset)
    model.load_state_dict(torch.load(f'./checkpoint/WAE-MID-10-{args.rate}.pth'))

    # 加载数据集
    testset = load_test_csv(f'./dataset/{args.dataset}/testset-{args.rate}')
    print(len(testset))

    # 求重构误差
    z, dis, cos, label, _,_ = choose_test_all(model, testset, args.model)
    label = label.astype(np.int32)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num=1000
    colors = ['black','red','blue','grey','green','yellow','orange','violet','cyan','brown','pink','teal','gold','lavender']
    for i, color in enumerate(colors):
        z_tmp = z[label==i]
        ax.scatter(z_tmp[:num,0], z_tmp[:num,1], z_tmp[:num,2], c=color)
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')
    ax.set_zlabel('z2')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, color in enumerate(colors):
        dis_tmp = dis[label==i]
        cos_tmp = cos[label==i]
        ax.scatter(dis_tmp[:num], cos_tmp[:num], c=color)
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
    # {'Normal': 0, 'Generic': 1, 'Exploits': 2, 'Backdoor': 3, ' Reconnaissance ': 4, ' Fuzzers ': 5, 'DoS': 6, 'Reconnaissance': 7, 'Worms': 8, 'Analysis': 9, ' Fuzzers': 10, 'Shellcode': 11, ' Shellcode ': 12, 'Backdoors': 13}

if __name__ == '__main__':
    main()