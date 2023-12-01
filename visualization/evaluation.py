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
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'NSL-KDD'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='WAE', choices=['VAE','BetaVAE','VQVAE','WAE','weightWAE'],
                        help='model name (default: WAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    parser.add_argument('--rate', type=str, default='1%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='anomaly rate (default: 0%)')
    parser.add_argument('--epoch', type=int, default=0,
                        help='model (default: 0)')
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)
    
    # 加载数据集
    # trainset = load_csv(f'./dataset/{args.dataset}/trainset-{args.rate}')
    testset = load_csv(f'./dataset/{args.dataset}/testset-{args.rate}')
    dataset0 = [ x for x in testset if x[1]==0]
    dataset1 = [ x for x in testset if x[1]>0]
    dataset = random.sample(dataset0,10000)+random.sample(dataset1,10000)
    
    model = choose_model(param_list, args.model, args.dataset)
    model.load_state_dict(torch.load(f'./checkpoint/{args.dataset}-{args.rate}-{args.model}-{args.model_size}-{args.epoch}.pth'))

    # 求异常分数
    _, score, _, label, _,_,_ = choose_test_all(model, dataset, args.model)
    label = label.astype(np.int32)
    
    # score = score[score<3]
    
    # 计算ROC曲线和AUC面积
    label[label>0] = 1
    normalized_score = score/np.max(score)
    fpr, tpr, thresholds = roc_curve(label, normalized_score)
    auc_score = roc_auc_score(label, normalized_score)
    print(auc_score)
    roc_auc = auc(fpr, tpr)
    
    # 密度分布曲线
    norm_score = score[label==0]
    # norm_score = norm_score[norm_score<3]
    attk_score = score[label>0]
    # attk_score = attk_score[attk_score<3]
    print(len(norm_score), len(attk_score))
    
    import seaborn as sns
    
    # 画密度分布图
    sns.set(style="whitegrid")
    sns.kdeplot(norm_score, color='blue', fill=True, label='normal')
    sns.kdeplot(attk_score, color='red', fill=True, label='anomaly')

    # 添加图例
    plt.legend()
    
    # 添加竖线
    sorted_norm_score = np.sort(norm_score)
    percentile_95 = sorted_norm_score[int(0.95*norm_score.shape[0])]
    plt.axvline(x=percentile_95, color='black', linestyle='--', label='Threshold')

    # 在竖线旁边添加文本
    plt.text(percentile_95 + 0.1, 0.2, 'TPR=0.95', color='black')

    # 添加标题和标签
    plt.title('')
    plt.xlabel('Score')
    plt.ylabel('Density')
    
    # # 
    # x_vals = np.linspace(min(norm_score.min(),attk_score.min()), max(norm_score.max(),attk_score.max()), 100)
    # plt.plot(x_vals, norm_kde(x_vals), label="norm", color='blue')
    # plt.plot(x_vals, attk_kde(x_vals), label='attk', color='orange')
    # plt.legend()
    # plt.title('distribution')
    # plt.xlabel('score')
    # plt.ylabel('density')
    plt.savefig(f"density distribution {args.dataset}-{args.rate}-{args.model}-{args.model_size}.png")
    
    '''
    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"ROC {args.dataset}-{args.rate}-{args.model}-{args.model_size}.png")
    '''
    
if __name__ == '__main__':
    main()