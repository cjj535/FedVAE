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

def generate_normal_data(model, samples_num):
    # generate samples
    gen_dataset = []
    samples = model.sample(samples_num*100, device)
    for sample in samples:
        gen_dataset.append([sample,2])
    
    return gen_dataset

def visual(feature):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=42)
    
    x_ts = ts.fit_transform(feature)
    print(x_ts.shape)
    
    x_min,x_max = x_ts.min(0),x_ts.max(0)
    x_final = (x_ts-x_min)/(x_max-x_min)
    
    return x_final

def plotlabels(S_lowDWeights, True_labels, name):
    True_labels = True_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    # print(S_data)
    # print(S_data.shape)  # [num, 3]

    colors = ['grey','blue']
    maker = ['o','^']
    for index in range(2):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.3)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    
def score_plotlabels(S_data, score, name):
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1]})
    
    X = S_data.loc[:]['x']
    Y = S_data.loc[:]['y']
    plt.scatter(X, Y, cmap='brg', s=100, marker='o', c=score, alpha=0.3)
    plt.colorbar()

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='tsne of normal attack pesudo')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018-Dos'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='WAE', choices=['VAE','WAE','weightWAE'],
                        help='model name (default: WAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    parser.add_argument('--rate', type=str, default='2%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='anomaly rate (default: 0%)')
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)
    
    # 加载数据集
    trainset = load_csv(f'./dataset/{args.dataset}/trainset-{args.rate}')
    testset = load_csv(f'./dataset/{args.dataset}/testset-{args.rate}')
    dataset0 = [ x for x in trainset if x[1]==0]
    dataset1 = [ x for x in testset if x[1]>0]
    dataset = random.sample(dataset0,10000)+random.sample(dataset1,10000)
    
    model = choose_model(param_list, args.model, args.dataset)
    model.load_state_dict(torch.load(f'./checkpoint/{args.dataset}-{args.rate}-{args.model}-{args.model_size}-9.pth'))

    _,_, cos, label, _,_,_ = choose_test_all(model, dataset, args.model)
    label = label.astype(np.int32)
    
    score = (cos-cos.min())/(cos.max()-cos.min())
    
    # t-SNE
    fig = plt.figure(figsize=(10, 10))
    X = np.array([x[0] for x in dataset])
    reduct_X = visual(X)
    true_label = np.array([ x[1] for x in dataset])
    plotlabels(reduct_X, true_label, 't-SNE true label')
    plt.savefig(f'tSNE-label {args.dataset}-{args.rate}-{args.model}-{args.model_size}.png')
    
    plt.figure(figsize=(10, 10))
    # plt.close()
    score_plotlabels(reduct_X, score, 't-SNE anomaly score')
    plt.savefig(f'tSNE-score {args.dataset}-{args.rate}-{args.model}-{args.model_size}.png')

if __name__ == '__main__':
    main()