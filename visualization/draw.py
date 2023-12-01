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

def visual(feature):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=42)
    
    x_ts = ts.fit_transform(feature)
    print(x_ts.shape)
    
    x_min,x_max = x_ts.min(0),x_ts.max(0)
    x_final = (x_ts-x_min)/(x_max-x_min)
    
    return x_final

def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    # print(S_data)
    # print(S_data.shape)  # [num, 3]

    colors = ['black','red','blue','grey','green','yellow','orange','violet','cyan','brown','pink','teal','gold','lavender']
    maker = ['o','^','^','^','^','^','^','^','^','^','^','^','^','^']
    for index in range(14):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)

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
    
    args = parser.parse_args()
    print(args)

    param_list = choose_param(args.dataset, args.model_size)
    
    num=50
    colors = ['black','red','blue','grey','green','yellow','orange','violet','cyan','brown','pink','teal','gold','lavender']
    
    # 加载数据集
    testset = load_test_csv(f'./dataset/{args.dataset}/testset-{args.rate}')
    testset = random.sample(testset, 100000)
    print(len(testset))
    
    model = choose_model(param_list, args.model, args.dataset)
    model.load_state_dict(torch.load(f'./checkpoint/{args.dataset}-{args.rate}-{args.model}-{args.model_size}-49.pth'))

    # 求重构误差
    z, dis, cos, label, score,_,_ = choose_test_all(model, testset, args.model)
    label = label.astype(np.int32)
    
    '''
    pre_label = np.zeros_like(label)
    pre_label[dis>0.5]=2
    pre_label[dis<=0.5]=0
    true_label = np.zeros_like(label)
    true_label[label>0]=1
    true_label[label==0]=0
    pre_label = pre_label+true_label
    # PCA
    pca = PCA(n_components=2)  # 指定降维后的维度为3
    z_3d = pca.fit_transform(z)  # X是你的数据集
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, color in enumerate(colors):
        z_tmp = z_3d[label==i]
        
        ax.scatter(z_tmp[:num,0], z_tmp[:num,1], c=color)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.savefig(f'PCA{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    
    # 重构误差，真实标签与隐空间分布的关系
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, color in enumerate(colors):
        z_tmp = z_3d[pre_label==i]
        ax.scatter(z_tmp[:num,0], z_tmp[:num,1], c=color)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plt.savefig(f'pre_label-PCA{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    '''
    '''
    # t-SNE
    fig = plt.figure(figsize=(10, 10))
    random_indices = random.sample(range(0,z.shape[0]-1), 10000)
    tmp_z = z[random_indices,:]
    label_tmp = label[random_indices]
    plotlabels(visual(tmp_z), label_tmp, '(a)')
    plt.savefig(f'tSNE{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i, color in enumerate(colors):
    #     z_tmp = z[label==i]
    #     ax.scatter(z_tmp[:num,0], z_tmp[:num,1], z_tmp[:num,2], c=color)
    # ax.set_xlabel('z0')
    # ax.set_ylabel('z1')
    # ax.set_zlabel('z2')
    # plt.savefig(f'z dis-{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, color in enumerate(colors):
        dis_tmp = dis[label==i]
        cos_tmp = cos[label==i]
        ax.scatter(dis_tmp[:num], cos_tmp[:num], c=color)
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')
    plt.savefig(f'recon dis{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    
    '''
    # exit(0)
    '''
    dis_cum = None
    for i in range(100):
        model.load_state_dict(torch.load(f'./checkpoint/{args.dataset}-{args.rate}-{args.model_size}-{args.model}-{i}.pth'))
        _, dis_cur, _, _, _,_ = choose_test_all(model, testset, args.model)
        
        if i==0:
            dis_cum = dis_cur[:,np.newaxis]
        else:
            dis_cum = np.concatenate((dis_cum,dis_cur[:,np.newaxis]), axis=1)
    
    normal_dis = dis_cum[label==0]
    attack_dis = dis_cum[label>0]
    
    # 画密度图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 计算每个时刻下的分布密度
    data=normal_dis.T
    density_data = []
    bin_edges = np.arange(0, 15, 0.2)  # 以0.2为区间
    for t in range(data.shape[0]):
        hist, _ = np.histogram(data[t], bins=bin_edges)
        density_data.append(hist)
    # 将分布密度数据转换为NumPy数组
    density_data = np.array(density_data)
    # 创建热图
    plt.figure(figsize=(10, 10))
    plt.imshow(density_data, cmap='viridis', aspect='auto', extent=[0, 1, 0, data.shape[0]])
    # 添加横轴和纵轴标签
    plt.xlabel('Value Range')
    plt.ylabel('Time Steps')
    # 添加颜色条
    plt.colorbar(label='Density')
    # 显示图形
    plt.title('Density Distribution Over Time')
    plt.savefig(f'normal density distribution-{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 计算每个时刻下的分布密度
    data=attack_dis.T
    density_data = []
    bin_edges = np.arange(0, 15, 0.2)  # 以0.2为区间
    for t in range(data.shape[0]):
        hist, _ = np.histogram(data[t], bins=bin_edges)
        density_data.append(hist)
    # 将分布密度数据转换为NumPy数组
    density_data = np.array(density_data)
    # 创建热图
    plt.figure(figsize=(10, 10))
    plt.imshow(density_data, cmap='viridis', aspect='auto', extent=[0, 1, 0, data.shape[0]])
    # 添加横轴和纵轴标签
    plt.xlabel('Value Range')
    plt.ylabel('Time Steps')
    # 添加颜色条
    plt.colorbar(label='Density')
    # 显示图形
    plt.title('Density Distribution Over Time')
    plt.savefig(f'attack density distribution-{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    '''
    # 不易区分样本的重构误差变化曲线
    # dis_cum_ab = dis_cum[dis>0.5]
    # cos_ab = cos[dis>0.5]
    # label_ab = label[dis>0.5]
    # label_ab[label_ab>0] = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # colors = ['green','red']
    # for i, color in enumerate(colors):
    #     dis_cum_tmp = dis_cum_ab[label_ab==i]
    #     cos_tmp = cos_ab[label_ab==i]
    #     print(dis_cum_tmp.shape)
    #     num=20
    #     ax.plot((dis_cum_tmp[:num]).T, c=color)
    # ax.set_xlabel('recon error')
    # ax.set_ylabel('epoch')
    # plt.savefig(f'recon variation-{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    '''
    dis_cum = np.sum(dis_cum,axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, color in enumerate(colors):
        dis_cum_tmp = dis_cum[label==i]
        cos_tmp = cos[label==i]
        ax.scatter(dis_cum_tmp[:num], cos_tmp[:num], c=color)
    ax.set_xlabel('z0')
    ax.set_ylabel('z1')
    plt.savefig(f'recon cum dis{args.dataset}-{args.rate}-{args.model_size}-{args.model}.png')
    
    label[label>0] = 1
    normalized_dis_cum = dis_cum/np.max(dis_cum)
    fpr, tpr, thresholds = roc_curve(label, normalized_dis_cum)
    auc_score = roc_auc_score(label, normalized_dis_cum)
    print(auc_score)
    '''
    
    # 计算ROC曲线和AUC面积
    label[label>0] = 1
    normalized_score = score/np.max(score)
    fpr, tpr, thresholds = roc_curve(label, normalized_score)
    auc_score = roc_auc_score(label, normalized_score)
    print(auc_score)
    roc_auc = auc(fpr, tpr)

    # # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("show.png")
    # {'Normal': 0, 'Generic': 1, 'Exploits': 2, 'Backdoor': 3, ' Reconnaissance ': 4, ' Fuzzers ': 5, 'DoS': 6, 'Reconnaissance': 7, 'Worms': 8, 'Analysis': 9, ' Fuzzers': 10, 'Shellcode': 11, ' Shellcode ': 12, 'Backdoors': 13}

if __name__ == '__main__':
    main()