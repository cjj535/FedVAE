import torch
import numpy as np
import copy
import random
import logging
import pandas as pd
import os

def generate_data(data_list, batch_size=64, is_shuffle=True):

    if is_shuffle:
        random.shuffle(data_list)    

    # load data
    cnt = 0
    list_len = len(data_list)
    while cnt < list_len-1:
        X = [ torch.Tensor(data_list[cnt+i][0]) for i in range(min(batch_size, list_len-cnt))]
        Y = [ data_list[cnt+i][1] for i in range(min(batch_size, list_len-cnt))]
        X = torch.stack(X)
        Y = torch.Tensor(Y)
        # print(X, Y)
        # print(X.shape, Y.shape)
        # exit(0)
        yield X, Y
        cnt += batch_size

def load_csv(path):
    # 加载数据集
    files = os.listdir(path)
    df_list = []
    for file in files:
        file_path = os.path.join(path, file)
        tmp_df = pd.read_csv(file_path)
        df_list.append(tmp_df)

    # 拼接数据矩阵
    df = pd.concat(df_list,ignore_index=True)
    
    # 丢弃空白值行
    df = df.dropna()

    # 取出数据部分，和label部分
    data_list = [[row[:-2].to_numpy(dtype=float), row[-2] ] for _, row in df.iterrows()]
    
    return data_list

def load_test_csv(path):
    # 加载数据集
    files = os.listdir(path)
    df_list = []
    for file in files:
        file_path = os.path.join(path, file)
        tmp_df = pd.read_csv(file_path)
        df_list.append(tmp_df)

    # 拼接数据矩阵
    df = pd.concat(df_list,ignore_index=True)
    
    # 丢弃空白值行
    df = df.dropna()

    # 转换类别标签
    # {'Normal': 0, 'Exploits': 1, 'Reconnaissance': 2, 'DoS': 3, 'Generic': 4, 'Shellcode': 5, ' Fuzzers': 6, 'Worms': 7, 'Backdoors': 8, 'Analysis': 9}
    unique_categories = df['type'].unique()
    category_mapping = {category: index for index, category in enumerate(unique_categories)}
    print(category_mapping)

    # 使用map()方法将类别标签转换为数字标签
    df['numeric_label'] = df['type'].map(category_mapping)

    # 取出数据部分，和label部分
    data_list = [[row[:-3].to_numpy(dtype=float), row[-1] ] for _, row in df.iterrows()]
    
    return data_list