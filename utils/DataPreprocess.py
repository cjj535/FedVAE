import pandas as pd
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def Save_df(train_file, test_file, df, save_nums, rate):
    # for col in df.columns:
    #     print(df[col].name, df[col].dtype)

    # 划分训练集和测试集，训练集只包含正常流量
    normal_df = df[df['label']==0]
    attack_df = df[df['label']==1]
    normal_nums = normal_df.shape[0]
    attack_nums = attack_df.shape[0]
    print(normal_nums, attack_nums)

    # shuffle打乱数据
    normal_df = normal_df.sample(frac=1, random_state=42)
    attack_df = attack_df.sample(frac=1, random_state=42)

    # 取50%的正常流量作为训练数据，剩余作为测试数据
    train_norm_nums = int(normal_nums*0.8)
    # train_norm_nums = 200000
    train_atta_nums = int(train_norm_nums*rate)
    train_df = pd.concat([normal_df[:train_norm_nums], attack_df[:train_atta_nums]], ignore_index=True)
    test_df = pd.concat([normal_df[train_norm_nums:], attack_df[train_atta_nums:]], ignore_index=True)

    # 分文件保存训练集
    train_nums = train_norm_nums+train_atta_nums
    for i in range(save_nums-1):
        df_tmp = train_df[int(i*train_nums/save_nums):int((i+1)*train_nums/save_nums)]
        df_tmp.to_csv(train_file+str(i)+'.csv', index=False)
    df_tmp = train_df[int((save_nums-1)*train_nums/save_nums):]
    df_tmp.to_csv(train_file+str(save_nums-1)+'.csv', index=False)

    # 分文件保存测试集
    test_nums = normal_nums+attack_nums-train_nums
    for i in range(save_nums-1):
        df_tmp = test_df[int(i*test_nums/save_nums):int((i+1)*test_nums/save_nums)]
        df_tmp.to_csv(test_file+str(i)+'.csv', index=False)
    df_tmp = test_df[int((save_nums-1)*test_nums/save_nums):]
    df_tmp.to_csv(test_file+str(save_nums-1)+'.csv', index=False)

def UNSW_NB15_Preprocess(src_file_list, train_file, test_file, save_nums, rate):
    # 把所有csv文件导入放入一个list里
    df_list = []
    for src_file_name in src_file_list:
        # 首先只读取前49列，然后如果有空格的值会被视为Nan，第一行数据作废
        tmp_df = pd.read_csv(src_file_name, header=None, skiprows=1, usecols=range(49), na_values=' ', low_memory=False)
        df_list.append(tmp_df)    
    # df list拼接成一个df统一处理
    df = pd.concat(df_list,ignore_index=True)

    # 添加每一列的名字
    df.columns = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl',
               'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
               'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt',
               'Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd',
               'is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
               'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label']
    
    # for col in df.columns:
    #     print(df[col].name, df[col].dtype)
    
    # 填充attack cat列的normal标签
    # df.loc[df['label']==0, 'attack_cat'] = 'Normal'
    # 填充attack cat列的normal标签
    df['attack_cat'] = df['attack_cat'].fillna('Normal')
    
    # 填充label列的0,1
    df.loc[df['attack_cat']=='Normal', 'label'] = 0
    df.loc[df['attack_cat']!='Normal', 'label'] = 1

    # 填充0,1
    df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].fillna(0)
    df['is_ftp_login'] = df['is_ftp_login'].fillna(0)
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].fillna(0)
    print(df.shape[0])

    # 清除有空白的行，丢弃这些数据
    df = df.dropna()
    print(df.shape[0])

    # 数值截断
    df['dur']=df['dur'].clip(upper=60)
    df['sbytes']=df['sbytes'].clip(upper=2e6)
    df['dbytes']=df['dbytes'].clip(upper=2e6)
    df['Sload']=df['Sload'].clip(upper=1e9)
    df['Dload']=df['Dload'].clip(upper=2e7)
    df['Sjit']=df['Sjit'].clip(upper=1e5)
    df['Djit']=df['Djit'].clip(upper=1e5)
    df['Sintpkt']=df['Sintpkt'].clip(upper=1e3)
    df['Dintpkt']=df['Dintpkt'].clip(upper=1e3)

    # 丢弃列，并且不修改后两列关于label的信息
    df.pop('Stime')
    df.pop('Ltime')
    df.pop('srcip')
    df.pop('sport')
    df.pop('dstip')
    df.pop('dsport')
    # for column in ['dsport']:
    #     # 取出某一列，并统计不同的值出现的次数
    #     column_counts = df[column].value_counts()
    #     # 对统计结果进行排序
    #     sorted_counts = column_counts.sort_values(ascending=False)
    #     sorted_counts = sorted_counts.head(10)
    #     # print(sorted_counts)
    #     top_10_values = sorted_counts.index
    #     # 将这一列中不是前10个值的值变为 -1
    #     df[column] = df[column].apply(lambda x: x if x in top_10_values else -1)

    column_label = df.pop('label')
    column_attack = df.pop('attack_cat')

    # 最大最小归一化或者one-hot编码
    for col in df.columns:
        if df[col].dtype == 'object':
            df = pd.get_dummies(df, columns=[col])
            # print(col)
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            # df[col] = scalar.fit_transform(df[[col]])
            mmin = df[col].min()
            mmax = df[col].max()
            # print(col, mmin, mmax)
            if mmax <= mmin:
                df.pop(col)
            else:
                mmin = max(-1,mmin)
                df[col] = (df[col] - mmin) / (mmax - mmin)
        else:
            print('error!!!')
    print(f"after drop cols and rows: {df.shape}")

    # 把label放在后面，依次是数值、独热码、label
    print(df.shape, column_label.shape, column_attack.shape)
    df.insert(df.shape[1], 'label', column_label)
    df.insert(df.shape[1], 'type', column_attack)

    Save_df(train_file, test_file, df, save_nums, rate)

'''
def UNSW_NB15_Preprocess(src_file_list, train_file, test_file, save_nums, rate):
    # 把所有csv文件导入放入一个list里
    df_list = []
    for src_file_name in src_file_list:
        # 首先只读取前45列
        tmp_df = pd.read_csv(src_file_name, na_values=' ', usecols=range(45), low_memory=False)
        df_list.append(tmp_df)
    # df list拼接成一个df统一处理
    df = pd.concat(df_list,ignore_index=True)
    
    # for col in df.columns:
    #     print(df[col].name, df[col].dtype)

    # 清除有空白的行，丢弃这些数据
    print(df.shape[0])
    df = df.dropna()
    print(df.shape[0])

    # 丢弃列，并且不修改后两列关于label的信息
    df.pop('id')
    column_label = df.pop('label')
    column_attack = df.pop('attack_cat')

    # 最大最小归一化
    scalar = MinMaxScaler()
    # 最大最小归一化或者one-hot编码
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col] = scalar.fit_transform(df[[col]])
        elif df[col].dtype == 'object':
            df = pd.get_dummies(df, columns=[col])
        else:
            print('error!!!')
    print(f"after drop cols and rows: {df.shape}")

    # 把label放在后面，依次是数值、独热码、label
    df.insert(df.shape[1], 'label', column_label)
    df.insert(df.shape[1], 'type', column_attack)

    Save_df(train_file, test_file, df, save_nums, rate)
'''

def CIC_IDS2018_Preprocess(src_file_list, train_file, test_file, save_nums, rate):
    # 把所有csv文件导入放入一个list里
    df_list = []
    for src_file_name in src_file_list:
        # 首先只读取前80列，然后如果有空格的值会被视为Nan，第一行数据作废
        tmp_df = pd.read_csv(src_file_name, header=None, skiprows=1, usecols=range(80), na_values=' ', low_memory=False)

        # 删除多余的header
        tmp_df = tmp_df[tmp_df[79]!='Label']

        # 把选取的添加到list中
        df_list.append(tmp_df)
    
    # 拼接所有数据到一个df中统一处理
    df = pd.concat(df_list,ignore_index=True)

    # 添加每一列的名字
    df.columns = ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
                'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 
                'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 
                'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 
                'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 
                'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 
                'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 
                'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 
                'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
                'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label',]
    
    print(f"before drop na: {df.shape}")
    # 清除有空白的行，丢弃这些数据
    df = df.dropna()
    print(f"after drop na: {df.shape}")

    # for col in df.columns:
    #     print(df[col].name,df[col].dtype)
    df.pop('Timestamp')

    # df所有列转为数字
    # print(df.shape)
    column_type = df.pop('Label')
    df = df.apply(pd.to_numeric, errors='coerce')
    df.insert(df.shape[1], 'Label', column_type)
    # print(df.shape, column_type.shape)
    df = df.dropna()
    print(df.shape)
    # for col in df.columns:
    #     print(df[col].name,df[col].dtype)

    '''
    # 绘制每一列的直方图
    column_label = df.pop('Label')
    # 去掉带异常值的行，包含小于-1的数和inf无穷大
    # df = df[df>0]
    # df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(df.shape)
    # df = df.apply(lambda x: np.log10(x))
    df = df.clip(upper=1e8)
    df = df.clip(lower=1)
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        bins_edge = [1,100,1000,10000,100000,1000000,10000000,100000000]
        
        hist, bins = np.histogram(df[column],bins=bins_edge)
        per_values = (hist/len(df[column]))*100
        print(per_values)

        plt.hist(df[column], bins=bins_edge, edgecolor='k')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.xscale('log')
        plt.xticks(bins_edge)
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    exit(0)'''

    # 数值截断
    df['Tot Fwd Pkts']=df['Tot Fwd Pkts'].clip(upper=1e2)
    df['Tot Bwd Pkts']=df['Tot Bwd Pkts'].clip(upper=1e2)
    df['TotLen Fwd Pkts']=df['TotLen Fwd Pkts'].clip(upper=1e4)
    df['TotLen Bwd Pkts']=df['TotLen Bwd Pkts'].clip(upper=1e4)
    df['Fwd Pkt Len Max']=df['Fwd Pkt Len Max'].clip(upper=1e4)
    df['Fwd Pkt Len Min']=df['Fwd Pkt Len Min'].clip(upper=1e2)
    df['Fwd Pkt Len Mean']=df['Fwd Pkt Len Mean'].clip(upper=1e3)
    df['Fwd Pkt Len Std']=df['Fwd Pkt Len Std'].clip(upper=1e3)
    df['Bwd Pkt Len Max']=df['Bwd Pkt Len Max'].clip(upper=1e4)
    df['Bwd Pkt Len Min']=df['Bwd Pkt Len Min'].clip(upper=1e3)
    df['Bwd Pkt Len Mean']=df['Bwd Pkt Len Mean'].clip(upper=1e3)
    df['Bwd Pkt Len Std']=df['Bwd Pkt Len Std'].clip(upper=1e3)
    df['Flow Byts/s']=df['Flow Byts/s'].clip(upper=1e6)
    df['Flow Pkts/s']=df['Flow Pkts/s'].clip(upper=1e7)
    df['Fwd Header Len']=df['Fwd Header Len'].clip(upper=1e3)
    df['Bwd Header Len']=df['Bwd Header Len'].clip(upper=1e3)
    df['Fwd Pkts/s']=df['Fwd Pkts/s'].clip(upper=1e6)
    df['Bwd Pkts/s']=df['Bwd Pkts/s'].clip(upper=1e6)
    df['Pkt Len Min']=df['Pkt Len Min'].clip(upper=1e2)
    df['Pkt Len Max']=df['Pkt Len Max'].clip(upper=1e4)
    df['Pkt Len Mean']=df['Pkt Len Mean'].clip(upper=1e3)
    df['Pkt Len Std']=df['Pkt Len Std'].clip(upper=1e3)
    df['Pkt Len Var']=df['Pkt Len Var'].clip(upper=1e6)
    df['Down/Up Ratio']=df['Down/Up Ratio'].clip(upper=1e2)
    df['Pkt Size Avg']=df['Pkt Size Avg'].clip(upper=1e3)
    df['Fwd Seg Size Avg']=df['Fwd Seg Size Avg'].clip(upper=1e3)
    df['Bwd Seg Size Avg']=df['Bwd Seg Size Avg'].clip(upper=1e3)
    df['Subflow Fwd Pkts']=df['Subflow Fwd Pkts'].clip(upper=1e2)
    df['Subflow Fwd Byts']=df['Subflow Fwd Byts'].clip(upper=1e4)
    df['Subflow Bwd Pkts']=df['Subflow Bwd Pkts'].clip(upper=1e2)
    df['Subflow Bwd Byts']=df['Subflow Bwd Byts'].clip(upper=1e4)
    df['Fwd Act Data Pkts']=df['Fwd Act Data Pkts'].clip(upper=1e2)

    # 对dst port这一列做onehot处理，考虑可选值太多，只选取出现次数在前10个的端口
    df.pop('Dst Port')
    # for column in ['Dst Port']:
    #     # 取出某一列，并统计不同的值出现的次数
    #     column_counts = df[column].value_counts()
    #     # 对统计结果进行排序
    #     sorted_counts = column_counts.sort_values(ascending=False)
    #     sorted_counts = sorted_counts.head(10)
    #     # print(sorted_counts)
    #     top_10_values = sorted_counts.index
    #     # 将这一列中不是前10个值的值变为 -1
    #     df[column] = df[column].apply(lambda x: x if x in top_10_values else -1)

    column_label = df.pop('Label')

    # 最大最小归一化
    for col in df.columns:
        if col in ['Dst Port', 'Protocol']:
            df = pd.get_dummies(df, columns=[col])
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            mmin = df[col].min()
            mmax = df[col].max()
            # print(col, mmin, mmax)
            if mmax <= mmin:
                df.pop(col)
                print(col)
            else:
                mmin = max(-1,mmin)
                df[col] = (df[col] - mmin) / (mmax - mmin)
        else:
            print('error!!!')
    print(f"after drop cols and rows: {df.shape}")

    # 把label放在后面，依次是数值、one hot编码、label
    print(df.shape, column_label.shape)
    df['label'] = 0
    df.loc[column_label!='Benign', 'label'] = 1
    df.insert(df.shape[1], 'type', column_label)

    Save_df(train_file, test_file, df, save_nums, rate)

'''
def TON_IoT_Preprocess(src_file_list, train_file, test_file, save_nums, rate):
    # 把所有csv文件导入放入一个list里
    df_list = []
    for src_file_name in src_file_list:
        tmp_df = pd.read_csv(src_file_name, na_values=' ', low_memory=False)
        df_list.append(tmp_df)
    # df list拼接成一个df统一处理
    df = pd.concat(df_list,ignore_index=True)
        
    print(df.shape[0])
    # for col in df.columns:
    #     print(df[col].dtype)
    # # 清除有空白的行，丢弃这些数据
    df = df.dropna()
    print(df.shape[0])

    for column in ['dst_port']:
        # 取出某一列，并统计不同的值出现的次数
        column_counts = df[column].value_counts()
        # 对统计结果进行排序
        sorted_counts = column_counts.sort_values(ascending=False)
        sorted_counts = sorted_counts.head(40)
        # print(sorted_counts)
        top_40_values = sorted_counts.index
        # 将这一列中不是前40个值的值变为 -1
        df[column] = df[column].apply(lambda x: x if x in top_40_values else -1)

    # 丢弃前四列，并且不修改后两列关于label的信息
    df.pop('ts')
    df.pop('src_ip')
    df.pop('src_port')
    df.pop('dst_ip')
    df.pop('dns_query')
    column_label = df.pop('label')
    column_attack = df.pop('type')

    # 最大最小归一化或者one-hot编码
    for col in df.columns:
        if col in ['dst_port','dns_qclass','dns_qtype','dns_rcode','http_status_code'] or df[col].dtype=='object':
            df = pd.get_dummies(df, columns=[col])
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            mmin = df[col].min()
            mmax = df[col].max()
            # print(col, mmin, mmax)
            if mmax <= mmin:
                df.pop(col)
            else:
                mmin = max(-1,mmin)
                df[col] = (df[col] - mmin) / (mmax - mmin)
        else:
            print('error!!!')
    print(f"after drop cols and rows: {df.shape}")

    # 把label放在后面，依次是数值、独热码、label
    df.insert(df.shape[1], 'label', column_label)
    df.insert(df.shape[1], 'type', column_attack)

    Save_df(train_file, test_file, df, save_nums, rate)
'''

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Preprocess')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018-Dos', 'CIC-IDS2018-Infiltration'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--rate', type=float, default=0.00,
                        help='attack rate in trainset (default: 0.00)')
    args = parser.parse_args()
    print(args)
    
    path = './dataset/'+args.dataset+'/src'
    file_list = []
    file_name = os.listdir(path)
    for name in file_name:
        file_list.append(os.path.join(path,name))
    train_file = f'./dataset/{args.dataset}/trainset-{int(args.rate*100)}%/preprocessed_train'
    test_file = f'./dataset/{args.dataset}/testset-{int(args.rate*100)}%/preprocessed_test'
    if args.dataset == 'UNSW-NB15':
        UNSW_NB15_Preprocess(file_list, train_file, test_file, save_nums=10, rate = args.rate)
    elif args.dataset == 'CIC-IDS2018-Dos':
        CIC_IDS2018_Preprocess(file_list, train_file, test_file, save_nums=10, rate = args.rate)
    elif args.dataset == 'CIC-IDS2018-Infiltration':
        CIC_IDS2018_Preprocess(file_list, train_file, test_file, save_nums=10, rate = args.rate)

if __name__ == '__main__':
    main()