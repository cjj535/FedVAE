import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='VAE')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2018-Dos'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--model', type=str, default='WAE', choices=['VAE','BetaVAE','VQVAE','WAE',],
                        help='model name (default: WAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    parser.add_argument('--rate', type=str, default='0%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='model size (default: SMALL)')
    
    args = parser.parse_args()
    print(args)
    
    dir = f'./result/{args.dataset}'

    import os
    # 加载对应的csv文件
    target = f'{args.rate}'
    files_name = os.listdir(dir)
    target_list = [file for file in files_name if target in file]
    
    import pandas as pd
    
    auc_data = pd.DataFrame()
    
    for file in target_list:
        file_path = os.path.join(dir, file)
        df = pd.read_csv(file_path)
        column = df.iloc[:,0]
        auc_data[file] = column
    
    plt.style.use('seaborn-whitegrid')
    auc_data.plot(legend=True)
    plt.title("auc cmp")
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.savefig('test.png')

if __name__ == '__main__':
    main()