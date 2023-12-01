import argparse
import torch
import torch.nn.functional as F
import random
import numpy as np

from utils.LoadData import *
from utils.trans import *
from utils.Logger import *
from utils.Params import *

from model.VAE import *
from model.WAE import *
from model.model import *
from run.train import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='VAEAD')
    
    # local setting
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='epochs for local training in each round aggregation (default: 2)')
    parser.add_argument('--batch_size', type=int, default=2048, metavar='B',
                        help='batch size for local training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate for local training (default: 1e-3)')
    
    # server setting
    parser.add_argument('--isload', type=str, default='No', choices=['Yes', 'No'],
                        help='is load model (default: No)')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'NSL-KDD'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--anomaly_rate', type=str, default='0%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='anomaly rate in trainset (default: 0%)')
    parser.add_argument('--model', type=str, default='VAE', choices=['VAE','WAE','DSVDD','DAGMM','weightWAE'],
                        help='model name (default: VAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    
    args = parser.parse_args()
    
    import logging
    serlogger = logging.getLogger('ser_logger')
    serlogger.setLevel(logging.INFO)
    file_hander = logging.FileHandler(f'./log/{args.dataset}-{args.anomaly_rate}-{args.model}-{args.model_size}.log')
    file_hander.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_hander.setFormatter(formatter)
    serlogger.addHandler(file_hander)
    
    serlogger.info(args)
    
    # 加载相应数据集
    trainset = load_csv(f'./dataset/{args.dataset}/trainset-{args.anomaly_rate}')
    testset = load_test_csv(f'./dataset/{args.dataset}/testset-{args.anomaly_rate}')
    
    # 定义模型size
    param_list = choose_param(args.dataset, args.model_size)

    # 初始化所有参与方的数据集，以及训练超参数
    # InitPool(param_list, trainset, args)
    
    metrics = []

    for rd_seed in [175,43,130,144,175]:
        serlogger.info(f'=================rd{rd_seed}=================')
        print(f'=================rd{rd_seed}=================')
        # set random seed
        torch.manual_seed(rd_seed)
        torch.cuda.manual_seed(rd_seed)
        np.random.seed(rd_seed)
        random.seed(rd_seed)
        # the convolution algorithm is fixed, so that the same input will get same output
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # ser self train
        global_epoch = 30
        metrics_column = ['auc', 'aupr', 'fpr95', 'test_norm_loss', 'test_all_loss',]
        if args.model in ['VAE','BetaVAE','WAE','weightWAE']:
            serModel = choose_model(param_list, args.model, args.dataset)
            # if args.isload=='Yes':
                # serModel.load_state_dict(torch.load("./checkpoint/UNSW-NB15-2%-MID-No-weightWAE-49.pth"))
            optimizer = torch.optim.Adam(serModel.parameters(), lr=args.lr)
            rec = server_train(serModel, optimizer, trainset, testset, serlogger, global_epoch, args)
            # df = pd.DataFrame(rec, columns=metrics_column)
            # df.to_csv(f"./result/{args.dataset}/{args.anomaly_rate}-{args.model}-{args.model_size}.csv", index=False)
        elif args.model=='DSVDD':
            DSVDD_model = choose_DSVDD_model(param_list)
            DSVDD_optimizer = torch.optim.AdamW(DSVDD_model.parameters(), lr=args.lr)
            rec = DSVDD_train(DSVDD_model, DSVDD_optimizer, trainset, testset, serlogger, global_epoch, args)
            # df = pd.DataFrame(rec, columns=metrics_column)
            # df.to_csv(f"./result/{args.dataset}/{args.anomaly_rate}-{args.model}-{args.model_size}.csv", index=False)
        elif args.model=='DAGMM':
            DAGMM_model = choose_DAGMM_model(param_list)
            DAGMM_optimizer = torch.optim.Adam(DAGMM_model.parameters(), lr=args.lr)
            rec = DAGMM_train(DAGMM_model, DAGMM_optimizer, trainset, testset, serlogger, global_epoch, args)
            # df = pd.DataFrame(rec, columns=metrics_column)
            # df.to_csv(f"./result/{args.dataset}/{args.anomaly_rate}-{args.model}-{args.model_size}.csv", index=False)
        metrics.append(rec[-1])
    metrics=np.array(metrics)
    mean = np.mean(metrics, axis=0)
    std = np.std(metrics, axis=0)
    print(f"mean:{mean}")
    print(f"std:{std}")
    serlogger.info(f'mean:{mean},std:{std}')


if __name__ == '__main__':
    main()