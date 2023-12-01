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

from server import *
from client import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='FedVAE')
    
    # global setting
    parser.add_argument('--comm_round', type=int, default=100, metavar='T',
                        help='communication round for federated learning (default: 20)')
    parser.add_argument('--clt_num', type=int, default=100, metavar='K',
                        help='client number in federated learning (default: 10)')
    parser.add_argument('--act_rate', type=float, default=0.1, metavar='C',
                        help='the rate of client number for each round aggregation (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='N',
                        help='dirichlet distribution parameter (default: 1.0)')
    parser.add_argument('--distri', type=str, default='iid', choices=['iid','non-iid'],
                        help='iid or non-iid (default: iid)')
    
    parser.add_argument('--fl_method', type=str, default='FedAvg', choices=['FedAvg', 'FedProx', 'FedDyn', 'FedFT'],
                        help='aggregation method (default: FedAvg)')
    parser.add_argument('--isFT', type=str, default='No', choices=['Yes', 'No'],
                        help='is fine tuning (default: No)')
    
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
    parser.add_argument('--anomaly_rate', type=str, default='5%', choices=['0%','2%','5%'],
                        help='anomaly rate in trainset (default: 0%)')
    parser.add_argument('--model', type=str, default='weightWAE', choices=['VAE','WAE','DSVDD','DAGMM','weightWAE'],
                        help='model name (default: VAE)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: MID)')
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    
    args = parser.parse_args()
    
    import logging
    serlogger = logging.getLogger('ser_logger')
    serlogger.setLevel(logging.INFO)
    file_hander = logging.FileHandler(f'./FL_log/{args.dataset}-{args.fl_method}-{args.alpha}-{args.anomaly_rate}.log')
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
    InitPool(param_list, trainset, args)
    
    # set random seed
    rd_seed = args.seed
    torch.manual_seed(rd_seed)
    torch.cuda.manual_seed(rd_seed)
    np.random.seed(rd_seed)
    random.seed(rd_seed)
    # the convolution algorithm is fixed, so that the same input will get same output
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # fed learning
    if args.fl_method == 'FedAvg':
        rec = serFedAvg(trainset, testset, serlogger, param_list, args)
    elif args.fl_method == 'FedProx':
        rec = serFedProx(trainset, testset, serlogger, param_list, args)
    elif args.fl_method == 'FedDyn':
        rec = serFedDyn(trainset, testset, serlogger, param_list, args)
    elif args.fl_method == 'FedFT':
        _,rec = serFedFT(trainset, testset, serlogger, param_list, args)
    else:
        raise Warning('others FL methods are not implemented.')
    
    metrics_column = ['auc', 'test_norm_loss', 'test_all_loss']
    df = pd.DataFrame(rec, columns=metrics_column)
    df.to_csv(f"./result/{args.dataset}/{args.fl_method}-{args.alpha}-{args.anomaly_rate}.csv", index=False)


if __name__ == '__main__':
    main()