import argparse
import torch
import torch.nn.functional as F
import random
import numpy as np

from utils.LoadData import *
from utils.Logger import *
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
    
    parser.add_argument('--aggregation_method', type=str, default='FedAvg', choices=['FedAvg', 'FedProx', 'FedDyn'],
                        help='aggregation method (default: FedAvg)')
    parser.add_argument('--isFT', type=str, default='No', choices=['Yes', 'No'],
                        help='is fine tuning (default: No)')
    
    # local setting
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='epochs for local training in each round aggregation (default: 2)')
    parser.add_argument('--batch_size', type=int, default=1024*2, metavar='B',
                        help='batch size for local training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate for local training (default: 1e-3)')
    
    # server setting
    parser.add_argument('--isser', type=str, default='Yes', choices=['Yes', 'No'],
                        help='is server train itself (default: No)')
    parser.add_argument('--isclt', type=str, default='No', choices=['Yes', 'No'],
                        help='is client train itself (default: No)')
    parser.add_argument('--isload', type=str, default='No', choices=['Yes', 'No'],
                        help='is load model (default: No)')
    
    parser.add_argument('--dataset', type=str, default='UNSW-NB15', choices=['UNSW-NB15', 'CIC-IDS2017-Dos', 'CIC-IDS2017-Infiltration'],
                        help='dataset name (default: UNSW-NB15)')
    parser.add_argument('--anomaly_rate', type=str, default='0%', choices=['0%','1%','2%','3%','4%','5%'],
                        help='anomaly rate in trainset (default: 0%)')
    parser.add_argument('--model', type=str, default='VAE', choices=['VAE','BetaVAE','WAE'],
                        help='model name (default: VAE)')
    parser.add_argument('--cmp_model', type=str, default='No', choices=['No','DSVDD','DAGMM'],
                        help='model name (default: No)')
    parser.add_argument('--model_size', type=str, default='MID', choices=['SMALL','MID','BIG'],
                        help='model size (default: SMALL)')
    
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    
    args = parser.parse_args()
    serlogger.info(args)
    
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # the convolution algorithm is fixed, so that the same input will get same output
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载相应数据集
    trainset = load_csv(f'./dataset/{args.dataset}/trainset-{args.anomaly_rate}')
    testset = load_test_csv(f'./dataset/{args.dataset}/testset-{args.anomaly_rate}')
    testset0 = [ x for x in testset if x[1]==0]
    testset1 = [ x for x in testset if x[1]>0]
    testset = random.sample(testset0,20000)+random.sample(testset1,20000)
    # print(f'testset number: {len(testset)} normal: {len(testset0)} attack: {len(testset1)}')
    
    # 定义模型size
    param_list = choose_param(args.dataset, args.model_size)

    # 初始化所有参与方的数据集，以及训练超参数
    InitPool(param_list, trainset, args)

    # ser self train
    global_epoch = 100
    metrics_column = ['auc', 'train_loss', 'test_norm_loss', 'test_loss',]
    if args.isser == 'Yes':
        if args.cmp_model == 'No':
            serModel = choose_model(param_list, args.model, args.dataset)
            optimizer = torch.optim.Adam(serModel.parameters(), lr=args.lr)
            rec = server_train(serModel, optimizer, trainset, testset, global_epoch, args)
            df = pd.DataFrame(rec, columns=metrics_column)
            df.to_csv(f"./result/{args.dataset}/{args.anomaly_rate}-{args.model}-{args.model_size}.csv", index=False)
        elif args.cmp_model=='DSVDD':
            DSVDD_model = choose_DSVDD_model(param_list)
            DSVDD_optimizer = torch.optim.AdamW(DSVDD_model.parameters(), lr=args.lr)
            rec = DSVDD_train(DSVDD_model, DSVDD_optimizer, trainset, testset, global_epoch, args)
            df = pd.DataFrame(rec, columns=metrics_column)
            df.to_csv(f"./result/{args.dataset}/{args.anomaly_rate}-{args.cmp_model}-{args.model_size}.csv", index=False)
        elif args.cmp_model=='DAGMM':
            DAGMM_model = choose_DAGMM_model(param_list)
            DAGMM_optimizer = torch.optim.Adam(DAGMM_model.parameters(), lr=args.lr)
            rec = DAGMM_train(DAGMM_model, DAGMM_optimizer, trainset, testset, global_epoch, args)
            df = pd.DataFrame(rec, columns=metrics_column)
            df.to_csv(f"./result/{args.dataset}/{args.anomaly_rate}-{args.cmp_model}-{args.model_size}.csv", index=False)
        return
    
    # fed learning
    if args.aggregation_method == 'FedAvg':
        serFedAvg(testset, param_list, args)
    elif args.aggregation_method == 'FedProx':
        serFedProx(testset, param_list, args)
    elif args.aggregation_method == 'FedDyn':
        serFedDyn(testset, param_list, args)
    else:
        raise Warning('others FL methods are not implemented.')


if __name__ == '__main__':
    main()