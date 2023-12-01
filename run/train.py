import torch
import math

from model.VAE import *
from model.VQVAE import *
from model.WAE import *
from model.weightWAE import *

from utils.LoadData import *
from utils.Logger import *
from utils.Params import *
from utils.SVDD import *

from run.test import *

def server_train(model, optimizer, trainset, testset, logger, epochs, args, isTest=True):
    batch_size = args.batch_size

    # batch_num = len(trainset)/batch_size
    metrics = []

    for epoch in range(epochs):
        model.train()
        # loss_rec = 0
        # recon_loss_rec = 0
        # reg_loss_rec = 0
        dataloader = generate_data(trainset, batch_size=batch_size, is_shuffle=True)
        for _,data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # if args.ismask=='Yes':
            #     mask = torch.ones_like(inputs, dtype=torch.float)
            #     for i in range(inputs.size(0)):
            #         indices_to_zero = torch.randperm(inputs.size(1))[:int(inputs.size(1)*args.mask_rate)]
            #         mask[i,indices_to_zero] = 0.0
            #     masked_inputs = inputs*mask

            if args.model in ['VAE','BetaVAE']:
                recon, mu, logvar, _ = model(inputs)
                loss, recon_loss, reg_loss, _ = model.loss(inputs, recon, mu, logvar)
            elif args.model in ['WAE','weightWAE']:
                # if args.ismask=='Yes':
                #     recon, z = model(masked_inputs)
                # else:
                recon, z = model(inputs)
                loss, recon_loss, reg_loss, _ = model.loss(inputs, recon, z)
            else:
                raise Warning('no this model.')

            # loss_rec += loss.item()
            # recon_loss_rec += recon_loss.item()
            # reg_loss_rec += reg_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if isTest and ((epoch+1)%5==0 or epoch<5):
            # loss_rec = loss_rec/batch_num
            # recon_loss_rec = recon_loss_rec/batch_num
            # reg_loss_rec = reg_loss_rec/batch_num
            # logger.info(f'E{epoch} / loss:{loss_rec:.3f} / recon_loss:{recon_loss_rec:.3f} / reg_loss:{reg_loss_rec:.3f}')

            dis, label, score, normal_loss, all_loss = choose_test(model, testset, args.model)
            auc_cur, aupr, fpr95 = metrics_value(label, dis)
            logger.info(f'E{epoch} / auc:{auc_cur:.3f} / aupr:{aupr:.3f} / fpr95:{fpr95:.3f} / norm:{normal_loss:.3f} / all:{all_loss:.3f}')
            print(f'E{epoch} / auc:{auc_cur:.3f} / norm:{normal_loss:.3f} / all:{all_loss:.3f}')

            metrics.append([auc_cur,aupr,fpr95,normal_loss,all_loss])

        if isTest and ((epoch+1)%10==0 or epoch<5):
            torch.save(model.state_dict(), f'./checkpoint/{args.dataset}-{args.anomaly_rate}-{args.model}-{args.model_size}-{epoch}.pth')
    
    return metrics

def DSVDD_train(model, optimizer, trainset, testset, logger, epochs, args, isTest=True):
    batch_size = args.batch_size

    # batch_num = len(trainset)/batch_size
    metrics = []

    fixed_center = find_center(model, trainset, 'DSVDD')
    for epoch in range(epochs):
        model.train()
        # loss_rec = 0
        dataloader = generate_data(trainset, batch_size=batch_size, is_shuffle=True)
        for _,data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            z = model(inputs)
            loss,_ = model.loss(z, fixed_center)

            # loss_rec += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if isTest and ((epoch+1)%5==0 or epoch<5):
            # loss_rec = loss_rec/batch_num
            # logger.info(f'E{epoch} / loss:{loss_rec}')

            _, dis, label, normal_loss, loss = DSVDD_test(model, testset, fixed_center)
            auc_cur, aupr, fpr95 = metrics_value(label, dis)
            logger.info(f'E{epoch} / auc:{auc_cur:.3f} / aupr:{aupr:.3f} / fpr95:{fpr95:.3f} / norm:{normal_loss:.3f} / all:{loss:.3f}')
            print(f'E{epoch} / auc:{auc_cur:.3f} / norm:{normal_loss:.3f} / all:{loss:.3f}')
            
            metrics.append([auc_cur,aupr,fpr95,normal_loss,loss])

        if isTest and ((epoch+1)%10==0 or epoch<5):
            model_params = model.state_dict()
            model_params['center'] = fixed_center
            torch.save(model_params, f'./checkpoint/{args.dataset}-{args.anomaly_rate}-{args.model_size}-{args.model}-{epoch}.pth')
            
    return metrics

def DAGMM_train(model, optimizer, trainset, testset, logger, epochs, args, isTest=True):
    batch_size = args.batch_size

    # batch_num = len(trainset)/batch_size
    metrics = []

    for epoch in range(epochs):
        model.train()
        # loss_rec = 0
        dataloader = generate_data(trainset, batch_size=batch_size, is_shuffle=True)
        for _,data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            _, recon, z, gamma = model(inputs)
            loss, _, _, _ = model.loss(inputs, recon, z, gamma)

            # loss_rec += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if isTest and ((epoch+1)%5==0 or epoch<5):
            # loss_rec = loss_rec/batch_num
            # logger.info(f'E{epoch} / loss:{loss_rec:.3f}')

            _, Energy, label, normal_loss, loss = DAGMM_test(model, testset)
            auc_cur, aupr, fpr95 = metrics_value(label, Energy)
            logger.info(f'E{epoch} / auc:{auc_cur:.3f} / aupr:{aupr:.3f} / fpr95:{fpr95:.3f} / norm:{normal_loss:.3f} / all:{loss:.3f}')
            print(f'E{epoch} / auc:{auc_cur:.3f} / norm:{normal_loss:.3f} / all:{loss:.3f}')
            
            metrics.append([auc_cur,aupr,fpr95,normal_loss,loss])

        if isTest and ((epoch+1)%10==0 or epoch<5):
            torch.save(model.state_dict(), f'./checkpoint/{args.dataset}-{args.anomaly_rate}-{args.model_size}-{args.model}-{epoch}.pth')

    return metrics

'''
def VAESVDD_train(model, optimizer, trainset, testset, logger, epochs, batch_size, model_size, split_rate, isTest=True):
    batch_num = len(trainset)/batch_size
    metrics = []

    for epoch in range(epochs):
        model.train()
        loss_rec = 0
        dataloader = generate_data(trainset, batch_size=batch_size, is_shuffle=True)
        for _,data in enumerate(dataloader):
            inputs, _ = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            recon, mu, logvar, z = model(inputs)
            trainset_sample = random.sample(trainset, 10000)
            center = find_center(model, trainset_sample, 'VAESVDD')
            loss, recon_loss, reg_loss, dis_loss = model.loss(inputs, recon, mu, logvar, z, center)

            loss_rec += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if isTest and (epoch+1)%1==0:
            loss_rec = loss_rec/batch_num
            logger.info(f'E{epoch} / loss:{loss_rec:.3f}')

            center = find_center(model, trainset, 'VAESVDD')
            _, dis, label, normal_loss, loss = VAESVDD_test(model, testset, center)
            # sorted_score = find_VAESVDD_split(model, trainset, center, split_rate=split_rate)
            # auc_cur, f1_cur, recall_cur, precision_cur = metrics_value(label, dis, sorted_score)
            auc_cur = auc_value(label, dis)
            logger.info(f'E{epoch} / auc:{auc_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')
            print(f'E{epoch} / auc:{auc_cur:.3f} / norm_loss:{normal_loss:.3f} / loss:{loss:.3f}')

            model_params = model.state_dict()
            model_params['center'] = center
            torch.save(model_params, f'./checkpoint/VAESVDD-{model_size}-{epoch}-0%.pth')

            metrics.append([auc_cur,loss_rec,normal_loss,loss])
    return metrics
'''

'''
def FineTune(model, clt_model_list, args, round):
    
    x = math.floor(round/5)+1
    epochs = 4
    samples_num = 16
    
    # normal fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4*(min(x,10)))
    # generate samples
    trainset = []
    for i in range(len(clt_model_list)):
        samples = clt_model_list[i].sample(samples_num, device)
        for sample in samples:
            trainset.append([sample,0])
    
    # train with generated samples
    # server_train(model, optimizer, trainset, None, serlogger, epochs=epochs, batch_size=80, model_name=args.model, split_rate=args.norm_rate, isTest=False)
    

    # distilation fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # generate latent samples
    clt_dis_list = np.array([samples_num*dis for dis in clt_dis_list], dtype=int)
    clt_dis_list = np.cumsum(clt_dis_list)

    for epoch in range(epochs):
        optimizer.zero_grad()
        running_loss = 0.0

        z = torch.randn(samples_num, model.latent_size).to(device)
        t_recon = [clt_model_list[i].decoder(z[(clt_dis_list[i-1] if i>0 else 0):
                                               (clt_dis_list[i] if i+1<len(clt_dis_list) else samples_num)]) for i in range(len(clt_dis_list))]
        t_recon = torch.cat(t_recon, dim=0).detach()
        s_recon = model.decoder(z)
        # print(t_recon.shape, s_recon.shape)

        loss = F.mse_loss(s_recon, t_recon, reduction='none').sum(dim=-1).mean(dim=0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
        optimizer.step()

        running_loss = loss.item()
        
        serlogger.info(f'Fine-tune epoch: {round*args.epochs+epoch} | dis Loss: {running_loss:.2f}')
    

def VQVAE_FineTune(model, clt_model_list, emb_weight, args, round):
    
    x = math.floor(round/5)+1
    epochs = 4
    samples_num = 16
    
    # normal fine tune
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4*(min(x,10)))
    # generate samples
    trainset = []
    for i in range(len(clt_model_list)):
        samples = clt_model_list[i].sample(emb_weight, samples_num, device)
        for sample in samples:
            trainset.append([sample,0])
    
    # train with generated samples
    server_train(model, optimizer, trainset, None, serlogger, epochs=epochs, batch_size=80, model_name=args.model, split_rate=args.norm_rate, isTest=False)
    '''