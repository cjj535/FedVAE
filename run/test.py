import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from scipy.interpolate import interp1d
from utils.LoadData import *
from utils.Params import *

def choose_test(model, testset, model_name):
    if model_name in ['VAE','BetaVAE']:
        _, dis, _, label, loss, normal_loss, all_loss = VAE_test(model, testset)
    elif model_name in ['WAE','weightWAE']:
        _, dis, _, label, loss, normal_loss, all_loss = WAE_test(model, testset)
    else:
        raise Warning('others are not implemented.')

    return dis, label, loss, normal_loss, all_loss

def choose_test_all(model, testset, model_name):
    if model_name in ['VAE','BetaVAE']:
        z, dis, cos, label, loss, normal_loss, all_loss = VAE_test(model, testset)
    elif model_name in ['WAE','weightWAE']:
        z, dis, cos, label, loss, normal_loss, all_loss = WAE_test(model, testset)
    else:
        raise Warning('others are not implemented.')

    return z, dis, cos, label, loss, normal_loss, all_loss

def distance(x, y):
    return np.sum((x-y)**2, axis=1)

def cosine(x, y):
    return np.sum(x*y, axis=1)/(np.sum(x**2+eps, axis=1)*np.sum(y**2+eps, axis=1))

def VAE_test(model, testset, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_recon = None
    samples_label = None
    samples_cos = None
    samples_loss = None
    normal_samples_loss = 0.0
    all_samples_loss = 0.0

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, mu, logvar, z = model(inputs)

        _,_,_,loss = model.loss(inputs, recon, mu, logvar)
        all_samples_loss += (loss.mean(dim=0)).item()

        if i==0:
            samples_Z = z.detach().cpu().numpy().copy()
            samples_recon = distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
            samples_cos = cosine(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
            samples_label = labels.detach().cpu().numpy().copy()
            samples_loss = loss.detach().cpu().numpy().copy()
        else:
            samples_Z = np.concatenate((samples_Z, z.detach().cpu().numpy().copy()), axis=0)
            samples_label = np.concatenate((samples_label, labels.detach().cpu().numpy().copy()), axis=0)
            samples_recon = np.concatenate((samples_recon, distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
            samples_cos = np.concatenate((samples_cos, cosine(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
            samples_loss = np.concatenate((samples_loss, loss.detach().cpu().numpy().copy()), axis=0)
    
    # normal samples loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, mu, logvar, z = model(inputs)
        _,_,_,loss = model.loss(inputs, recon, mu, logvar)
        normal_samples_loss += (loss.mean(dim=0)).item()
    
    # print(sample_Z.shape, sample_label.shape)
    all_samples_loss = all_samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_recon, samples_cos, samples_label, samples_loss, normal_samples_loss, all_samples_loss

def WAE_test(model, testset, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_recon = None
    samples_label = None
    samples_cos = None
    samples_loss = None
    all_samples_loss = 0.0
    normal_samples_loss = 0.0

    feature = None

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, z = model(inputs)

        _,_,_,loss = model.loss(inputs, recon, z)
        all_samples_loss += (loss.mean(dim=0)).item()

        if i==0:
            samples_Z = z.detach().cpu().numpy().copy()
            samples_recon = distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
            samples_cos = cosine(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
            samples_label = labels.detach().cpu().numpy().copy()
            samples_loss = loss.detach().cpu().numpy().copy()
        else:
            samples_Z = np.concatenate((samples_Z, z.detach().cpu().numpy().copy()), axis=0)
            samples_label = np.concatenate((samples_label, labels.detach().cpu().numpy().copy()), axis=0)
            samples_recon = np.concatenate((samples_recon, distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
            samples_cos = np.concatenate((samples_cos, cosine(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
            samples_loss = np.concatenate((samples_loss, loss.detach().cpu().numpy().copy()), axis=0)
        
        # if i == 0:
        #     feature = (inputs.detach().cpu().numpy().copy() - recon.detach().cpu().numpy().copy())**2
        # else:
        #     feature = np.concatenate((feature, ((inputs.detach().cpu().numpy().copy() - recon.detach().cpu().numpy().copy())**2)), axis=0)
    
    # normal sample loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, z = model(inputs)
        _, _, _, loss = model.loss(inputs, recon, z)
        normal_samples_loss += (loss.mean(dim=0)).item()

    #     if i == 0:
    #         feature = (inputs.detach().cpu().numpy().copy() - recon.detach().cpu().numpy().copy())**2
    #     else:
    #         feature = np.concatenate((feature, ((inputs.detach().cpu().numpy().copy() - recon.detach().cpu().numpy().copy())**2)), axis=0)
    # bad_feature = feature[samples_recon>0.5]
    # bad_feature = np.mean(bad_feature, axis=0)
    # print(bad_feature)
    # good_feature = feature[samples_recon<0.2]
    # good_feature = np.mean(good_feature, axis=0)
    # print(good_feature)
    # print(bad_feature/good_feature)
    # import matplotlib.pyplot as plt
    # feature = np.mean(feature, axis=0)
    # plt.plot(list(range(0,202,1)),feature)
    # plt.savefig('feature.png')

    # print(sample_Z.shape, sample_label.shape)
    all_samples_loss = all_samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_recon, samples_cos, samples_label, samples_loss, normal_samples_loss, all_samples_loss

'''
def VQVAE_test(model, testset, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_recon = None
    samples_label = None
    samples_cos = None
    samples_loss = 0.0
    normal_samples_loss = 0.0

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        z, _, recon, loss, _, _ = model(inputs)

        samples_loss += loss.item()

        if i==0:
            samples_Z = z.detach().cpu().numpy().copy()
            samples_recon = distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
            samples_cos = cosine(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
            samples_label = labels.detach().cpu().numpy().copy()
        else:
            samples_Z = np.concatenate((samples_Z, z.detach().cpu().numpy().copy()), axis=0)
            samples_label = np.concatenate((samples_label, labels.detach().cpu().numpy().copy()), axis=0)
            samples_recon = np.concatenate((samples_recon, distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
            samples_cos = np.concatenate((samples_cos, cosine(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
    
    # normal sample loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        _, _, _, loss, _, _ = model(inputs)
        normal_samples_loss += loss.item()
    
    # print(sample_Z.shape, sample_label.shape)
    samples_loss = samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_recon, samples_cos, samples_label, normal_samples_loss, samples_loss
'''

def DSVDD_test(model, testset, c, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_dis = None
    samples_label = None
    normal_samples_loss = 0.0
    samples_loss = 0.0

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        z = model(inputs)

        loss, dis = model.loss(z, c)
        samples_loss += loss.item()

        if i==0:
            samples_Z = z.detach().cpu().numpy().copy()
            samples_dis = dis.detach().cpu().numpy().copy()
            samples_label = labels.detach().cpu().numpy().copy()
        else:
            samples_Z = np.concatenate((samples_Z, z.detach().cpu().numpy().copy()), axis=0)
            samples_label = np.concatenate((samples_label, labels.detach().cpu().numpy().copy()), axis=0)
            samples_dis = np.concatenate((samples_dis, dis.detach().cpu().numpy().copy()), axis=0)
    
    # normal samples loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        z = model(inputs)
        loss,_ = model.loss(z, c)
        normal_samples_loss += loss.item()
    
    # print(sample_Z.shape, sample_label.shape)
    samples_loss = samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_dis, samples_label, normal_samples_loss, samples_loss

'''
def VAESVDD_test(model, testset, c, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_dis = None
    samples_label = None
    normal_samples_loss = 0.0
    samples_loss = 0.0

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, mu, logvar, z = model(inputs)
        loss,_,_,_ = model.loss(inputs, recon, mu, logvar, z, c)
        samples_loss += loss.item()

        if i==0:
            samples_Z = z.detach().cpu().numpy().copy()
            samples_dis = torch.sqrt(torch.sum((z-c)**2,dim=-1)).detach().cpu().numpy().copy()
            samples_label = labels.detach().cpu().numpy().copy()
        else:
            samples_Z = np.concatenate((samples_Z, z.detach().cpu().numpy().copy()), axis=0)
            samples_dis = np.concatenate((samples_dis, torch.sqrt(torch.sum((z-c)**2,dim=-1)).detach().cpu().numpy().copy()), axis=0)
            samples_label = np.concatenate((samples_label, labels.detach().cpu().numpy().copy()), axis=0)
    
    # normal samples loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, mu, logvar, z = model(inputs)
        loss,_,_,_ = model.loss(inputs, recon, mu, logvar, z, c)
        normal_samples_loss += loss.item()
    
    # print(sample_Z.shape, sample_label.shape)
    samples_loss = samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_dis, samples_label, normal_samples_loss, samples_loss
'''

def DAGMM_test(model, testset, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_en = None
    samples_label = None
    normal_samples_loss = 0.0
    samples_loss = 0.0

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        _, recon, z, gamma = model(inputs)
        loss,_,_,_ = model.loss(inputs, recon, z, gamma)
        en,_ = model.compute_energy(z, size_average=False)
        samples_loss += loss.item()

        if i==0:
            samples_Z = z.detach().cpu().numpy().copy()
            samples_en = en.detach().cpu().numpy().copy()
            samples_label = labels.detach().cpu().numpy().copy()
        else:
            samples_Z = np.concatenate((samples_Z, z.detach().cpu().numpy().copy()), axis=0)
            samples_en = np.concatenate((samples_en, en.detach().cpu().numpy().copy()), axis=0)
            samples_label = np.concatenate((samples_label, labels.detach().cpu().numpy().copy()), axis=0)
    
    # normal samples loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        _, recon, z, gamma = model(inputs)
        loss,_,_,_ = model.loss(inputs, recon, z, gamma)
        normal_samples_loss += loss.item()
    
    # print(sample_Z.shape, sample_label.shape)
    samples_loss = samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_en, samples_label, normal_samples_loss, samples_loss

def find_split(model, dataset, model_name, split_rate):
    model.eval()

    samples_recon = None

    dataloader = generate_data(dataset, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        if model_name == 'VAE' or model_name == 'BetaVAE':
            recon, _, _, _ = model(inputs)
        elif model_name == 'WAE':
            recon, _ = model(inputs)
        else:
            raise Warning('no this model')

        if i==0:
            samples_recon = distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())
        else:
            samples_recon = np.concatenate((samples_recon, distance(inputs.detach().cpu().numpy().copy(), recon.detach().cpu().numpy().copy())), axis=0)
    
    sorted_samples_recon = np.sort(samples_recon)
    # return sorted_samples_recon
    percentile_index = int(samples_recon.shape[0]*split_rate)
    return sorted_samples_recon[percentile_index]

def find_DAGMM_split(model, dataset, split_rate):
    model.eval()
    samples_en = None

    dataloader = generate_data(dataset, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, _ = data
        inputs = inputs.to(device)

        _, _, z, _ = model(inputs)
        en,_ = model.compute_energy(z, size_average=False)

        if i==0:
            samples_en = en.detach().cpu().numpy().copy()
        else:
            samples_en = np.concatenate((samples_en, en.detach().cpu().numpy().copy()), axis=0)
    
    sorted_samples_en = np.sort(samples_en)
    # return sorted_samples_en
    percentile_index = int(samples_en.shape[0]*split_rate)
    return sorted_samples_en[percentile_index]

def auc_value(label, score):
    label[label>0] = 1

    mmin = np.min(score)
    mmin = min(0.0,mmin)
    mmax = np.max(score)

    normalized_score = (score-mmin)/(mmax-mmin)
    auc = roc_auc_score(label, normalized_score)
    return auc

def metrics_value(label, score):
    # 0 or 1
    label[label>0] = 1
    
    # normalize
    mmin = np.min(score)
    mmin = min(0,mmin)
    mmax = np.max(score)

    normalized_score = (score-mmin)/(mmax-mmin)
    auroc = roc_auc_score(label, normalized_score)
    
    precision, recall, _ = precision_recall_curve(label, normalized_score)
    aupr = auc(recall, precision)
    
    fpr,tpr,_ = roc_curve(label, normalized_score)
    target_tpr = 0.95
    interp_func = interp1d(tpr, fpr, kind='linear', fill_value='extrapolate')
    fpr95 = interp_func(target_tpr)
    
    return auroc, aupr, fpr95

# def metrics_value(label, dis, split):
#     # 0 or 1
#     label[label>0] = 1
    
#     # normalize
#     mmin = np.min(dis)
#     mmin = min(0,mmin)
#     mmax = np.max(dis)

#     normalized_dis = (dis-mmin)/(mmax-mmin)
#     auc_score = roc_auc_score(label, normalized_dis)
    
#     # predict label
#     pred = np.copy(dis)
#     pred[dis>split] = 1
#     pred[dis<=split] = 0
#     recall = recall_score(y_true=label, y_pred=pred)
#     precision = precision_score(y_true=label, y_pred=pred)
#     f1 = f1_score(y_true=label, y_pred=pred)

#     return auc_score, f1, recall, precision