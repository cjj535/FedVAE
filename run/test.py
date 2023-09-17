import numpy as np
from utils.LoadData import *
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score
from utils.Params import *

def choose_test(model, testset, model_name):
    if model_name == 'VAE' or model_name == 'BetaVAE':
        _, dis, _, label, normal_loss, loss = VAE_test(model, testset)
    elif model_name == 'WAE':
        _, dis, _, label, normal_loss, loss = WAE_test(model, testset)
    else:
        raise Warning('others are not implemented.')

    return dis, label, normal_loss, loss

def choose_test_all(model, testset, model_name):
    if model_name == 'VAE' or model_name == 'BetaVAE':
        z, dis, cos, label, normal_loss, loss = VAE_test(model, testset)
    elif model_name == 'WAE':
        z, dis, cos, label, normal_loss, loss = WAE_test(model, testset)
    else:
        raise Warning('others are not implemented.')

    return z, dis, cos, label, normal_loss, loss

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
    normal_samples_loss = 0.0
    samples_loss = 0.0

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, mu, logvar, z = model(inputs)

        loss,_,_ = model.loss(inputs, recon, mu, logvar)
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
    
    # normal samples loss
    normal_dataset = [sample for sample in testset if sample[1]==0]
    dataloader = generate_data(normal_dataset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, mu, logvar, z = model(inputs)
        loss, _, _ = model.loss(inputs, recon, mu, logvar)
        normal_samples_loss += loss.item()
    
    # print(sample_Z.shape, sample_label.shape)
    samples_loss = samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_recon, samples_cos, samples_label, normal_samples_loss, samples_loss

def WAE_test(model, testset, batch_size=1024):
    model.eval()

    samples_Z = None
    samples_recon = None
    samples_label = None
    samples_cos = None
    samples_loss = 0.0
    normal_samples_loss = 0.0

    feature = None

    dataloader = generate_data(testset, batch_size=batch_size, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)

        recon, z = model(inputs)

        loss,_,_ = model.loss(inputs, recon, z)
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

        recon, z = model(inputs)
        loss, _, _ = model.loss(inputs, recon, z)
        normal_samples_loss += loss.item()

    #     if i == 0:
    #         feature = (inputs.detach().cpu().numpy().copy() - recon.detach().cpu().numpy().copy())**2
    #     else:
    #         feature = np.concatenate((feature, ((inputs.detach().cpu().numpy().copy() - recon.detach().cpu().numpy().copy())**2)), axis=0)
    # feature = np.mean(feature, axis=0)
    # print(feature.shape, feature)
    # import matplotlib.pyplot as plt
    # plt.plot(list(range(0,213,1),feature))
    # plt.show()

    # print(sample_Z.shape, sample_label.shape)
    samples_loss = samples_loss/(len(testset)/batch_size)
    normal_samples_loss = normal_samples_loss/(len(normal_dataset)/batch_size)
    return samples_Z, samples_recon, samples_cos, samples_label, normal_samples_loss, samples_loss

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

def auc_value(label, dis):
    label[label>0] = 1

    mmin = np.min(dis)
    mmin = min(0.0,mmin)
    mmax = np.max(dis)

    normalized_dis = (dis-mmin)/(mmax-mmin)
    auc_score = roc_auc_score(label, normalized_dis)
    return auc_score

def metrics_value(label, dis, split):
    # 0 or 1
    label[label>0] = 1
    
    # normalize
    mmin = np.min(dis)
    mmin = min(0,mmin)
    mmax = np.max(dis)

    normalized_dis = (dis-mmin)/(mmax-mmin)
    auc_score = roc_auc_score(label, normalized_dis)
    
    # predict label
    pred = np.copy(dis)
    pred[dis>split] = 1
    pred[dis<=split] = 0
    recall = recall_score(y_true=label, y_pred=pred)
    precision = precision_score(y_true=label, y_pred=pred)
    f1 = f1_score(y_true=label, y_pred=pred)

    return auc_score, f1, recall, precision