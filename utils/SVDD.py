from utils.LoadData import *
from utils.Params import *

def find_center(model, dataset, model_name):
    model.eval()

    center = None
    dataloader = generate_data(dataset, batch_size=1024, is_shuffle=False)
    for _, data in enumerate(dataloader):
        inputs, _ = data
        inputs = inputs.to(device)
        if model_name == 'DSVDD':
            z = model(inputs)
        elif model_name == 'VAESVDD':
            _, _, _, z = model(inputs)

        if center==None:
            center = z
        else:
            center = torch.cat((center,z),dim=0)
    
    return center.mean(dim=0).detach()

def find_DSVDD_split(model, dataset, c, split_rate):
    model.eval()

    samples_dis = None

    dataloader = generate_data(dataset, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, _ = data
        inputs = inputs.to(device)

        z = model(inputs)
        _,dis = model.loss(z,c)

        if i==0:
            samples_dis = dis.detach().cpu().numpy().copy()
        else:
            samples_dis = np.concatenate((samples_dis, dis.detach().cpu().numpy().copy()), axis=0)
    
    sorted_samples_dis = np.sort(samples_dis)
    # return sorted_samples_dis
    percentile_index = int(samples_dis.shape[0]*split_rate)
    return sorted_samples_dis[percentile_index]

def find_VAESVDD_split(model, dataset, c, split_rate):
    model.eval()

    samples_dis = None

    dataloader = generate_data(dataset, is_shuffle=False)
    for i, data in enumerate(dataloader):
        inputs, _ = data
        inputs = inputs.to(device)

        z = model(inputs)
        _, _, _, z = model(inputs)
        dis = torch.sum((z-c)**2, dim=1)

        if i==0:
            samples_dis = dis.detach().cpu().numpy().copy()
        else:
            samples_dis = np.concatenate((samples_dis, dis.detach().cpu().numpy().copy()), axis=0)
    
    sorted_samples_dis = np.sort(samples_dis)
    # return sorted_samples_dis
    percentile_index = int(samples_dis.shape[0]*split_rate)
    return sorted_samples_dis[percentile_index]