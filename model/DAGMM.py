import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.Params import *

class DAGMM(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size):
        super(DAGMM, self).__init__()

        self.latent_size = hidden_size
        self.lambda1 = 0.1
        self.lambda2 = 0.005
        
        n_gmm = 4

        self.encoder_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.Tanh(),
            nn.Linear(hidden_size3, hidden_size),
            # nn.Tanh(),
            # nn.Linear(hidden_size3, self.latent_size)
        )

        self.decoder_forward = nn.Sequential(
            # nn.Linear(self.latent_size, hidden_size3),
            # nn.Tanh(),
            nn.Linear(hidden_size, hidden_size3),
            nn.Tanh(),
            nn.Linear(hidden_size3, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.Tanh(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        )

        self.estimation_forward = nn.Sequential(
            nn.Linear(self.latent_size+2, 100),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(100, n_gmm),
            nn.Softmax(dim=1),
        )

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,self.latent_size+2))
        self.register_buffer("cov", torch.zeros(n_gmm,self.latent_size+2,self.latent_size+2))
    
    def encoder(self, x):
        z = self.encoder_forward(x)
        return z

    def decoder(self, z):
        recon = self.decoder_forward(z)
        return recon
    
    def forward(self, x):
        z = self.encoder(x)

        recon = self.decoder(z)
        
        euclidean = torch.sqrt(torch.sum((x-recon)**2, dim=1))
        cosine = torch.sum(x*recon, dim=-1)/torch.sqrt(torch.sum(x**2+1e-4, dim=-1) * torch.sum(recon**2+1e-4, dim=-1))

        expand_z = torch.cat([z, euclidean.unsqueeze(-1), cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation_forward(expand_z)

        return z, recon, expand_z, gamma
    
    def compute_gmm_params(self, z, gamma):
        B = gamma.size(0)

        # B batch, D dimension of z, K mixture number
        # z: B x D; gamma: B x K; sum_gamma: K
        sum_gamma = torch.sum(gamma, dim=0)

        # phi: K
        phi = sum_gamma / B
        self.phi = phi.data

        # mu: K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data

        # z_mu = B x K x D
        z_mu = z.unsqueeze(1)- mu.unsqueeze(0)
        # z_mu_outer = B x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = self.phi.detach()
        if mu is None:
            mu = self.mu.detach()
        if cov is None:
            cov = self.cov.detach()

        k, D, _ = cov.size()

        # z: B x D; mu: K x D; z_mu: B x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        
        cov_inverse = []
        det_cov = []
        cov_diag = None
        for i in range(k):
            # cov_k: D x D; cov_inverse: 1 x D x D  
            cov_k = cov[i] + (torch.eye(D)*(1e-1)).to(device)        #矩阵可能不是正定的
            cov_inverse.append(torch.inverse(cov_k.clone()).unsqueeze(0))

            det_cov.append((torch.pow(torch.linalg.cholesky(cov_k).diag().prod(),2)*2*np.pi).unsqueeze(0))
            if cov_diag==None:
                cov_diag = torch.sum(1.0 / cov_k.diag())
            else:
                cov_diag = cov_diag + torch.sum(1.0 / cov_k.diag())
            
        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov)

        # B x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        
        # for stability (logsumexp)
        # exp_term: B x K
        # max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        # exp_term = torch.exp(exp_term_tmp - max_val)
        exp_term = torch.exp(exp_term_tmp)

        # sam_energy: B
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1)) #+ eps)
        sample_energy = - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + 1e-4)
    
        if size_average:
            sample_energy = torch.mean(sample_energy)
        
        return sample_energy, cov_diag
    
    def loss(self, x, recon, z, gamma):
        recon_error = torch.mean(torch.sum(torch.pow(x - recon,2),dim=-1))
        # print(recon_error.shape)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + self.lambda1 * sample_energy + self.lambda2 * cov_diag

        return loss, sample_energy, recon_error, cov_diag