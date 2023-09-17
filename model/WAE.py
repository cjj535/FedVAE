import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class WAE(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, latent_size, weight=0.25,loss_mode='mse'):
        super(WAE, self).__init__()

        self.latent_size = latent_size
        self.weight = weight
        self.var = 2
        self.C = 2
        self.kernel_type = 'rbf'
        self.loss_mode = loss_mode
        self.mmin = -100

        self.encoder_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, latent_size)
        )

        self.decoder_forward = nn.Sequential(
            nn.Linear(latent_size, hidden_size3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        )
    
    def encoder(self, x):
        z = self.encoder_forward(x)
        return z

    def decoder(self, z):
        recon = self.decoder_forward(z)
        return recon
    
    def compute_kernel(self, x1, x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result
    
    def compute_rbf(self, x1, x2):
        z_dim = x1.shape[-1]
        sigma = 2.0 * z_dim * self.var
        # result = torch.sum(torch.exp(-(torch.sum(torch.square(x1 - x2),dim=-1) / sigma)))
        result = torch.sum(torch.exp(-(torch.sum(torch.square(x1 - x2),dim=-1) / sigma)),dim=1)
        return result

    def compute_inv_mult_quad(self, x1, x2):
        z_dim = x1.shape[-1]
        C = 2.0 * z_dim * self.var
        kernel = C / (C + torch.sum(torch.square(x1 - x2), dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result
    
    def compute_mmd(self, z, N):
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)
        # prior_z = torch.rand_like(z) * 2 - 1

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(z, prior_z)

        # mmd = 1.0 / (N*(N-1)) * prior_z__kernel + \
        #       1.0 / (N*(N-1)) * z__kernel - \
        #       2.0 / (N**2) * priorz_z__kernel
        mmd = 1.0 / (N-1) * prior_z__kernel + \
              1.0 / (N-1) * z__kernel - \
              2.0 / (N) * priorz_z__kernel
        return mmd
    
    def loss(self, x, recon, z):
        # recon_loss = torch.mean(torch.square(x-recon).sum(dim=1))
        if self.loss_mode=='mse':
            recon_loss = F.mse_loss(recon,x,reduction='none').sum(dim=-1)
        elif self.loss_mode=='bce':
            recon_loss = F.binary_cross_entropy(recon,x,reduction='none')
            recon_loss = torch.clamp(recon_loss, min=self.mmin).sum(dim=-1)
        else:
            raise ValueError('Undefined loss mode.')
        # recon_loss = recon_loss.mean(dim=0)
        
        B = x.shape[0]
        mmd_loss = self.compute_mmd(z, B)
        
        loss = recon_loss + self.weight * mmd_loss

        # mid: 0.9, big: 0.8
        # quantile = torch.quantile(recon_loss.detach(), 0.9).item()
        # scaled_loss = (1-1/(1+torch.exp(-10*(recon_loss.detach()-quantile))))*loss
        scaled_loss = loss
        
        scaled_loss = scaled_loss.mean(dim=0)
        recon_loss = recon_loss.mean(dim=0)
        mmd_loss = mmd_loss.mean(dim=0)
        
        return scaled_loss, recon_loss, mmd_loss
    
    def forward(self, x):
        z = self.encoder(x)

        recon = self.decoder(z)
        
        return recon, z
    
    def sample(self,num_samples,device):
        z = torch.randn(num_samples,self.latent_size).to(device)
        filtered_z = z[z.max(dim=1).values <= 2]
        filtered_z = filtered_z[filtered_z.min(dim=1).values >= -2]
        samples = self.decoder(filtered_z)
        return samples.detach().cpu().numpy().copy()

    def sample_attack(self,num_samples,device):
        z = torch.randn(num_samples,self.latent_size).to(device)
        filtered_z = z[z.max(dim=1).values <= 3]
        filtered_z = filtered_z[filtered_z.min(dim=1).values >= -3]
        row_mask = (filtered_z.max(dim=1).values >= 2) | (filtered_z.min(dim=1).values <= -2)
        filtered_z = filtered_z[row_mask]
        samples = self.decoder(filtered_z)
        return samples.detach().cpu().numpy().copy()