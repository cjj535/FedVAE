import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, latent_size, weight=1.0, loss_mode='mse'):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.weight = weight
        self.loss_mode = loss_mode
        self.mmin = -100

        self.encoder_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, latent_size * 2)
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
        out = self.encoder_forward(x)
        mu = out[:, :self.latent_size]
        logvar = out[:, self.latent_size:]
        return mu, logvar

    def decoder(self, z):
        recon = self.decoder_forward(z)
        return recon
    
    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        z = mu + eps * std
        return z
    
    def loss(self, x, recon, mu, logvar):
        if self.loss_mode=='mse':
            recon_loss = F.mse_loss(recon,x,reduction='none').sum(dim=-1)
        elif self.loss_mode=='bce':
            recon_loss = F.binary_cross_entropy(recon,x,reduction='none')
            recon_loss = torch.clamp(recon_loss, min=self.mmin).sum(dim=-1)
        else:
            raise ValueError('Undefined loss mode.')
        # recon_loss = recon_loss.mean(dim=0)
        
        kl_loss = 0.5 * (torch.square(mu) + logvar.exp() - 1 - logvar).sum(dim=1)

        loss = recon_loss + self.weight * kl_loss

        # quantile = torch.quantile(recon_loss.detach(), 0.90).item()
        # scaled_loss = (1-1/(1+torch.exp(-10*(recon_loss.detach()-quantile))))*loss
        scaled_loss = loss
        
        scaled_loss = scaled_loss.mean(dim=0)
        recon_loss = recon_loss.mean(dim=0)
        kl_loss = kl_loss.mean(dim=0)

        return scaled_loss, recon_loss, kl_loss, loss
    
    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z)
        
        return recon, mu, logvar, z

    # def sample(self,num_samples,device):
    #     z = torch.randn(num_samples,self.latent_size).to(device)
    #     samples = self.decoder(z)
    #     return samples.detach().cpu().numpy().copy()
    
    def sample(self,num_samples,device):
        z = torch.randn(num_samples,self.latent_size).to(device)
        filtered_z = z[z.max(dim=1).values <= 2]
        filtered_z = filtered_z[filtered_z.min(dim=1).values >= -2]
        samples = self.decoder(filtered_z)
        return samples.detach().cpu().numpy().copy()

    def sample_attack(self,num_samples,device):
        z = torch.randn(num_samples,self.latent_size).to(device)
        filtered_z = z[z.max(dim=1).values <= 10]
        filtered_z = filtered_z[filtered_z.min(dim=1).values >= -10]
        row_mask = (filtered_z.max(dim=1).values >= 4) | (filtered_z.min(dim=1).values <= -4)
        filtered_z = filtered_z[row_mask]
        samples = self.decoder(filtered_z)
        return samples.detach().cpu().numpy().copy()