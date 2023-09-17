import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQVAE(nn.Module):
    """VQ-VAE"""
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim, embedding_dim, num_embeddings, beta=0.25, loss_mode='mse'):
        super(VQVAE, self).__init__()

        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.loss_mode = loss_mode
        self.mmin = -100
        
        self.encoder_forward = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, latent_dim),
        )
        self.vq_layer = VectorQuantizer(embedding_dim, num_embeddings, beta)
        
        self.decoder_forward = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim3),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim1, in_dim),
            nn.Sigmoid()
        )
    
    def encoder(self, x):
        z = self.encoder_forward(x)
        return z
    
    def decoder(self, e):
        recon = self.decoder_forward(e)
        return recon

    def forward(self, x):
        z = self.encoder(x)
        e, vq_loss = self.vq_layer(z)
        recon = self.decoder(e)
        
        # recon_loss = torch.mean(torch.square(x-recon).sum(dim=1))
        if self.loss_mode=='mse':
            recon_loss = F.mse_loss(recon,x,reduction='none').sum(dim=-1)
        elif self.loss_mode=='bce':
            recon_loss = F.binary_cross_entropy(recon,x,reduction='none')
            recon_loss = torch.clamp(recon_loss, min=self.mmin).sum(dim=-1)
        else:
            raise ValueError('Undefined loss mode.')
        recon_loss = recon_loss.mean(dim=0)

        loss = recon_loss + vq_loss

        return z, e, recon, loss, recon_loss, vq_loss
    
    def get_label(self, x):
        z = self.encoder(x)
        z_shape = z.shape
        B = z_shape[0]
        CD = z_shape[1]
        reshaped_z = torch.reshape(z, (B*(int)(CD/self.embedding_dim), self.embedding_dim))    # [B x CD] -> [BC x D]

        dist = torch.sum(reshaped_z ** 2, dim=1, keepdim=True) + \
               torch.sum(self.vq_layer.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(reshaped_z, self.vq_layer.embedding.weight.t())  # [BC x K]
        
        encoding_inds = torch.argmin(dist, dim=1)  # [BC]

        return encoding_inds.detach().cpu().numpy().copy()

    def sample(self,p,num_samples,device):
        sampled_indices = np.random.choice(np.arange(len(p)), size=int(self.latent_dim/self.embedding_dim)*num_samples, p=p)

        onehot_encoded = torch.from_numpy(np.eye()[sampled_indices]).to(device)
        quantized_latents = torch.matmul(onehot_encoded, self.vq_layer.embedding.weight)  # [BC, D]
        quantized_latents = quantized_latents.view(num_samples, self.latent_dim)  # [B, CD]

        recon = self.decoder(quantized_latents) # [B, X]

        return recon.detach().cpu().numpy().copy()

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, embedding_dim, num_embeddings, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.D = embedding_dim
        self.K = num_embeddings
        self.beta = beta
        
        # initialize embeddings
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        
    def forward(self, latents, isN=True):
        latents_shape = latents.shape
        if isN == True:
            B = latents_shape[0]
            CD = latents_shape[1]
            reshaped_latents = torch.reshape(latents, (B*(int)(CD/self.D), self.D))    # [B x CD] -> [BC x D]
        else:
            reshaped_latents = latents

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(reshaped_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(reshaped_latents, self.embedding.weight.t())  # [BC x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BC, 1]

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K).to(device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BC x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BC, D]
        if isN:
            quantized_latents = quantized_latents.view(latents_shape)  # [B x CD]

        # Compute the VQ Losses
        # commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        commitment_loss = torch.mean(torch.square(quantized_latents.detach()-latents).sum(dim=1))
        # embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        embedding_loss = torch.mean(torch.square(quantized_latents-latents.detach()).sum(dim=1))

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        # 这里是为了传导梯度到x
        quantized_latents = latents + (quantized_latents - latents).detach()
        
        return quantized_latents, vq_loss