import torch
import torch.nn as nn

class DSVDD(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, latent_size):
        super(DSVDD, self).__init__()

        self.latent_size = latent_size

        center = torch.zeros(latent_size)
        self.center = nn.Parameter(center, requires_grad=False)

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size1, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size3, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, latent_size, bias=False)
        )
    
    def dis(self, x, c):
        return torch.sum((x-c)**2, dim=-1)

    def loss(self, x, c):
        dis = self.dis(x,c)
        loss = dis.mean(dim=0)
        return loss, torch.sqrt(dis)
    
    def forward(self, x):
        z = self.fc(x)
        return z