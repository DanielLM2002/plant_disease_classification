import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.3),  # LeakyReLU with negative slope of 0.3
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.3),  # LeakyReLU with negative slope of 0.3
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        
        self.fc_mu = nn.Linear(8 * 56 * 56, 128)
        self.fc_logvar = nn.Linear(8 * 56 * 56, 128)

        self.fc = nn.Linear(128, 8 * 56 * 56)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LeakyReLU(0.3),  # LeakyReLU with negative slope of 0.3
            nn.ConvTranspose2d(16, 3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def forward(self, x):
        x = self.encoder(x)
        h = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)
        self.z = z
        
        h = self.fc(z)
        h = h.view(-1, 8, 56, 56)
        x_recon = self.decoder(h)
        return x_recon, mu, logvar
