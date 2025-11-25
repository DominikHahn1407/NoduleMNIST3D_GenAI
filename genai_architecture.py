import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE3D(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1), # -> (16, 14, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32,7,7,7)
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64,4,4,4)
            nn.ReLU(inplace=True),
        )
        self.enc_fc_mu = nn.Linear(64*4*4*4, latent_dim)
        self.enc_fc_logvar = nn.Linear(64*4*4*4, latent_dim)
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 64*4*4*4)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), # -> (32,8,8,8)
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1), # -> (16,16,16,16)
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, 4, stride=2, padding=1),  # -> (1,32,32,32)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 64, 4, 4, 4)
        x_hat = self.dec_deconv(h)
        # Crop from 32^3 to 28^3 (center crop) for compatibility
        x_hat = x_hat[:, :, 2:30, 2:30, 2:30]
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar