import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== Residual Block ======
# ✅ Recovered ConvVAE structure based on your old state_dict

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class ConvVAE(nn.Module):
    def __init__(self, input_channels=3, latent_channels=32, num_classes=170):
        super().__init__()
        # Encoder: 3→112→224→448→448
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 112, 3, stride=2, padding=1),
            nn.BatchNorm2d(112), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(112, 224, 3, stride=2, padding=1),
            nn.BatchNorm2d(224), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(224, 448, 3, stride=2, padding=1),
            nn.BatchNorm2d(448), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(448, 448, 3, stride=1, padding=1),
            nn.BatchNorm2d(448), nn.LeakyReLU(0.2, inplace=True),
        )
        # 量化层
        self.quant_conv = nn.Conv2d(448, latent_channels * 2, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, 448, 1)

        # Decoder: 448→448→224→112→64
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(448, 448, 3, padding=1),
            nn.BatchNorm2d(448), nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(448),

            nn.Conv2d(448, 224, 3, padding=1),
            nn.BatchNorm2d(224), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(224),

            nn.Conv2d(224, 112, 3, padding=1),
            nn.BatchNorm2d(112), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(112),

            nn.Conv2d(112, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),
        )
        # refine 模块（保持不变）
        self.refine = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, input_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(latent_channels, num_classes)

    def preprocess(self, x):
        return 2 * x - 1

    def encode(self, x):
        x = self.preprocess(x)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mean + torch.randn_like(std) * std
        else:
            z = mean
        return z, mean, logvar

    def decode(self, z):
        h = self.post_quant_conv(z)
        h = self.decoder(h)
        return self.refine(h)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        z_pooled = self.pool(z).view(z.size(0), -1)
        logits = self.classifier(z_pooled)
        return x_recon, z, mean, logvar, logits



# ====== Custom VAE loss with log(mse * class) + beta * KL ======
def vae_loss(x, x_recon, z, mean, logvar, labels, classifier, beta=0.005, eps=1e-8):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    z_pooled = F.adaptive_avg_pool2d(z, (1, 1)).squeeze(-1).squeeze(-1)
    logits = classifier(z_pooled)
    class_loss = F.cross_entropy(logits, labels)
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    final_loss = torch.log(recon_loss + eps) + torch.log(class_loss + eps) + beta * kl
    return final_loss, recon_loss, class_loss
