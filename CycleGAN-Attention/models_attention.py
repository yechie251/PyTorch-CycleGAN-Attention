
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Self-Attention Block
# --------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 4, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        N = W * H
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # [B, N, C//4]
        proj_key   = self.key_conv(x).view(B, -1, N)                     # [B, C//4, N]
        energy     = torch.bmm(proj_query, proj_key)                     # [B, N, N]
        attention  = F.softmax(energy, dim=-1)                           # [B, N, N]
        proj_value = self.value_conv(x).view(B, -1, N)                   # [B, C, N]
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))   # [B, C, N]
        out        = out.view(B, C, W, H)
        return self.gamma * out + x


# --------------------------
# Residual Block
# --------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# --------------------------
# Generator with Attention
# --------------------------
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6): 
        super(Generator, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
            
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        model += [SelfAttention(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# --------------------------
# Discriminator with SpectralNorm
# --------------------------
def spectral_conv(in_ch, out_ch, k=4, s=2, p=1):
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p))

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [
            spectral_conv(input_nc, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_conv(256, 512, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1)  # PatchGAN output
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
