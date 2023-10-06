import torch.nn as nn
from .  import helper


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 128
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(helper.ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(helper.NonLocalBlock(in_channels))
            if i != len(channels)-2:
                layers.append(helper.DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(helper.ResidualBlock(channels[-1], channels[-1]))
        layers.append(helper.NonLocalBlock(channels[-1]))
        layers.append(helper.ResidualBlock(channels[-1], channels[-1]))
        layers.append(helper.GroupNorm(channels[-1]))
        layers.append(helper.Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)