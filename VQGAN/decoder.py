import torch.nn as nn
from . import helper


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  helper.ResidualBlock(in_channels, in_channels),
                  helper.NonLocalBlock(in_channels),
                  helper.ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(helper.ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(helper.NonLocalBlock(in_channels))
            if i != 0:
                layers.append(helper.UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(helper.GroupNorm(in_channels))
        layers.append(helper.Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

