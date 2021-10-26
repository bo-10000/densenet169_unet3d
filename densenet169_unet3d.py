import torch
import torch.nn as nn
from torch.nn import functional as F
from densenet3d import densenet3d169

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=4, stride=2,
                            padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Densenet_Unet3D(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, num_filters=32, norm_layer='bn'):
        super().__init__()

        self.out_channel = out_channel

        encoder = densenet3d169(in_channel=in_channel, norm_layer=norm_layer)

        self.conv1 = encoder.conv1 # 64
        enc = encoder.features

        self.enc1 = nn.Sequential(enc[0], enc[1]) # 128
        self.enc2 = nn.Sequential(enc[2], enc[3]) # 256
        self.enc3 = nn.Sequential(enc[4], enc[5]) # 640
        self.enc4 = nn.Sequential(enc[6], enc[7], enc[8], nn.MaxPool3d(2, 2)) # 1664

        self.center = DecoderBlock(1664, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(640 + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64 + num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv3d(num_filters, out_channel, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        enc1 = self.enc1(conv1)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)

        dec4 = self.dec4(torch.cat([center, enc3], 1))
        dec3 = self.dec3(torch.cat([dec4, enc2], 1))
        dec2 = self.dec2(torch.cat([dec3, enc1], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.out_channel > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out
