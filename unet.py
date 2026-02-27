import torch
from torch import nn


def double_conv_block(in_channels, out_channels):
    """
    This double conv block are the blocks used in the encoder part of UNet.
    It uses a padding of 1 to preserve spatial dimensions.
    
    :param in_channels: Description
    :param out_channels: Description
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    """
    Decoder block of UNet. It consists of an upconvolution layer followed by a double conv block.
    Use the double_conv_block defined above.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample spatial dims by 2, halve channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concat with skip: out_channels (up) + out_channels (skip) = 2*out_channels
        self.conv = double_conv_block(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # concat along channel axis
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        in_channels = input_shape

        # Encoder
        self.encoder_block1 = double_conv_block(in_channels, 64)
        self.encoder_block2 = double_conv_block(64, 128)
        self.encoder_block3 = double_conv_block(128, 256)
        self.encoder_block4 = double_conv_block(256, 512)
        self.encoder_block5 = double_conv_block(512, 1024)  # bottleneck
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.decoder_block1 = DecoderBlock(1024, 512)
        self.decoder_block2 = DecoderBlock(512, 256)
        self.decoder_block3 = DecoderBlock(256, 128)
        self.decoder_block4 = DecoderBlock(128, 64)

        # Output
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(self.pool(e1))
        e3 = self.encoder_block3(self.pool(e2))
        e4 = self.encoder_block4(self.pool(e3))
        e5 = self.encoder_block5(self.pool(e4))  # bottleneck

        # Decoder
        d1 = self.decoder_block1(e5, e4)
        d2 = self.decoder_block2(d1, e3)
        d3 = self.decoder_block3(d2, e2)
        d4 = self.decoder_block4(d3, e1)

        return self.outconv(d4)