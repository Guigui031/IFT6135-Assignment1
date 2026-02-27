"""
Problem 4, Question 1 — UNet without skip connections.

Architecture is identical to UNet (same encoder/decoder channel widths)
but skip connections are removed: each decoder block receives only the
upsampled feature map from the block below, not the concatenated encoder
feature map.  This turns the network into a plain encoder-decoder
(autoencoder-style bottleneck architecture).
"""

import torch
from torch import nn
from unet import double_conv_block


class DecoderBlockNoSkip(nn.Module):
    """
    Decoder block without skip connection.
    upconv halves spatial channels, then double_conv refines.
    Channel flow: in_channels -> out_channels (no concat).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv   = double_conv_block(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.upconv(x))


class UNetNoSkip(nn.Module):
    """
    UNet encoder-decoder without skip connections.
    Encoder and decoder channel widths match the standard UNet so that
    the comparison is controlled: the only variable is the presence of
    skip connections.
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()
        in_channels = input_shape

        # ── Encoder (identical to UNet) ───────────────────────────────────
        self.encoder_block1 = double_conv_block(in_channels, 64)
        self.encoder_block2 = double_conv_block(64,  128)
        self.encoder_block3 = double_conv_block(128, 256)
        self.encoder_block4 = double_conv_block(256, 512)
        self.encoder_block5 = double_conv_block(512, 1024)  # bottleneck
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Decoder (no skip: in_channels halved, no concat) ──────────────
        self.decoder_block1 = DecoderBlockNoSkip(1024, 512)
        self.decoder_block2 = DecoderBlockNoSkip(512,  256)
        self.decoder_block3 = DecoderBlockNoSkip(256,  128)
        self.decoder_block4 = DecoderBlockNoSkip(128,  64)

        # ── Output ────────────────────────────────────────────────────────
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder — feature maps are NOT reused
        x = self.encoder_block1(x)
        x = self.encoder_block2(self.pool(x))
        x = self.encoder_block3(self.pool(x))
        x = self.encoder_block4(self.pool(x))
        x = self.encoder_block5(self.pool(x))  # bottleneck

        # Decoder — plain upsampling, no skip concat
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)

        return self.outconv(x)
