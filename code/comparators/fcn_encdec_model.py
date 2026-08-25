"""Fully convolutional encoder-decoder denoiser, after Loc et al. (2022).

Loc, Kecoglu, Unlu & Parlatan, "Denoising Raman spectra using fully convolutional
encoder-decoder network", J. Raman Spectrosc. 53 (2022) 1445-1452.

This is an implementation of the architecture *family* that work describes -- a
fully convolutional encoder-decoder that compresses the spectrum through a
bottleneck and reconstructs it, with no skip connections between the contracting
and expanding paths -- and not a reproduction of the published hyperparameters,
which are not restated here. Depth and width were chosen to bring the parameter
count close to that of the dual-branch network so that the comparison is not
confounded by capacity; the absence of skip connections is the architectural
property being represented and is preserved.

Its purpose in this work is to provide a third structure alongside the
dual-branch network and the cascaded U-Net: one that regresses the clean Raman
signal directly from the measurement in a single pass, without ever forming an
explicit baseline estimate.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolutions at constant resolution, group-normalised."""

    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FCNEncoderDecoder(nn.Module):
    """Encoder -> bottleneck -> decoder, fully convolutional, no skips.

    forward() returns the recovered Raman signal only. The network never
    produces a baseline estimate, which is the point of the comparison.
    """

    def __init__(self, widths=(64, 128, 256, 512), kernel=7):
        super().__init__()
        w1, w2, w3, w4 = widths

        self.enc1 = ConvBlock(1, w1, kernel)
        self.enc2 = ConvBlock(w1, w2, kernel)
        self.enc3 = ConvBlock(w2, w3, kernel)
        self.pool = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock(w3, w4, kernel)

        self.up3 = nn.ConvTranspose1d(w4, w3, 2, stride=2)
        self.dec3 = ConvBlock(w3, w3, kernel)
        self.up2 = nn.ConvTranspose1d(w3, w2, 2, stride=2)
        self.dec2 = ConvBlock(w2, w2, kernel)
        self.up1 = nn.ConvTranspose1d(w2, w1, 2, stride=2)
        self.dec1 = ConvBlock(w1, w1, kernel)

        self.out = nn.Conv1d(w1, 1, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)

        x = self.bottleneck(x)

        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.out(x)


if __name__ == "__main__":
    m = FCNEncoderDecoder()
    n = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(2, 1, 864))
    print(f"parameters : {n:,}")
    print(f"in  (2,1,864) -> out {tuple(y.shape)}")
