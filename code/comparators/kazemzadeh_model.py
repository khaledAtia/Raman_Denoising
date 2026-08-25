"""
Cascaded U-Net baseline comparator (Kazemzadeh et al., Anal. Chem. 2022, 94, 12907-12918).

Implements the cascaded-U-Net architecture of that paper (their Figure 3(b2)) so that it
can be trained on OUR synthetic engine and OUR empirical noise, and placed head-to-head
against the dual-branch model on the identical validation set. Answers reviewer comments
R1.2 / R2.4 / R3.7 (comparison against a learned method), and specifically against the
cascaded arrangement our shared-encoder design is argued to improve upon.

WHY THE CASCADED U-NET, AND NOT THE ResNet-latent VARIANT
    Kazemzadeh et al. present several architectures. Their best single model is a
    ResNet autoencoder with a latent-layer auxiliary output (their Fig. 3(a3)). We
    deliberately reproduce the cascaded U-NET (their Fig. 3(b2)) instead, for one reason:
    our model is itself a U-Net, so a U-Net-vs-U-Net comparison isolates the single
    variable the reviewers care about -- two independent networks in sequence
    (cascade, two forward passes) versus one shared encoder with two heads (joint) --
    without confounding it with a ResNet-vs-U-Net difference. This is the fair test of the
    paper's central claim.

FAITHFULNESS TO THE SOURCE
    Reproduced exactly as described in the paper text:
      - two U-Nets in sequence (Fig. 3(b2));
      - an INTERMEDIATE output between them, supervised on a "only baseline corrected"
        target, i.e. signal + noise with the baseline removed (paper, p.12912);
      - the FINAL output supervised on the clean signal;
      - 1D strided-convolution downsampling ("strides ... shrink the feature map width
        ... by a factor of 2", p.12912), batch normalisation ("we have also used batch
        normalization layers", p.12908), and L2 weight regularisation
        (regularization parameter 0.001, p.12909, applied here as optimiser weight decay).
    Adapted where the source is unreadable or setup-specific:
      - the exact per-layer filter counts are given only in the paper's Supplementary
        Figures 1-4, which are not machine-readable. We therefore use the SAME depth and
        channel schedule as our own encoder (64-128-256-512 over three stride-2 stages,
        864 -> 108), so that each stage of the cascade has capacity comparable to our
        model and the comparison is not skewed by width. The cascade's total parameter
        count is roughly twice a single U-Net, which is the honest and intended cost of a
        two-network pipeline.
      - implemented in PyTorch rather than the original TensorFlow; the architecture, not
        the framework, is what the comparison rests on.
      - trained on OUR data (RamanDataGenerator: empirical baseline and noise banks,
        pseudo-Voigt peaks) rather than the paper's Gaussian-peak / polynomial-baseline
        engine, because a head-to-head requires an identical training distribution.

    Predicted baseline (for a baseline-MAE, not used in the headline signal comparison):
    baseline = input - intermediate, since the intermediate is the baseline-corrected
    spectrum in the input's own amplitude frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv1d -> BN -> ReLU) x 2, the standard U-Net feature block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Stride-2 convolution (halves length, as in the paper) then a DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.reduce(x))


class Up(nn.Module):
    """Transposed-conv upsample, concatenate the encoder skip, then a DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:                 # guard odd lengths
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diff // 2, diff - diff // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet1D(nn.Module):
    """
    A single 1D U-Net stage. Depth and widths match our own encoder so that each stage of
    the cascade is capacity-comparable to the dual-branch model.
        864 -> 432 -> 216 -> 108 (bottleneck)  ->  216 -> 432 -> 864
    """
    def __init__(self, in_ch=1, out_ch=1, widths=(64, 128, 256, 512)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.inc = DoubleConv(in_ch, w0)
        self.down1 = Down(w0, w1)
        self.down2 = Down(w1, w2)
        self.down3 = Down(w2, w3)
        self.up1 = Up(w3, w2, w2)
        self.up2 = Up(w2, w1, w1)
        self.up3 = Up(w1, w0, w0)
        self.outc = nn.Conv1d(w0, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        return self.outc(y)


class CascadedUNet(nn.Module):
    """
    Kazemzadeh et al. Fig. 3(b2): two U-Nets in sequence.

        raw  --[U-Net 1]--> intermediate (baseline-corrected)  --[U-Net 2]--> clean signal

    forward() returns (final_signal, intermediate, predicted_baseline).
      - intermediate is supervised on the baseline-corrected target (signal + noise);
      - final_signal is supervised on the clean signal;
      - predicted_baseline = raw - intermediate/sig_scale, the baseline the first stage
        has implicitly inferred; scored against the true baseline in SI Table S12.

    Two full forward passes are performed, with no shared parameters between the stages --
    this is exactly the property the joint dual-branch model is claimed to improve upon,
    and it is preserved here on purpose.
    """
    def __init__(self, widths=(64, 128, 256, 512), sig_scale=10.0):
        super().__init__()
        self.unet_baseline = UNet1D(1, 1, widths)   # raw -> baseline-corrected
        self.unet_denoise = UNet1D(1, 1, widths)    # baseline-corrected -> clean
        self.sig_scale = sig_scale                  # matches train_kazemzadeh.SIG_SCALE

    def forward(self, x):
        intermediate = self.unet_baseline(x)        # baseline-corrected (still noisy)
        final_signal = self.unet_denoise(intermediate)
        # the intermediate is supervised on (x - baseline) * sig_scale, so undo that
        # scaling before subtracting or the baseline comes out an order of magnitude wrong
        predicted_baseline = x - intermediate / self.sig_scale
        return final_signal, intermediate, predicted_baseline
