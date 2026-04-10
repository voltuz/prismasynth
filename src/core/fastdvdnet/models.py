"""FastDVDnet model definition.
Vendored from https://github.com/m-tassano/fastdvdnet (MIT License, Matias Tassano).
"""
import torch
import torch.nn as nn


class CvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    def __init__(self, num_in_frames, out_ch):
        super().__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames * 4, num_in_frames * self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """Denoising block: 3-frame U-Net with residual learning."""

    def __init__(self, num_input_frames=3):
        super().__init__()
        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=32)
        self.downc0 = DownBlock(in_ch=32, out_ch=64)
        self.downc1 = DownBlock(in_ch=64, out_ch=128)
        self.upc2 = UpBlock(in_ch=128, out_ch=64)
        self.upc1 = UpBlock(in_ch=64, out_ch=32)
        self.outc = OutputCvBlock(in_ch=32, out_ch=3)

    def forward(self, in0, in1, in2, noise_map):
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        x = self.outc(x0 + x1)
        return in1 - x  # residual


class FastDVDnet(nn.Module):
    """Two-stage temporal denoiser using 5-frame windows.

    Input: [N, 15, H, W] (5 RGB frames concatenated) + [N, 1, H, W] noise map
    Output: [N, 3, H, W] denoised center frame
    """

    def __init__(self, num_input_frames=5):
        super().__init__()
        self.num_input_frames = num_input_frames
        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)

    def forward(self, x, noise_map):
        x0, x1, x2, x3, x4 = tuple(
            x[:, 3 * m:3 * m + 3, :, :] for m in range(self.num_input_frames)
        )
        # Stage 1: denoise overlapping triplets
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        # Stage 2: fuse into final denoised center frame
        return self.temp2(x20, x21, x22, noise_map)
