import sys

import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F

from .ExtractParts import Encoder, channel_reduction
from .CBAM import CBAM
from .generate_mask_process import *

# class AFF(nn.Module) :
#     def __init__(self, in_channels, r=4):
#         super(AFF, self).__init__()
#
#         self.global_feature_score = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Linear(in_channels, in_channels // r),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // r, in_channels),
#             nn.Sigmoid()
#         )
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.global_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // r, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.squeeze = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        #se = self.squeeze(x)#.view(batch, channels)
        ge = self.global_extractor(x)#.view(batch, channels, 1, 1)
        se = self.sigmoid(ge)
        # se = self.squeeze(se)

        return x * se

class ExtractModel(nn.Module) :
    def __init__(self, args, device):
        super(ExtractModel, self).__init__()

        self.device = device
        self.image_size = args.image_size
        self.angle = args.angle
        self.length = args.length
        self.num_enc = args.num_enc
        self.preserve_range = args.preserve_range
        ratio = 16

        # self.idxx, self.idxy = get_small_region(self.image_size, self.angle, self.length, self.preserve_range)

        self.encs = nn.ModuleList([Encoder() for _ in range(self.num_enc)])
        # self.encs = Encoder()
        self.channel_reduction = channel_reduction(ratio=ratio)
        self.attention = CBAM(self.num_enc * int(512//ratio), 16)
        # self.test = SEBlock(self.num_enc * 256, 16)

    def forward(self, x):
        self.idxx, self.idxy = get_small_region(self.image_size, self.angle, self.length, self.preserve_range)
        patterns = self.extract(x)

        out = []
        test1 = []

        for i in range(self.num_enc):
            patternFeatureMap = self.encs[i](patterns[i].unsqueeze(1))
            # patternFeatureMap = self.encs(patterns[i].unsqueeze(1))
            out.append(self.channel_reduction(patternFeatureMap))

        out = torch.cat(out, dim=1)
        # test = self.test(out)
        # for i in range(self.num_enc) :
        #     test1.append(torch.mean(test[:, 256 * i : 256 * (i + 1), :, :], dim=1))
        # test1 = torch.cat(test1, dim=1).squeeze()
        # test1 = F.softmax(test1, dim=1)
        # out = torch.exp((test1 - torch.mean(test1, dim=1, keepdim=True)) / torch.std(test1, dim=1, keepdim=True))

        out = self.attention(out)
        # out = self.test(out)
        # out = F.sigmoid(out)
        out = nn.AdaptiveAvgPool2d(1)(out).squeeze()
        out = out.reshape(-1, self.num_enc, out.size()[1] // self.num_enc)
        out = torch.mean(out, dim=2)
        out = F.softmax(out, dim=1)# * self.num_enc
        # # print(out.shape)
        #
        out = torch.exp((out - torch.mean(out, dim=1, keepdim=True)) / torch.std(out, dim=1, keepdim=True))
        # out = (out - torch.mean(out, dim=1, keepdim=True)) / torch.std(out, dim=1, keepdim=True) + 1

        return out  # (?, number of encoder)

    def extract(self, x):
        x_fft = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        # x_spectrum = torch.log(1 + torch.abs(x_fft))
        x_fft_abs, x_fft_angle = torch.abs(x_fft), torch.angle(x_fft)
        x_spectrum = torch.log(1 + x_fft_abs)
        # x_spectrum = x_fft_angle
        self.clustered_idx, self.X, self.labels = fourier_intensity_extraction(x_spectrum, self.idxx, self.idxy, self.num_enc, x.shape[2])
        patterns = torch.empty((self.num_enc, x_fft.size(0), x_fft.size(1), x_fft.size(2))).to(self.device, dtype=torch.float32)
        for i, (idxx, idxy) in enumerate(self.clustered_idx):
            mask = torch.zeros(x_fft.size(1), x_fft.size(2)).to(self.device)
            mask[idxx, idxy] = 1
            temp = torch.empty_like(x_fft)
            for j in range(len(x_fft)):
                temp[j] = x_fft[j] * mask
            patterns[i] = temp
        patterns = torch.real(fft.ifft2(fft.ifftshift(patterns, dim=(-2, -1)), dim=(-2, -1)))

        return patterns