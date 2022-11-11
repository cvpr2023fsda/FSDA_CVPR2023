import sys

import torch
import torch.nn as nn
import torch.fft as fft

class Emphasize(nn.Module):
    def __init__(self, device, args, extract):
        super(Emphasize, self).__init__()

        self.device = device
        self.extract = extract
        self.args = args

    def forward(self, x, pattern_importance):
        x_fft = fft.fftshift(fft.fft2(x, s=[self.args.image_size, self.args.image_size], dim=(-2, -1)), dim=(-2, -1))
        mask = torch.ones((x_fft.size(0), x_fft.size(1), x_fft.size(2))).to(self.device)
        for i, (idxx, idxy) in enumerate(self.extract.clustered_idx) :
            for j in range(len(pattern_importance)) :
                mask[j, idxx, idxy] = pattern_importance[j, i]
                # mask[j, idxx, idxy] = torch.exp(torch.randn(1)).to(self.device)

        x_reject = x_fft * mask
        # x_fft_abs, x_fft_angle = torch.abs(x_fft), torch.angle(x_fft)
        # x_reject_abs = x_fft_abs * mask
        # x_reject_angle = x_fft_angle * mask
        # x_reject = x_reject_abs * torch.exp((1j) * x_fft_angle)
        x_ifft = torch.real(fft.ifft2(fft.ifftshift(x_reject, dim=(-2, -1)), s=[x.shape[1], x.shape[2]], dim=(-2, -1)))
        return x_ifft