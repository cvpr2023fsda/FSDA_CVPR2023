import torch
import torch.nn as nn

from .Extract.ExtractModel import ExtractModel
from .Emphasize import Emphasize

class FSDA(nn.Module) :
    def __init__(self, args, device):
        super(FSDA, self).__init__()

        self.args = args
        self.extract = ExtractModel(args, device)
        self.emphasize = Emphasize(device, args, self.extract).to(device)

    def forward(self, x):

        pattern_importance = self.extract(x)
        new = self.emphasize(x, pattern_importance)

        return new