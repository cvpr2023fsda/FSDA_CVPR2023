import torch.nn as nn

class Encoder(nn.Module) :
    def __init__(self):
        super(Encoder, self).__init__()

        # input : (?, 1, 256, 256)
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

    def forward(self, x):
        out = self.inc(x)
        out = self.down1(out)
        out = self.down2(out)
        out = self.down3(out)

        return out

class Down(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)

        return out

class DoubleConv(nn.Module) :
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()

        if not mid_channels : mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.double_conv(x)

        return out

class channel_reduction(nn.Module) :
    def __init__(self, ratio=2):
        super(channel_reduction, self).__init__()
        self.channel_reduction = nn.Conv2d(512, int(512//ratio), kernel_size=(1, 1))

    def forward(self, x):
        x = self.channel_reduction(x)

        return x