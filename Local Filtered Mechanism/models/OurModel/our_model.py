import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel):
        super(ConvBlock, self).__init__()
        self.body = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, 80, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            nn.Conv2d(80, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        x1 = self.body(x)
        out = torch.cat([x, x1], dim=1)

        return out


class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), groups=group, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), groups=group, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), groups=group, bias=False),

        )
        self.relu4 = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)
        out2 = out + out1
        out3 = self.conv3(out2)
        out4 = out3 + out
        out5 = self.relu4(out4)
        return out5


class DenseBlock(torch.nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.conv_block1 = ConvBlock(16)
        self.conv_block2 = ConvBlock(32)
        self.conv_block3 = ConvBlock(48)
        self.conv_block4 = ConvBlock(64)
        self.trans = EResidualBlock(80, 16)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        transition = self.trans(conv4)
        return transition


class OurModel(torch.nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        self.conv = nn.Conv2d(1, 16, 7, 1, 3)
        self.Net_A1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=None),
            DenseBlock()
        )
        self.Net_A2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=None),
            DenseBlock()
        )
        self.Net_A3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=None),
            DenseBlock()
        )
        self.Net_A4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=None),
            DenseBlock()
        )

        self.Net_B1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )

        self.Net_B2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )
        self.Net_B3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )
        self.Net_B4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m,nn.ConvTranspose2d):
                nn.init.normal_(m.weight,mean=0,std=0.01)

    def forward(self, x):
        conv = self.conv(x)
        Net_A1 = self.Net_A1(conv)
        Net_A2 = self.Net_A2(Net_A1)
        Net_A3 = self.Net_A3(Net_A2)
        Net_A4 = self.Net_A4(Net_A3)

        pool_B1 = F.interpolate(Net_A4, scale_factor=2)
        concat_B1 = torch.cat([Net_A3, pool_B1], dim=1)
        Net_B1 = self.Net_B1(concat_B1)

        pool_B2 = F.interpolate(Net_B1, scale_factor=2)
        concat_B2 = torch.cat([Net_A2, pool_B2], dim=1)
        Net_B2 = self.Net_B2(concat_B2)

        pool_B3 = F.interpolate(Net_B2, scale_factor=2)
        concat_B3 = torch.cat([Net_A1, pool_B3], dim=1)
        Net_B3 = self.Net_B3(concat_B3)

        pool_B4 = F.interpolate(Net_B3, scale_factor=2)
        concat_B4 = torch.cat([conv, pool_B4], dim=1)
        Net_B4 = self.Net_B4(concat_B4)
        return Net_B4


if __name__ == '__main__':
    x = torch.rand([1, 1, 512, 512])
    model = OurModel()
    out = model(x)
    print("out {}".format(out.shape))
