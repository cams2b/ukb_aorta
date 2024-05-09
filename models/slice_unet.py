import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class slice_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(slice_unet, self).__init__()
        filters = [16, 32, 64, 128, 256]

        self.conv1 = conv_block(in_channels, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.conv5 = conv_block(filters[3], filters[4])

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up5 = up_conv(filters[4], filters[3])
        self.up_conv5 = conv_block(filters[4], filters[3])

        self.up4 = up_conv(filters[3], filters[2])
        self.up_conv4 = conv_block(filters[3], filters[2])

        self.up3 = up_conv(filters[2], filters[1])
        self.up_conv3 = conv_block(filters[2], filters[1])

        self.up2 = up_conv(filters[1], filters[0])
        self.up_conv2 = conv_block(filters[1], filters[0])

        self.out_conv = nn.Conv2d(filters[0], out_channels+1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        ### encoder
        e1 = self.conv1(x)

        e2 = self.maxpool1(e1)
        e2 = self.conv2(e2)

        e3 = self.maxpool2(e2)
        e3 = self.conv3(e3)

        e4 = self.maxpool3(e3)
        e4 = self.conv4(e4)

        e5 = self.maxpool4(e4)
        e5 = self.conv5(e5)

        ### decoder
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up_conv2(d2)

        out = self.out_conv(d2)

        return out

if __name__ == '__main__':
    x = torch.randn(2, 1, 48, 48)
    model = slice_unet()
    print(model(x).shape)