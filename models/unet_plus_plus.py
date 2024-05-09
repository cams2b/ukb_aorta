import torch.nn as nn
import torch


class conv_block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class unet_plus_plus(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False):
        super(unet_plus_plus, self).__init__()
        filters = [16, 32, 64, 128, 256]
        self.deep_supervision = deep_supervision

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ### encoding
        self.conv0_0 = conv_block(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block(filters[3], filters[4], filters[4])
        ### layer 1
        self.conv0_1 = conv_block(filters[0]+filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block(filters[1]+filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block(filters[2]+filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block(filters[3]+filters[4], filters[3],filters[3])
        ### layer 2
        self.conv0_2 = conv_block(filters[0]*2+filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block(filters[1]*2+filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block(filters[2]*2+filters[3], filters[2], filters[2])
        ### layer 3
        self.conv0_3 = conv_block(filters[0]*3+filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block(filters[1]*3+filters[2], filters[1], filters[1])
        ### layer 4
        self.conv0_4 = conv_block(filters[0]*4+filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes+1, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes+1, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes+1, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes+1, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes+1, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], 1))

        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], 1))

        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], 1))

        x4_0 = self.conv4_0(self.maxpool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

if __name__ == '__main__':
    x = torch.randn(2, 1, 48, 48)
    model = unet_plus_plus(in_channels=1, num_classes=1, deep_supervision=False)
    print(model(x).shape)