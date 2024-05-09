import numpy as np
import torch
from torch import nn


class convolution(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, dropout=True, init=False):
        super(convolution, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=1,
                              padding=1, bias=True)
        if init:
            nn.init.kaiming_normal(self.conv.weight)
        if dropout:
            self.dropout = nn.Dropout3d(p=0.1)
        if norm:
            self.norm = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        if self.dropout != None:
            x = self.dropout(x)
        if self.norm != None:
            x = self.norm(x)
        x = self.relu(x)

        return x

class downsample(nn.Module):
    def __init__(self, downsample_op='max', kernel=(2, 2, 2), stride=2):
        super(downsample, self).__init__()
        if downsample_op == 'max':
            self.downsample = nn.MaxPool3d(kernel_size=kernel, stride=stride)
        else:
            self.downsample = nn.MaxPool3d(kernel_size=kernel, stride=stride)

    def forward(self, x):
        x = self.downsample(x)

        return x


class convolutional_layer(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, dropout=True, init=False, num_layers=2):
        super(convolutional_layer, self).__init__()
        self.operations = nn.Sequential()
        for layer in num_layers:
            if layer == 0:
                self.operations.append(convolution(in_channels, out_channels, norm, dropout, init))
            else:
                self.operations.append(convolution(out_channels, out_channels, out_channels, norm, dropout, init))
        
        self.operations.append(downsample())

    def forward(self, x):
        x = self.operations(x)

        return x


        
class dev_unet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_conv_per_layer=2, downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d, dropout_op=nn.Dropout3d, base_num_features=32, activation_function=nn.ReLU, deep_supervision=False,
                 dropout=False, init_weights=False, pooling_size=None, conv_kernel_size=None, convolutional_pooling=False,
                 convolutional_upsampling=False):
        super(unet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.conv_op = conv_op
        self.activation_function = activation_function
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        self.num_features = base_num_features
        self.modules = []

        for layer in range(num_layers):
            if layer == 0:
                self.modules.append(convolutional_layer(in_channels, self.num_features))
            



    

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
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
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unet, self).__init__()
        filters = [32, 64, 128, 256, 512]

        self.conv1 = conv_block(in_channels, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.conv5 = conv_block(filters[3], filters[4])

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.up5 = up_conv(filters[4], filters[3])
        self.up_conv5 = conv_block(filters[4], filters[3])

        self.up4 = up_conv(filters[3], filters[2])
        self.up_conv4 = conv_block(filters[3], filters[2])

        self.up3 = up_conv(filters[2], filters[1])
        self.up_conv3 = conv_block(filters[2], filters[1])

        self.up2 = up_conv(filters[1], filters[0])
        self.up_conv2 = conv_block(filters[1], filters[0])

        self.out_conv = nn.Conv3d(filters[0], out_channels + 1, kernel_size=1, stride=1, padding=0)

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


class GenericUNet(nn.Module):
    DEFAULT_BATCH = 2
    DEFAULT_PATCH = (64, 192, 160)

    def __init__(self, input_channels, base_features, num_classes, num_layers, num_convolutions_per_layer=2,
                 feature_map_pooling=2,
                 dropout=0.0, deep_supervision=False, dropout_localization=False, weight_init=True,
                 convolutional_pooling=False,
                 convolutional_upsampling=False, activation_function=nn.ReLU, output_activation=nn.Softmax):
        super(GenericUNet, self).__init__()
        self.convolutional_pooling = convolutional_pooling
        self.convolutional_upsampling = convolutional_upsampling
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.base_features = base_features
        self.deep_supervision = deep_supervision


        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d
        transpose_conv = nn.ConvTranspose3d

        pooling_kernel = [(2, 2, 2)] * num_layers
        convolution_kernel = [(3, 3, 3)] * (num_layers + 1)

        self.input_shape_divisibility = np.prod(pooling_kernel, 0, dtype=np.int64)

        self.pooling_kernel = pooling_kernel
        self.convolutional_kernel = convolution_kernel
        self.convolutional_padding = []
        for kernel in self.convolutional_kernel:
            self.convolutional_padding.append([1 if kernel_dim == 3 else 0 for kernel_dim in kernel])

        self.convolutional_blocks_context = []
        self.convolutional_blocks_localization = []
        self.td = []
        self.tu = []
        self.segmented_outputs = []

    def init_model(self):

        for layer in range(self.num_layers):
            if layer != 0 and self.convolutional_pooling:
                first_stride = self.pooling_kernel[layer - 1]
            else:
                first_stride = None


if __name__ == '__main__':
    x = torch.randn(2, 1, 48, 48, 48)
    model = unet(1, 1)
    print(model(x).shape)
