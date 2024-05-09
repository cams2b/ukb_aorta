import torch.nn as nn
import torch
import torch.nn.functional as F
#from torchsummary import summary
import os
from abc import ABC, abstractmethod


"""
Implementations based on the HyperDenseNet paper: https://arxiv.org/pdf/1804.02967.pdf
"""


class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()
        self.best_loss = 1000000

    @abstractmethod
    def forward(self, x):
        pass



    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        r"""
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file to be restored.")

        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        # Restore model weights
        self.load_state_dict(ckpt_dict['model_state_dict'])

        # Restore optimizer status if existing. Evaluation doesn't need this
        # TODO return optimizer?????
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # Return global step
        return ckpt_dict['epoch']

    def save_checkpoint(self,
                        directory,
                        epoch, loss,
                        optimizer=None,
                        name=None):
        r"""
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            epoch (int): The training. epoch
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':
                self.state_dict(),
            'optimizer_state_dict':
                optimizer.state_dict() if optimizer is not None else None,
            'epoch':
                epoch
        }

        # Save the file with specific name
        if name is None:
            name = "{}_{}_epoch.pth".format(
                os.path.basename(directory),  # netD or netG
                'last')

        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST.pth".format(
                os.path.basename(directory))
            torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()
        
class _HyperDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_channels, drop_rate):
        super(_HyperDenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features,
                                           num_output_channels, kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_HyperDenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return torch.cat([x, new_features], 1)


class _HyperDenseBlock(nn.Sequential):
    """
    Constructs a series of dense-layers based on in and out kernels list
    """

    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlock, self).__init__()
        out_kernels = [1, 25, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 9

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        print("out:", out_kernels)
        print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _HyperDenseBlockEarlyFusion(nn.Sequential):
    def __init__(self, num_input_features, drop_rate):
        super(_HyperDenseBlockEarlyFusion, self).__init__()
        out_kernels = [1, 25, 25, 50, 50, 50, 75, 75, 75]
        self.number_of_conv_layers = 8

        in_kernels = [num_input_features]
        for j in range(1, len(out_kernels)):
            temp = in_kernels[-1]
            in_kernels.append(temp + out_kernels[j])

        print("out:", out_kernels)
        print("in:", in_kernels)

        for i in range(self.number_of_conv_layers):
            layer = _HyperDenseLayer(in_kernels[i], out_kernels[i + 1], drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class SinglePathDenseNet(BaseModel):
    def __init__(self, in_channels, classes=2, drop_rate=0.1, return_logits=True, early_fusion=False):
        super(SinglePathDenseNet, self).__init__()
        self.return_logits = return_logits
        self.features = nn.Sequential()
        self.num_classes = classes
        self.input_channels = in_channels

        if early_fusion:
            block = _HyperDenseBlockEarlyFusion(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 52:
                total_conv_channels = 477
            else:
                if in_channels == 3:
                    total_conv_channels = 426
                else:
                    total_conv_channels = 503

        else:
            block = _HyperDenseBlock(num_input_features=in_channels, drop_rate=drop_rate)
            if in_channels == 2:
                total_conv_channels = 452
            else:
                total_conv_channels = 451

        self.features.add_module('denseblock1', block)

        self.features.add_module('conv1x1_1', nn.Conv3d(total_conv_channels,
                                                        400, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_1', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_2', nn.Conv3d(400,
                                                        200, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_2', nn.Dropout(p=0.5))

        self.features.add_module('conv1x1_3', nn.Conv3d(200,
                                                        150, kernel_size=1, stride=1, padding=0,
                                                        bias=False))

        self.features.add_module('drop_3', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', nn.Conv3d(150,
                                                           self.num_classes, kernel_size=1, stride=1, padding=0,
                                                           bias=False))

    def forward(self, x):
        features = self.features(x)
        if self.return_logits:
            out = self.classifier(features)
            return out

        else:
            return features


   

if __name__ == '__main__':
    x = torch.randn(2, 1, 48, 48, 48)
    model = SinglePathDenseNet(in_channels=1)
    print(model(x).shape)
