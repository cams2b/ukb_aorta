import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import einsum
import numpy as np


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        input = flatten(net_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return  - 2. * intersect / denominator.clamp(min=self.smooth)



def multiclass_dice_coef(y_pred, y_true):
    smooth = 1.

    num_classes = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred, num_classes)
    y_pred = torch.swapaxes(y_pred, 1, -1)
    y_pred = torch.swapaxes(y_pred,2, -1)

    total_dice = []
    for c in range(1, num_classes):
        intersection = torch.sum(y_true[:, c, :, :, :] * y_pred[:, c, :, :, :])
        nom = 2 * intersection
        denom = torch.sum(y_true[:, c, :, :, :]) + torch.sum(y_pred[:, c, :, :, :])
        dice = (nom + smooth) / (denom + smooth)
        total_dice.append(dice)

    dice_sum = total_dice[0]
    for d in total_dice[1:]:
        dice_sum += d

    mean_dice = dice_sum / (num_classes - 1)
    return mean_dice