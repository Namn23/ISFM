"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

import logging

import numpy as np
import torch

from einops import rearrange
from PIL import Image

from torchvision.utils import make_grid


logger = logging.getLogger()


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = torch.clamp(Y, 0, 1)
    Cr = torch.clamp(Cr, 0, 1)
    Cb = torch.clamp(Cb, 0, 1)
    return Y, Cr, Cb


def YCrCb2RGB(Y, Cr, Cb):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    #out = clamp(out)
    out = out.clamp(0, 1.0)
    return out

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_image(tensor, nrow):
    tensor = tensor * 255.
    tensor = torch.clamp(tensor, min=0., max=255.)
    tensor = rearrange(tensor, 'n b c h w -> b n c h w')
    tensor = rearrange(tensor, 'b n c h w -> (b n) c h w')
    tensor = make_grid(tensor, nrow=nrow)
    img = tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(img)

