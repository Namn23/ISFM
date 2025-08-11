import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torch.autograd import Variable
from math import exp

class make_loss(nn.Module):
    def __init__(self):
        super().__init__()
        #gradloss = Gradloss()

    def forward(self, src1, src2, fused_image, task_type, epoch=None):
        loss_vif, pixel_loss_vif, max_pixel_loss_vif, grad_loss_vif, ssim_loss_vif = (
            fusion_loss_VIF(src1, src2, fused_image, task_type, epoch))
        LOSS_VIF = {"loss": loss_vif, "pixel_loss": pixel_loss_vif, "grad_loss": grad_loss_vif,
                    "ssim_loss": ssim_loss_vif, "max_pixel_loss": max_pixel_loss_vif}

        return LOSS_VIF


def fusion_loss_VIF(src1, src2, fused_image, task_type, epoch):
    gradloss = MaxGradLoss()

    loss_s1 = F.l1_loss(fused_image, src1)
    loss_s2 = F.l1_loss(fused_image, src2)

    loss_max = F.l1_loss(fused_image, torch.max(src1, src2))
    ir_3 = torch.cat([src2] * 3, dim=1)
    vi_y3 = torch.cat([src1] * 3, dim=1)
    fused_y3 = torch.cat([fused_image] * 3, dim=1)
    gradinet_loss = gradloss(fused_y3, vi_y3, ir_3)

    ssim_loss = 0.5 * (1 - ssim(fused_image, src1, 12)) + 0.5 * (1 - ssim(fused_image, src2, 12))
    pixel_loss = loss_s1 + loss_s2
    loss = pixel_loss + loss_max * 5 + 5 * gradinet_loss + 0.5 * ssim_loss

    return loss, pixel_loss, loss_max, gradinet_loss, ssim_loss


class SobelxyRGB(nn.Module):
    def __init__(self, isSignGrad=True):
        super(SobelxyRGB, self).__init__()
        self.isSignGrad = isSignGrad
        kernelx = [[-0.2, 0, 0.2],
                   [-1, 0, 1],
                   [-0.2, 0, 0.2]]
        kernely = [[0.2, 1, 0.2],
                   [0, 0, 0],
                   [-0.2, -1, -0.2]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernelx = kernelx * 1
        kernely = kernely * 1
        kernelx = kernelx.repeat(1, 3, 1, 1)
        kernely = kernely.repeat(1, 3, 1, 1)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.relu = nn.ReLU()

    def forward(self, x):
        # R,G,B = x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        if self.isSignGrad:
            return sobelx + sobely
        else:
            return torch.abs(sobelx) + torch.abs(sobely)


class MaxGradLoss(nn.Module):
    """Loss function for the grad loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0, isSignGrad=True):
        super(MaxGradLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = SobelxyRGB(isSignGrad)
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None, *args, **kwargs):
        """Forward function.

        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): TIR image with shape (N, C, H, W).
        """
        if im_tir != None:
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            mask = torch.ge(torch.abs(rgb_grad), torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + rgb_grad.masked_fill_(~mask, 0)

            generate_img_grad = self.sobelconv(im_fusion)

            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
            loss_grad = self.loss_weight * sobel_loss
        else:
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad, rgb_grad)
            loss_grad = self.loss_weight * sobel_loss

        return loss_grad


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    img1 = img1.to(dtype=img2.dtype)
    mu1 = F.conv2d(img1, window, padding=int(window_size / 2), groups=channel)
    mu2 = F.conv2d(img2, window, padding=int(window_size / 2), groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=int(window_size / 2), groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=int(window_size / 2), groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=int(window_size / 2), groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).cuda()
    ssim_loss = _ssim(img1, img2, window, window_size, channel, size_average).cuda()
    return ssim_loss


class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, 'forward', func)

    def extra_repr(self):
        return getattr(self.forward, '__name__', type(self.forward).__name__) + '()'


class WeightedLoss(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights, verbose=False):
        super().__init__()
        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'({i}) {type(self[i]).__name__}: {loss.item()}')

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)

