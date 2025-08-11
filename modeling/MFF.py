import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.DWT_Function.dwt_function import DWT_2D, IDWT_2D


class FrequencyFusinoMoudle(nn.Module):
    def __init__(self, dim):
        super(FrequencyFusinoMoudle, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), nn.SiLU())
        self.LFB = LowFrequencyFusionBlock(dim=dim)
        self.GFHB = GuidedFilterHFBlock(dim=dim)


    def Fuse_Freq_Fuc(self, Freq1, Freq2):
        """

        Args:
            Freq1: b 4*C H W, include LF: b C H W, HF: b C*3 H W
            Freq2: b 4*C H W, include LF: b C H W, HF: b C*3 H W

        Returns:
            LF_fuse: b C H W
            HF_fuse: b C H W
        """

        B, C1, H, W = Freq1.shape
        C = C1 // 4
        LF1 = Freq1[:, :C, :, :]  # b C H W
        HF_vi = Freq1[:, C:, :, :]  # b C*3 H W
        LF2 = Freq2[:, :C, :, :]  # b C H W
        HF_ir = Freq2[:, C:, :, :]  # b C*3 H W
        LF_fuse = (LF1 + LF2) / 2
        HF1 = []
        for i in range(3):  # 针对每个高频分量（3 * C）
            high_freq_part = HF_vi[:, i * C:(i + 1) * C, :, :]
            HF1.append(high_freq_part)
        HF2 = []
        for i in range(3):  # 针对每个高频分量（3 * C）
            high_freq_part = HF_ir[:, i * C:(i + 1) * C, :, :]
            HF2.append(high_freq_part)
        HF = []
        for i in range(3):  # 针对每个高频分量（3 * C）
            high_freq_part = (HF1[i] + HF2[i]) / 2
            HF.append(high_freq_part)
        HF_fuse = torch.cat(HF, dim=1)

        return LF_fuse, HF_fuse

    def forward(self, vi_features, ir_features, vi_shallow_features, ir_shallow_features, input_size):
        B, L, C = vi_features.shape
        vi_features = vi_features.permute(0, 2, 1).view(B, C, *input_size)
        ir_features = ir_features.permute(0, 2, 1).view(B, C, *input_size)
        dwt_vi = self.dwt(vi_features)
        dwt_ir = self.dwt(ir_features)
        LF_fuse_g, HF_fuse_g = self.Fuse_Freq_Fuc(dwt_vi, dwt_ir)
        dwt_vi = self.dwt(vi_shallow_features)
        dwt_ir = self.dwt(ir_shallow_features)
        LF_fuse_l, HF_fuse_l = self.Fuse_Freq_Fuc(dwt_vi, dwt_ir)
        LF_fuse = (LF_fuse_g + LF_fuse_l) / 2
        HF_fuse = (HF_fuse_g + HF_fuse_l) / 2
        H, W = input_size
        fused_LF = self.LFB(LF_fuse)
        fused_HF = self.GFHB(HF_fuse)
        fre_fused = torch.cat([fused_LF, fused_HF], dim=1)
        HF = []
        for i in range(3):  # 针对每个高频分量（3 * C）
            high_freq_part = fused_HF[:, i * C:(i + 1) * C, :, :]
            HF.append(high_freq_part)
        HF_fuse = (HF[0] + HF[1] + HF[2]) / 3
        fused_fre = self.idwt(fre_fused)  # B C H W
        _, _, h1, w1 = fused_fre.shape
        if h1 != H or w1 != W:
            fused_fre = F.interpolate(fused_fre, size=(H, W), mode='bilinear')
        fused_fre = self.conv2(fused_fre)

        return fused_fre, fused_LF, HF_fuse


class LowFrequencyFusionBlock(nn.Module):
    def __init__(self, dim):
        super(LowFrequencyFusionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 3, 1, 1), nn.SiLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0), nn.SiLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0), nn.SiLU())
        self.conv_before = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0), nn.SiLU())
        self.conv_after = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0), nn.SiLU())

        conv_bias = True
        d_conv = 3
        self.dw_conv_3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            groups=dim,
            kernel_size=d_conv,
            stride=1,
            padding=(d_conv - 1) // 2,
        )
        self.dw_conv_5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            groups=dim,
            kernel_size=5,
            stride=1,
            padding=(5 - 1) // 2,
        )
        self.act = nn.SiLU()

    def forward(self, LF_spa):
        B, C, H, W = LF_spa.shape
        LF_spa = self.conv_before(LF_spa)
        avg_p = self.avg_pool(LF_spa)
        max_p = self.max_pool(LF_spa)
        features_pooling = torch.cat((avg_p, max_p), dim=1)
        attn_w = self.conv1(features_pooling)
        attn_w = self.sigmoid(attn_w)

        LF_3x3 = self.act(self.dw_conv_3(LF_spa))
        LF_5x5 = self.act(self.dw_conv_5(LF_spa))
        LF_3x3 = self.conv2(LF_3x3)
        LF_5x5 = self.conv3(LF_5x5)
        LF_ms = LF_3x3 + LF_5x5

        LF_fused = LF_spa + LF_ms * attn_w
        LF_fused = self.conv_after(LF_fused)

        return LF_fused


class GuidedFilterHFBlock(nn.Module):
    def __init__(self, dim):
        super(GuidedFilterHFBlock, self).__init__()

        self.avg_pool_3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool_5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_before = nn.Sequential(
            nn.Conv2d(3 * dim, 3 * dim, 1, 1, 0), nn.SiLU())
        self.conv_after = nn.Sequential(
            nn.Conv2d(3 * dim, 3 * dim, 1, 1, 0), nn.SiLU())
        conv_bias = True
        self.conv2 = nn.Sequential(
            nn.Conv2d(3 * dim, 3 * dim, 1, 1, 0), nn.SiLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(3 * dim, 3 * dim, 1, 1, 0), nn.SiLU())
        self.act = nn.SiLU()

    def forward(self, HF):
        B, C, H, W = HF.shape
        HF = self.conv_before(HF)
        HF_3 = HF - self.avg_pool_3(HF)
        HF_5 = HF - self.avg_pool_5(HF)
        HF_3 = self.conv2(HF_3)
        HF_5 = self.conv3(HF_5)
        HF_fused = HF + HF_3 + HF_5
        HF_fused = self.conv_after(HF_fused)

        return HF_fused
