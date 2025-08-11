import math

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from modeling.DWT_Function.dwt_function import DWT_2D, IDWT_2D
from modeling.ISF import ISFLayer
from modeling.MFF import FrequencyFusinoMoudle
from modeling.mamba_layer_base import BasicLayer as MambaLayer


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class build_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        mamba_embed_dim = cfg.MODEL.MAMBA_CONFIG.EMBED_DIM
        num_feat = 64
        num_out_ch = 1
        self.conv1_before = nn.Sequential(
            nn.Conv2d(1, mamba_embed_dim, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(mamba_embed_dim, mamba_embed_dim, 3, 1, 1), nn.SiLU())
        self.conv2_before = nn.Sequential(
            nn.Conv2d(1, mamba_embed_dim, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(mamba_embed_dim, mamba_embed_dim, 3, 1, 1), nn.SiLU())

        self.num_layers = len(cfg.MODEL.MAMBA_CONFIG.DEPTHS)
        self.ISF_layers = nn.ModuleList([])
        self.MFF_layers = nn.ModuleList([])
        self.BMM_layers1 = nn.ModuleList([])
        self.BMM_layers2 = nn.ModuleList([])
        input_size = (cfg.MODEL.MAMBA_CONFIG.IMG_SIZE, cfg.MODEL.MAMBA_CONFIG.IMG_SIZE)
        self.input_resolution = input_size
        self.test_size = cfg.TEST.IMG_SIZE
        for i_layer in range(self.num_layers):
            s_layer = ISFGroup(
                dim=cfg.MODEL.MAMBA_CONFIG.EMBED_DIM,
                input_resolution=(input_size[0], input_size[1]),
                depth=cfg.MODEL.MAMBA_CONFIG.DEPTHS[i_layer],
                mlp_ratio=cfg.MODEL.MAMBA_CONFIG.MLP_RATIO,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            )
            f_layer = FrequencyFusinoMoudle(dim=cfg.MODEL.MAMBA_CONFIG.EMBED_DIM)
            self.ISF_layers.append(s_layer)
            self.MFF_layers.append(f_layer)
        for i in range(2):
            m_layer1 = BaseMambaGroup(
                dim=cfg.MODEL.MAMBA_CONFIG.EMBED_DIM,
                input_resolution=(input_size[0], input_size[1]),
                depth=cfg.MODEL.MAMBA_CONFIG.DEPTHS[0],
                mlp_ratio=cfg.MODEL.MAMBA_CONFIG.MLP_RATIO,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            )
            m_layer2 = BaseMambaGroup(
                dim=cfg.MODEL.MAMBA_CONFIG.EMBED_DIM,
                input_resolution=(input_size[0], input_size[1]),
                depth=cfg.MODEL.MAMBA_CONFIG.DEPTHS[0],
                mlp_ratio=cfg.MODEL.MAMBA_CONFIG.MLP_RATIO,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            )
            self.BMM_layers1.append(m_layer1)
            self.BMM_layers2.append(m_layer2)

        self.norm = nn.LayerNorm(mamba_embed_dim)
        in_dim = 2 * self.num_layers * mamba_embed_dim  # * self.num_layers
        self.conv_final_fuse = nn.Sequential(
            nn.Conv2d(in_dim, num_feat, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))

        self.apply(_init_weights)


    def forward(self, batched_inputs):
        bsz, C, H, W = batched_inputs["vi_y"].shape
        input_size = (H, W)
        vi_shallow_features = self.conv1_before(batched_inputs["vi_y"])
        ir_shallow_features = self.conv2_before(batched_inputs["ir"])

        #name = batched_inputs["jpg_names"]

        bsz, C, H, W = vi_shallow_features.shape
        vi_features = vi_shallow_features.permute(0, 2, 3, 1).view(bsz, int(H * W), -1)
        ir_features = ir_shallow_features.permute(0, 2, 3, 1).view(bsz, int(H * W), -1)
        spa_features = []
        fre_features = []

        for bmm_layer1, bmm_layer2 in zip(self.BMM_layers1, self.BMM_layers2):

            vi_features = bmm_layer1(vi_features, input_size)  # B L C
            ir_features = bmm_layer2(ir_features, input_size)  # B L C

        vi_input = vi_features
        ir_input = ir_features

        for spa_layer, fre_layer in zip(self.ISF_layers, self.MFF_layers):
            fre_fuse, LF_fuse, HF_fuse = fre_layer(vi_input, ir_input, vi_shallow_features, ir_shallow_features, input_size)
            spa_fuse = spa_layer(vi_input, ir_input, LF_fuse, HF_fuse, input_size)  # B L C
            spa_features.append(spa_fuse)
            fre_features.append(fre_fuse)

        features_spa = torch.cat(spa_features, dim=-1)
        features_fre = torch.cat(fre_features, dim=1)
        B, L, C = features_spa.shape
        features_spa = features_spa.permute(0, 2, 1).view(B, C, *input_size)
        bsz, C, H, W = features_fre.shape
        fused_features = torch.cat((features_spa, features_fre), dim=1)

        out = self.conv_final_fuse(fused_features)

        return out


class ISFGroup(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False
                 ):
        super(ISFGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution  # [64, 64]

        self.ISF = ISFLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

    def forward(self, x, input2, LF_fuse, HF_fuse, test_size=None):
        B, L, C = x.shape
        if test_size is not None:
            input_size = test_size
        else:
            H = W = math.sqrt(L)
            input_size = (int(H), int(W))
        x = self.ISF(x, input2, LF_fuse, HF_fuse, input_size)

        return x

    def flops(self):
        flops = 0
        flops += self.ISF_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class BaseMambaGroup(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False
                 ):
        super(BaseMambaGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.SPFusion_group = MambaLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

    def forward(self, x, test_size=None):
        B, L, C = x.shape
        if test_size is not None:
            input_size = test_size
        else:
            H = W = math.sqrt(L)
            input_size = (int(H), int(W))
        x = self.SPFusion_group(x, input_size)

        return x

