# Copyright (c) OpenMMLab. All rights reserved.
# -----------------------------------------------------------------------------
# This file is adapted from the HaMeR / ViTPose-style Vision Transformer
# backbone and adds a random token-dropping path for random sparsification.
# Modifications for RANTS by Phuong Truong.
# Main references:
#   [1] G. Pavlakos et al., "Reconstructing Hands in 3D with Transformers,"
#       CVPR 2024.  (HaMeR)
#   [2] Y. Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human
#       Pose Estimation," NeurIPS 2022.
#   [3] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers
#       for Image Recognition at Scale," ICLR 2021.
#   [4] K. He et al., "Masked Autoencoders Are Scalable Vision Learners,"
#       CVPR 2022.
#   [5] G. Huang et al., "Deep Networks with Stochastic Depth," ECCV 2016.
#
# What is added in this file:
#   - A configurable TOKEN_DROP_RATIO read from the YAML config.
#   - A fixed random token subset generated once and reused during training.
#   - A sparse forward path that encodes only kept tokens and writes them back
#     into a full token grid before reshaping to a feature map.
# -----------------------------------------------------------------------------
import math

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

def random_masking(x, drop_ratio: float):
    """
    Random token dropping implemented in the style of MAE patch masking.

    References:
      - K. He et al., "Masked Autoencoders Are Scalable Vision Learners,"
        CVPR 2022.
      - A. Dosovitskiy et al., "An Image is Worth 16x16 Words:
        Transformers for Image Recognition at Scale," ICLR 2021.

    Args:
        x (Tensor): Token sequence of shape [B, L, D] after patch embedding
            (with or without positional embeddings).
        drop_ratio (float): Fraction of tokens to drop.

    Returns:
        x_kept (Tensor): Kept tokens with shape [B, L_keep, D].
        mask (Tensor): Binary mask of shape [B, L], where 0 means keep and
            1 means drop, in the original token order.
        ids_restore (Tensor): Indices that map the shuffled sequence back to
            the original order.
    """
    B, L, D = x.shape
    L_keep = int(L * (1.0 - drop_ratio))
   # print("Drop " + str(drop_ratio)) 
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)           # ascending order
    ids_restore = torch.argsort(ids_shuffle, dim=1)     # inverse permutation
    ids_keep = ids_shuffle[:, :L_keep]                  # kept subset

    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, L, device=x.device)
    mask[:, :L_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)           # restore the mask to the original token order

    return x_kept, mask, ids_restore


def vit(cfg):
    token_drop_ratio = getattr(cfg.MODEL.BACKBONE, 'TOKEN_DROP_RATIO', 0.0)
    return ViT(
                img_size=(256, 192),
                patch_size=16,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                ratio=1,
                use_checkpoint=False,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.55,
                drop_ratio=token_drop_ratio,
            )

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """
    Per-sample stochastic depth on the residual branch.

    Reference:
      - G. Huang et al., "Deep Networks with Stochastic Depth," ECCV 2016.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Image-to-patch embedding.

    This module converts an input image into a sequence of patch tokens, which
    is the standard first step in Vision Transformers.

    Reference:
      - A. Dosovitskiy et al., ICLR 2021.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """
    CNN feature-map embedding.

    If a convolutional backbone is provided, this module projects the CNN
    feature map into the transformer embedding space.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ViT(nn.Module):
    """
    Vision Transformer backbone with optional fixed random token dropping.

    Design summary:
      1. PatchEmbed converts the image into patch tokens.
      2. Absolute positional embeddings are added.
      3. During training, a fixed random subset of tokens can be kept according
         to `drop_ratio`.
      4. Only the kept tokens are processed by the transformer blocks.
      5. The encoded kept tokens are written back into a full token grid so the
         downstream decoder still receives a dense spatial feature map.

    This keeps the external backbone interface compatible with HaMeR while
    enabling a simple random sparsification baseline inspired by MAE-style
    token selection.
    """

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 drop_ratio=0.0,
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        # Fraction of patch tokens dropped during training.
        # This value is read from cfg.MODEL.BACKBONE.TOKEN_DROP_RATIO.
        self.drop_ratio = float(drop_ratio)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches
        
        # Initialize one fixed random keep/drop partition shared by all training images.
        if self.drop_ratio > 0.0:
            L = num_patches
            L_keep = int(L * (1.0 - self.drop_ratio))

            # Randomly permute patch indices once at model initialization.
            perm = torch.randperm(L)
            ids_keep = perm[:L_keep]     # (L_keep,)
            ids_drop = perm[L_keep:]     # (L_drop,)

            # Save as non-persistent buffers so they follow device placement.
            self.register_buffer("fixed_ids_keep", ids_keep, persistent=False)
            self.register_buffer("fixed_ids_drop", ids_drop, persistent=False)

        else:
            self.fixed_ids_keep = None
            self.fixed_ids_drop = None

        # Positional embedding includes one extra slot for a class token in
        # the pretrained checkpoint, although this backbone uses only patch
        # positions in forward_features().
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Linearly increase stochastic depth across layers, as is common in
        # modern ViT implementations.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C_img, H, W = x.shape                      # C_img = 3
        x, (Hp, Wp) = self.patch_embed(x)             # x has shape [B, L, Cemb], where L = Hp * Wp.
        Cemb = x.shape[-1]                            # lưu embed dim

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:]             # bỏ cls_token, pos cho patch thôi

        # ================= FIX MASK POSITIONS =================
        if self.drop_ratio > 0.0  and self.training and self.fixed_ids_keep is not None:
            # x has shape [B, L, Cemb], where L = Hp * Wp.
            B_, L, C_ = x.shape
            assert C_ == Cemb
            
            ids_keep = self.fixed_ids_keep.to(x.device)              # [L_keep]
            ids_keep_batch = ids_keep.unsqueeze(0).expand(B_, -1)     # [B, L_keep]
            
            # Gather only the kept tokens before entering the transformer.
            x_kept = torch.gather(x, 1, ids_keep_batch.unsqueeze(-1).expand(-1, -1, Cemb))
            
            # Encode only the reduced token set.
            for blk in self.blocks:
                x_kept = checkpoint.checkpoint(blk, x_kept) if self.use_checkpoint else blk(x_kept)
            x_kept = self.last_norm(x_kept)
            
            # Restore the kept features into the full token grid.
            # Only kept positions are updated; dropped positions stay unchanged.
            x_full = x.clone()
            x_full[:, ids_keep, :] = x_kept
            x = x_full
            del x_full, x_kept

        else:
            # Standard dense path used at evaluation time or when dropping is disabled.
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
            x = self.last_norm(x)

        # Convert tokens back into a dense feature map [B, Cemb, Hp, Wp].
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp


    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
