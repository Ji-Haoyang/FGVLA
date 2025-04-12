import os
import re
import json
import math
import torch
import logging
import warnings
import numpy as np
from torch import nn
from pathlib import Path
from copy import deepcopy
import torch.nn.functional as F
from dataclasses import dataclass
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union, Callable
from utils import to_2tuple
from open_clip.tokenizer import tokenize
from open_clip.pretrained import get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model, get_pretrained_url, download_pretrained_from_url


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out

    pass


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

    pass


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

    pass


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

    pass


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

    pass


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    pass


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

    pass


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, ls_init_value: float = None,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
                 layer_id: int = 0, adapter: Optional[nn.ModuleList] = None):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        self.adapter = adapter[layer_id] if adapter is not None else None
        pass

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        xls1 = self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + xls1
        x_mlp = self.mlp(self.ln_2(x))

        if self.adapter is not None:
            x_mlp = x_mlp + self.adapter(x_mlp)
            pass

        x = x + self.ls_2(x_mlp)
        return x

    pass


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

    pass


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0, ls_init_value: float = None,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm, adapter: nn.Module = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([ResidualAttentionBlock(
            width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer,
            norm_layer=norm_layer, layer_id=layer_id, adapter=adapter) for layer_id in range(layers)])
        pass

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
                pass
            pass
        return x

    pass


class VisionTransformer(nn.Module):

    def __init__(self, image_size: int, patch_size: int, width: int, layers: int,
                 heads: int, mlp_ratio: float, ls_init_value: float = None, output_dim: int = 512,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm, adapter: nn.ModuleList = None):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)

        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
                                       act_layer=act_layer, norm_layer=norm_layer, adapter=adapter)

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.init_parameters()
        pass

    def init_parameters(self):
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Add class token
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [B, N+1, D]
        
        # Add positional encoding
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Separate class token and patch tokens
        cls_token_out = self.ln_post(x[:, 0])
        patch_tokens = self.ln_post(x[:, 1:])  # [B, N, D]

        if self.proj is not None:
            cls_token_out = cls_token_out @ self.proj
            patch_tokens = patch_tokens @ self.proj

        return patch_tokens, cls_token_out

    pass


class TextTransformer(nn.Module):

    def __init__(self, context_length: int = 77, vocab_size: int = 49408, width: int = 512,
                 heads: int = 8, layers: int = 12, ls_init_value: float = None, output_dim: int = 512,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm, adapter: nn.ModuleList = None):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(width=width, layers=layers, heads=heads, ls_init_value=ls_init_value,
                                       act_layer=act_layer, norm_layer=norm_layer, adapter=adapter)
        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)
        self.init_parameters()
        pass

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            pass
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, return_word_feats=False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [B, n_ctx, D]
        x = x + self.positional_embedding.to(cast_dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [B, n_ctx, D]

        # Get word-level features
        word_feats = x @ self.text_projection  # [B, n_ctx, output_dim]
        
        # Get sentence-level features
        sent_feat = word_feats[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # [B, output_dim]

        if return_word_feats:
            return word_feats, sent_feat
        return sent_feat

    pass


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    pass


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    pass


class Adapter(nn.Module):

    def __init__(self, hidden_dim, num_heads, down_dim=128):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, down_dim)
        self.l2 = nn.Linear(down_dim, hidden_dim)
        self.msa = nn.MultiheadAttention(down_dim, num_heads)
        self.init_weights()
        pass

    def init_weights(self):
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()
        pass

    def forward(self, x):
        x = self.l1(x)
        attn, _ = self.msa(x, x, x)
        attn = attn + x
        x = self.l2(attn)
        return x

    pass

class CLIP(nn.Module):

    def __init__(self, embed_dim: int, vision_cfg: CLIPVisionCfg, text_cfg: CLIPTextCfg,
                 quick_gelu: bool = False, cast_dtype: Optional[torch.dtype] = None):
        super().__init__()

        self.adapter_img  = nn.ModuleList([Adapter(768, 8) for _ in range(12)])
        self.adapter_text = nn.ModuleList([Adapter(512, 8) for _ in range(12)])

        self.visual = self._build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype, adapter_img=self.adapter_img)
        text = self._build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype, adapter_text=self.adapter_text)

        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        pass

    @staticmethod
    def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg, quick_gelu: bool = False,
                            cast_dtype: Optional[torch.dtype] = None, adapter_img: Optional[nn.ModuleList] = None):
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        act_layer = QuickGELU if quick_gelu else nn.GELU

        if isinstance(vision_cfg.layers, (tuple, list)):
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width
            )
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else nn.LayerNorm
            visual = VisionTransformer(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                ls_init_value=vision_cfg.ls_init_value,
                output_dim=embed_dim,
                act_layer=act_layer,
                norm_layer=norm_layer,
                adapter=adapter_img
            )
        return visual

    @staticmethod
    def _build_text_tower(embed_dim: int, text_cfg: CLIPTextCfg, quick_gelu: bool = False,
                          cast_dtype: Optional[torch.dtype] = None, adapter_text: Optional[nn.ModuleList] = None):
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else nn.LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            adapter=adapter_text)
        return text

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable


    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        if isinstance(features, tuple):

            patch_feats, global_feat = features
        else:

            global_feat = features
            patch_feats = None
            
        if normalize:
            if patch_feats is not None:
                patch_feats = F.normalize(patch_feats, dim=-1)
            global_feat = F.normalize(global_feat, dim=-1)
            
        return global_feat if patch_feats is None else (patch_feats, global_feat)
    
    def encode_text(self, text, normalize: bool = False, return_word_feats: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [B, n_ctx, D]

        # Get word-level features
        word_feats = x @ self.text_projection  # [B, n_ctx, output_dim]
        
        # Get sentence-level features
        sent_feat = word_feats[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # [B, output_dim]

        if normalize:
            word_feats = F.normalize(word_feats, dim=-1)
            sent_feat = F.normalize(sent_feat, dim=-1)
            
        if return_word_feats:
            return word_feats, sent_feat
        return sent_feat

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()

    pass


class CreateModel(object):

    def __init__(self):
        self._MODEL_CONFIG_PATHS = [Path(__file__).parent / f"open_clip/model_configs/"]
        self._MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

        self._rescan_model_configs()  # initial populate of model config registry

        self.OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

        self.tokenize = tokenize
        pass

    def create_model_and_transforms(self, model_name: str, pretrained: Optional[str] = 'laion2b_s34b_b79k'):
        model = self.create_model(model_name, pretrained, precision='fp32', device='cpu')
        return model

    def create_model(self, model_name: str, pretrained: Optional[str],
                     precision: str = 'fp32', device: Union[str, torch.device] = 'cpu'):
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        if isinstance(device, str):
            device = torch.device(device)

        if pretrained and pretrained.lower() == 'openai':
            logging.info(f'Loading pretrained {model_name} from OpenAI.')
            model = self.load_openai_model(model_name, precision=precision, device=device, jit=False, cache_dir=None)
        else:
            model_cfg = self.get_model_config(model_name)
            if model_cfg is not None:
                logging.info(f'Loaded {model_name} model config.')
            else:
                logging.error(
                    f'Model config for {model_name} not found; available models {list(self._MODEL_CONFIGS.keys())}.')
                raise RuntimeError(f'Model config for {model_name} not found.')

            model = CLIP(**model_cfg, cast_dtype=self.get_cast_dtype(precision))

            pretrained_cfg = {}
            if pretrained:
                checkpoint_path = ''
                pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
                if pretrained_cfg:
                    checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=None)
                elif os.path.exists(pretrained):
                    checkpoint_path = pretrained

                if checkpoint_path:
                    logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                    self.load_checkpoint(model, checkpoint_path)
                else:
                    error_str = (f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                                 f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                    logging.warning(error_str)
                    raise RuntimeError(error_str)
                pass

            model.to(device=device)
            if precision in ("fp16", "bf16"):
                self.convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)
                pass

            # set image / mean metadata from pretrained_cfg if available, or use default
            model.visual.image_mean = pretrained_cfg.get('mean', None) or self.OPENAI_DATASET_MEAN
            model.visual.image_std = pretrained_cfg.get('std', None) or self.OPENAI_DATASET_STD
            pass
        return model

    def load_openai_model(self, name: str, precision: Optional[str] = None, device: Optional[Union[str, torch.device]] = None,
                          jit: bool = True, cache_dir: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision is None:
            precision = 'fp32' if device == 'cpu' else 'fp16'

        if get_pretrained_url(name, 'openai'):
            model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found;")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(model_path, map_location="cpu")

        if not jit:
            # Build a non-jit model from the OpenAI jitted model state dict
            cast_dtype = self.get_cast_dtype(precision)
            try:
                model = self.build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
            except KeyError:
                sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
                model = self.build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

            # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
            model = model.to(device)
            if precision.startswith('amp') or precision == 'fp32':
                model.float()
            elif precision == 'bf16':
                self.convert_weights_to_lp(model, dtype=torch.bfloat16)

            return model

        # patch the device names
        device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
        device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

        def patch_device(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                        node.copyAttributes(device_node)

        model.apply(patch_device)
        patch_device(model.encode_image)
        patch_device(model.encode_text)

        # patch dtype to float32 (typically for CPU)
        if precision == 'fp32':
            float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
            float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
            float_node = float_input.node()

            def patch_float(module):
                try:
                    graphs = [module.graph] if hasattr(module, "graph") else []
                except RuntimeError:
                    graphs = []

                if hasattr(module, "forward1"):
                    graphs.append(module.forward1.graph)

                for graph in graphs:
                    for node in graph.findAllNodes("aten::to"):
                        inputs = list(node.inputs())
                        for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                            if inputs[i].node()["value"] == 5:
                                inputs[i].node().copyAttributes(float_node)

            model.apply(patch_float)
            patch_float(model.encode_image)
            patch_float(model.encode_text)
            model.float()

        # ensure image_size attr available at consistent location for both jit and non-jit
        model.visual.image_size = model.input_resolution.item()
        return model

    def build_model_from_openai_state_dict(self, state_dict: dict, quick_gelu=True, cast_dtype=torch.float16):
        vit = "visual.proj" in state_dict

        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_size = vision_patch_size * grid_size
        else:
            counts: list = [
                len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_size = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        vision_cfg = CLIPVisionCfg(
            layers=vision_layers,
            width=vision_width,
            patch_size=vision_patch_size,
            image_size=image_size,
        )
        text_cfg = CLIPTextCfg(
            context_length=context_length,
            vocab_size=vocab_size,
            width=transformer_width,
            heads=transformer_heads,
            layers=transformer_layers
        )
        model = CLIP(
            embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
            cast_dtype=cast_dtype,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)

        self.convert_weights_to_lp(model)  # OpenAI state dicts are partially converted to float16
        model.load_state_dict(state_dict, strict=False)
        return model.eval()

    @staticmethod
    def _natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

    @staticmethod
    def get_cast_dtype(precision: str):
        cast_dtype = None
        if precision == 'bf16':
            cast_dtype = torch.bfloat16
        elif precision == 'fp16':
            cast_dtype = torch.float16
        return cast_dtype

    def _rescan_model_configs(self):
        config_ext = ('.json',)
        config_files = []
        for config_path in self._MODEL_CONFIG_PATHS:
            if config_path.is_file() and config_path.suffix in config_ext:
                config_files.append(config_path)
            elif config_path.is_dir():
                for ext in config_ext:
                    config_files.extend(config_path.glob(f'*{ext}'))

        for cf in config_files:
            with open(cf, 'r') as f:
                model_cfg = json.load(f)
                if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                    self._MODEL_CONFIGS[cf.stem] = model_cfg

        self._MODEL_CONFIGS = {k: v for k, v in sorted(self._MODEL_CONFIGS.items(), key=lambda x: self._natural_key(x[0]))}
        pass

    def get_model_config(self, model_name):
        if model_name in self._MODEL_CONFIGS:
            return deepcopy(self._MODEL_CONFIGS[model_name])
        else:
            return None

    @staticmethod
    def load_state_dict(checkpoint_path: str, map_location='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        return state_dict

    def load_checkpoint(self, model, checkpoint_path, strict=True):
        state_dict = self.load_state_dict(checkpoint_path)
        # detect old format and make compatible with new format
        if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
            state_dict = self.convert_to_custom_text_state_dict(state_dict)
        self.resize_pos_embed(state_dict, model)
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        return incompatible_keys

    @staticmethod
    def convert_to_custom_text_state_dict(state_dict: dict):
        if 'text_projection' in state_dict:
            # old format state_dict, move text tower -> .text
            new_state_dict = {}
            for k, v in state_dict.items():
                if any(k.startswith(p) for p in (
                        'text_projection',
                        'positional_embedding',
                        'token_embedding',
                        'transformer',
                        'ln_final',
                )):
                    k = 'text.' + k
                new_state_dict[k] = v
            return new_state_dict
        return state_dict

    @staticmethod
    def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
        """Convert applicable model parameters to low-precision (bf16 or fp16)"""

        def _convert_weights(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.to(dtype)
                if l.bias is not None:
                    l.bias.data = l.bias.data.to(dtype)

            if isinstance(l, (nn.MultiheadAttention, Attention)):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.to(dtype)

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.to(dtype)

        model.apply(_convert_weights)
        pass

    @staticmethod
    def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
        # Rescale the grid of position embeddings when loading from state_dict
        old_pos_embed = state_dict.get('visual.positional_embedding', None)
        if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
            return
        grid_size = to_2tuple(model.visual.grid_size)
        extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
        new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
        if new_seq_len == old_pos_embed.shape[0]:
            return

        if extra_tokens:
            pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
        else:
            pos_emb_tok, pos_emb_img = None, old_pos_embed
        old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

        logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
        pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
        pos_emb_img = F.interpolate(
            pos_emb_img,
            size=grid_size,
            mode=interpolation,
            align_corners=True,
        )
        pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
        if pos_emb_tok is not None:
            new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
        else:
            new_pos_embed = pos_emb_img
        state_dict['visual.positional_embedding'] = new_pos_embed
        pass

    pass

