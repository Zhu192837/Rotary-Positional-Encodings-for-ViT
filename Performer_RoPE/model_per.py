'''
This code is modified from https://github.com/lucidrains/performer-pytorch/ and https://github.com/naver-ai/rope-vit/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = t_x.shape[0] 
    depth = freqs.shape[1]  
    dim_per_head = freqs.shape[-1]  
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2) 
        freqs_y = t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)  
        freqs_x = freqs_x.permute(1, 0, 2).contiguous() 
        freqs_y = freqs_y.permute(1, 0, 2).contiguous()  
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis




def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Adjust freqs_cis to ensure compatibility with x for broadcasting.
    x: [batch_size, num_heads, seq_len, dim_per_head]
    freqs_cis: [seq_len, num_heads, dim_per_head] or [seq_len, dim_per_head]
    """
    if freqs_cis.ndim == 3:  # [seq_len, num_heads, dim_per_head]
        freqs_cis = freqs_cis.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, seq_len, dim_per_head]
    elif freqs_cis.ndim == 2:  # [seq_len, dim_per_head]
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim_per_head]
    else:
        raise ValueError(f"Unexpected freqs_cis shape: {freqs_cis.shape}")

    # Expand batch_size if necessary
    if freqs_cis.size(0) != x.size(0):
        freqs_cis = freqs_cis.expand(x.size(0), -1, -1, -1)

    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
        
class PerformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, kernel='relu', attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if freqs_cis is not None:
            q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class RoPE_Performer_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=1e-4, kernel='relu'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PerformerAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, kernel=kernel,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, freqs_cis=None):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RoPEAttention(Attention):
    """Multi-head Attention block with relative position embeddings."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x

class rope_performer_models(nn.Module):
    def __init__(self, img_size=32, patch_size=16, num_heads=8, embed_dim=768, depth=12,
                 mlp_ratio=4., drop_path_rate=0., drop_rate=0., rope_theta=100.0, rope_mixed=False,
                 use_ape=False, num_classes=1000, qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.use_ape = use_ape
        self.rope_mixed = rope_mixed

        if self.use_ape:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size * img_size // patch_size + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=num_heads)
            freqs = init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
            self.freqs = nn.Parameter(freqs, requires_grad=True)
            t_x, t_y = init_t_xy(end_x=img_size // patch_size, end_y=img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim // num_heads, theta=rope_theta)
            self.freqs_cis = self.compute_cis(end_x=img_size // patch_size, end_y=img_size // patch_size)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.ModuleList([
            RoPE_Performer_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i],
                                 norm_layer=norm_layer, kernel='relu', qkv_bias=qkv_bias)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x)  
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size // patch_size) ** 2 + 1, self.pos_embed.size(-1)
                )
            x = x + pos_embed
        if self.rope_mixed:
            freqs_cis = self.compute_cis(self.freqs, self.freqs_t_x, self.freqs_t_y)
        else:
            freqs_cis = self.freqs_cis.to(x.device)

        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.head(x)
        return x

@register_model
def rope_axial_ape_performer_small(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs
    )
    model.default_cfg = _cfg()
    return model



@register_model
def rope_axial_performer_small(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=100.0, rope_mixed=False, use_ape=False, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_mixed_performer_small(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=10.0, rope_mixed=True, use_ape=False, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_mixed_ape_performer_small(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_axial_ape_performer_base(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=512, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_axial_performer_base(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=512, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=100.0, rope_mixed=False, use_ape=False, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_mixed_performer_base(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=10.0, rope_mixed=True, use_ape=False, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_mixed_ape_performer_base(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs
    )
    model.default_cfg = _cfg()
    return model



@register_model
def rope_axial_ape_performer_large(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs
    )
    model.default_cfg = _cfg()
    return model



@register_model
def rope_axial_performer_large(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=100.0, rope_mixed=False, use_ape=False, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_mixed_performer_large(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=10.0, rope_mixed=True, use_ape=False, **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def rope_mixed_ape_performer_large(pretrained=False, img_size=32, pretrained_21k=False, **kwargs):
    model = rope_performer_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs
    )
    model.default_cfg = _cfg()
    return model