# DiT for Non-Square Input
# Based on: https://github.com/facebookresearch/DiT/blob/main/models.py
# Modified to handle non-square spatial dimensions (e.g., 64x256)
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

import torch.utils.checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#               Patch Embedding for Non-Square Input                            #
#################################################################################

class PatchEmbedNonSquare(nn.Module):
    """
    Patch embedding for non-square images.
    Unlike timm's PatchEmbed which assumes square input, this handles H != W.
    """
    def __init__(self, img_height, img_width, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        return torch.utils.checkpoint.checkpoint(self._forward, x, c, use_reentrant=False)

    def _forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTNonSquare(nn.Module):
    """
    Diffusion Transformer for non-square spatial inputs.

    Unlike the original DiT which assumes square input (H=W), this version
    handles rectangular input like (C=64, H=64, W=256) for image features.

    Key modifications:
    - PatchEmbedNonSquare: handles H != W
    - Positional embeddings: 2D sincos for rectangular grid
    - Unpatchify: reconstructs non-square output
    """
    def __init__(
        self,
        config,  # Must have: height, width, channels, learn_sigma
        patch_size=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_height = config.height
        self.img_width = config.width
        self.in_channels = config.channels
        self.learn_sigma = config.learn_sigma
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.grid_h = self.img_height // patch_size
        self.grid_w = self.img_width // patch_size

        self.x_embedder = PatchEmbedNonSquare(
            self.img_height, self.img_width, patch_size,
            self.in_channels, hidden_size
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.grid_h * self.grid_w
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding for non-square grid:
        pos_embed = get_2d_sincos_pos_embed_nonsquare(
            self.pos_embed.shape[-1], self.grid_h, self.grid_w
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Reconstruct non-square image from patches.

        x: (B, num_patches, patch_size² * out_channels)
        return: (B, out_channels, height, width)
        """
        c = self.out_channels
        p = self.patch_size
        h, w = self.grid_h, self.grid_w

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def _forward(self, x, t, null_indicator):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        c = t
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, T, patch_size² * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return [x]

    def forward(self, x, t=None, log_snr=None, shape=None, mask=None, null_indicator=None):
        return self._forward(x=x, t=t, null_indicator=null_indicator)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed_nonsquare(embed_dim, grid_h, grid_w, cls_token=False, extra_tokens=0):
    """
    2D sincos positional embedding for non-square grids.

    grid_h: int of the grid height (number of patches in height)
    grid_w: int of the grid width (number of patches in width)
    return:
    pos_embed: [grid_h*grid_w, embed_dim]
    """
    grid_h_vals = np.arange(grid_h, dtype=np.float32)
    grid_w_vals = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_vals, grid_h_vals)  # w first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                           DiT Non-Square Configs                              #
#################################################################################

def DiTNonSquare_XL_4(config, **kwargs):
    return DiTNonSquare(config=config, depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiTNonSquare_XL_8(config, **kwargs):
    return DiTNonSquare(config=config, depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiTNonSquare_L_4(config, **kwargs):
    return DiTNonSquare(config=config, depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiTNonSquare_L_8(config, **kwargs):
    return DiTNonSquare(config=config, depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiTNonSquare_B_4(config, **kwargs):
    return DiTNonSquare(config=config, depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiTNonSquare_B_8(config, **kwargs):
    return DiTNonSquare(config=config, depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiTNonSquare_S_4(config, **kwargs):
    return DiTNonSquare(config=config, depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiTNonSquare_S_8(config, **kwargs):
    return DiTNonSquare(config=config, depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiTNonSquare_models = {
    'DiT-XL/4': DiTNonSquare_XL_4,  'DiT-XL/8': DiTNonSquare_XL_8,
    'DiT-L/4':  DiTNonSquare_L_4,   'DiT-L/8':  DiTNonSquare_L_8,
    'DiT-B/4':  DiTNonSquare_B_4,   'DiT-B/8':  DiTNonSquare_B_8,
    'DiT-S/4':  DiTNonSquare_S_4,   'DiT-S/8':  DiTNonSquare_S_8,
}
