import math

import torch


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    assert pos_tensor.ndim == 3

    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000**(2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_sineembed_for_relation(pos_tensor, heading_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    # assert pos_tensor.ndim == 4  # (B, N, N, 2)

    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)

    # sliced_hidden_dim = hidden_dim // 3  # Devided by 3 now

    scale = 2 * math.pi
    dim_t = torch.arange(hidden_dim, dtype=heading_tensor.dtype, device=heading_tensor.device)
    dim_t = 10000**(2 * (dim_t // 2) / hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    h_embed = heading_tensor * scale
    pos_h = h_embed[..., None] / dim_t
    pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)

    pe = torch.cat((pos_x, pos_y, pos_h), dim=-1)

    # print(111)

    # Concatenate position and heading tensors
    # combined_tensor = torch.cat((pos_tensor, heading_tensor), dim=-1)
    #
    # B, N, _, _ = combined_tensor.shape
    # half_hidden_dim = hidden_dim // 3  # Divided by 3 because we now have x, y, and heading components
    # scale = 2 * math.pi
    #
    # # Create a tensor of dimension indices scaled according to their position
    # dim_t = torch.arange(half_hidden_dim, dtype=combined_tensor.dtype, device=combined_tensor.device)
    # dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    #
    # # Scale and embed each component separately
    # combined_embed = combined_tensor * scale
    # combined_embed = combined_embed / dim_t
    #
    # # Apply sine and cosine alternately across the last dimension
    # embed_sin = combined_embed[..., 0::2].sin()
    # embed_cos = combined_embed[..., 1::2].cos()
    # combined_embed = torch.stack((embed_sin, embed_cos), dim=-1).flatten(-2)

    return pe
