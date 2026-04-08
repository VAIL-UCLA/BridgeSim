import collections.abc
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class Tokenizer(nn.Module):
    def __init__(self, num_actions, d_model, add_one_more_action=True):
        super(Tokenizer, self).__init__()
        if add_one_more_action:
            self.tokens = nn.Embedding(
                num_actions + 1, d_model
            )  # The last token is used for the dummy token at step=0.
        else:
            self.tokens = nn.Embedding(num_actions, d_model)
        self.num_actions = num_actions
        self.add_one_more_action = add_one_more_action

    def forward(self, actions):
        new_actions = actions.clone()
        if self.add_one_more_action:
            new_actions[actions == -1] = self.num_actions
        else:
            new_actions[actions == -1] = self.num_actions - 1
        return self.tokens(new_actions)


# def get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# class SquaredReLU(nn.Module):
#     def forward(self, x):
#         return torch.square(F.relu(x))


def build_mlps(c_in, mlp_channels, ret_before_act=False, without_norm=False, is_v7=None, zero_init=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k]))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k]), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        # nn.BatchNorm1d(mlp_channels[k]),
                        nn.LayerNorm(mlp_channels[k]),
                        nn.ReLU(),
                    ]
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


class Mlp(nn.Module):
    """Copied from  https://github.com/huggingface/pytorch-image-models/blob/4d4bdd64a996bf7b5919ec62f20af4a1c07d5848/timm/layers/mlp.py#L13"""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
        use_conv=False,
        is_v7=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # if is_v7:
        #     bias = False

        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        # self.use_squared_relu = is_v7
        # if is_v7:
        # Use relu:
        # self.act = nn.ReLU()
        # else:
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        # if is_v7:
        #     self.fc2.weight.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # if self.use_squared_relu:
        #     x = torch.square(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
