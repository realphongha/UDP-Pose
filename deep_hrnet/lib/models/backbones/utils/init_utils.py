#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
import logging
from typing import Optional
from torch.nn import (
    BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm, InstanceNorm1d, InstanceNorm2d, GroupNorm
)


class GroupLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_groups: int,
                 bias: Optional[bool] = True,
                 feature_shuffle: Optional[bool] = False,
                 *args, **kwargs):
        """
            Applies a group linear transformation as defined in the following papers:
                https://arxiv.org/abs/1808.09029
                https://arxiv.org/abs/1911.12385
                https://arxiv.org/abs/2008.00623

            :param in_features: size of each input sample
            :param out_features: size of each output sample
            :param n_groups: Number of groups
            :param bias: Add bias (learnable) or not
            :param feature_shuffle: Mix output of each group after group linear transformation
            :param is_ws: Standardize weights or not (experimental)
        """
        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            logging.error(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            logging.error(err_msg)

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        super(GroupLinear, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        self.out_features = out_features
        self.in_features = in_features
        self.n_groups = n_groups
        self.feature_shuffle = feature_shuffle

        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def _forward(self, x: Tensor) -> Tensor:
        """
        :param x: Tensor of shape [B, N] where B is batch size and N is the number of input features
        :return:
            Tensor of shape [B, M] where M is the number of output features
        """

        bsz = x.shape[0]
        # [B, N] -->  [B, g, N/g]
        x = x.reshape(bsz, self.n_groups, -1)

        # [B, g, N/g] --> [g, B, N/g]
        x = x.transpose(0, 1)
        # [g, B, N/g] x [g, N/g, M/g] --> [g, B, M/g]
        x = torch.bmm(x, self.weight)

        if self.bias is not None:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g, B, M/g] --> [B, M/g, g]
            x = x.permute(1, 2, 0)
            # [B, M/g, g] --> [B, g, M/g]
            x = x.reshape(bsz, self.n_groups, -1)
        else:
            # [g, B, M/g] --> [B, g, M/g]
            x = x.transpose(0, 1)

        return x.reshape(bsz, -1)

    def _glt_transform(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = self._forward(x)
            return x
        elif x.dim() == 3:
            dim_0, dim_1, inp_dim = x.size()
            x = x.reshape(dim_1 * dim_0, -1)
            x = self._forward(x)
            x = x.reshape(dim_0, dim_1, -1)
            return x
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self._glt_transform(x)


class LinearLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: Optional[bool] = True,
                 *args, **kwargs) -> None:
        """
            Applies a linear transformation to the input data

            :param in_features: size of each input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x


supported_conv_inits = ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform', 'normal', 'trunc_normal']
supported_fc_inits = ['kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform', 'normal', 'trunc_normal']
norm_layers_tuple = (BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm, InstanceNorm1d, InstanceNorm2d, GroupNorm)

def _init_nn_layers(module, init_method: Optional[str] = 'kaiming_normal', std_val: Optional[float] = None):
    init_method = init_method.lower()
    if init_method == 'kaiming_normal':
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'kaiming_uniform':
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'xavier_normal':
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'xavier_uniform':
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'normal':
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == 'trunc_normal':
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        supported_conv_message = 'Supported initialization methods are:'
        for i, l in enumerate(supported_conv_inits):
            supported_conv_message += '\n \t {}) {}'.format(i, l)
        logging.error('{} \n Got: {}'.format(supported_conv_message, init_method))


def initialize_conv_layer(module, init_method='kaiming_normal', std_val: float = 0.01):
    _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_fc_layer(module, init_method='normal', std_val: float = 0.01):
    if hasattr(module, "layer"):
        _init_nn_layers(module=module.layer, init_method=init_method, std_val=std_val)
    else:
        _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_norm_layers(module):
    def _init_fn(module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    _init_fn(module.layer) if hasattr(module, "layer") else _init_fn(module=module)


def initialize_weights(opts, modules):
    # weight initialization
    conv_init_type = getattr(opts, "model.layer.conv_init", "kaiming_normal")
    linear_init_type = getattr(opts, "model.layer.linear_init", "normal")

    conv_std = getattr(opts, "model.layer.conv_init_std_dev", None)
    linear_std = getattr(opts, "model.layer.linear_init_std_dev", 0.01)
    group_linear_std = getattr(opts, "model.layer.group_linear_init_std_dev", 0.01)

    for m in modules:
        if isinstance(m, nn.Conv2d):
            initialize_conv_layer(module=m, init_method=conv_init_type, std_val=conv_std)
        elif isinstance(m, norm_layers_tuple):
            initialize_norm_layers(module=m)
        elif isinstance(m, (nn.Linear, LinearLayer)):
            initialize_fc_layer(module=m, init_method=linear_init_type, std_val=linear_std)
        elif isinstance(m, GroupLinear):
            initialize_fc_layer(module=m, init_method=linear_init_type, std_val=group_linear_std)