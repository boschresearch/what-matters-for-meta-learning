#   Copyright (c) 2022 Robert Bosch GmbH
#   Author: Ning Gao
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

#   This source code is derived from Torchmeta (https://github.com/tristandeleu/pytorch-meta)
#   Copyright (c) 2019 Tristan Deleu, licensed under the MIT license,
#   cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        # ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
        #     track_running_stats=False)),
        ('relu', nn.ReLU()),
        # ('pool', nn.MaxPool2d(2))
    ]))


def conv_feature_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        # ('pool', nn.MaxPool2d(2))
    ]))


class MAMLShapeNet1D(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].
    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.
    out_features : int
        Number of classes (output of the model).
    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.
    feature_size : int (default: 64)
        Number of features returned by the convolutional head.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, config):
        super(MAMLShapeNet1D, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.dim_hidden = config.dim_hidden
        self.agg_mode = config.agg_mode
        self.img_agg = config.img_agg
        self.output_dim = config.output_dim
        self.dim_w = config.dim_w
        self.img_w_size = int(np.sqrt(self.dim_w))
        self.n_hidden_units_r = config.n_hidden_units_r
        self.dim_r = config.dim_r
        self.dim_z = config.dim_z
        seed = config.seed
        torch.manual_seed(seed)  # make network initialization fixed

        self.encoder_w = MetaSequential(OrderedDict([
            ('layer1', conv_block(self.img_channels, 32, kernel_size=3,
                                  stride=2, padding=1, bias=True)),
            ('layer2', conv_block(32, 48, kernel_size=3,
                                  stride=2, padding=1, bias=True)),
            ('pool', nn.MaxPool2d((2, 2))),
            ('layer3', conv_block(48, 64, kernel_size=3,
                                  stride=2, padding=1, bias=True)),
            ('flatten', nn.Flatten()),
            ('linear', MetaLinear(4096, self.dim_w, bias=True))
        ]))

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_feature_block(self.img_channels, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True)),
            ('layer2', conv_feature_block(self.dim_hidden, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True)),
            ('layer3', conv_feature_block(self.dim_hidden, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True)),
            ('layer4', conv_feature_block(self.dim_hidden, self.dim_hidden, kernel_size=3,
                                          stride=1, padding=1, bias=True)),
            ('pool', nn.AdaptiveAvgPool2d(1)),
        ]))

        self.regressor = MetaSequential(OrderedDict([
            ('regressor', MetaLinear(self.dim_hidden, self.output_dim, bias=True)),
            ('tanh', nn.Tanh()),
        ]))

    def forward(self, inputs, params=None):
        # features = self.features(inputs, params=self.get_subdict(params, 'features'))
        # features = features.view((features.size(0), -1))
        # logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        self.ctx_num = inputs.shape[0]

        if self.ctx_num:
            inputs = inputs.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
            inputs = self.encoder_w(inputs, params=self.get_subdict(params, 'encoder_w')).reshape(-1, self.img_channels, self.img_w_size, self.img_w_size)

            inputs = self.features(inputs, params=self.get_subdict(params, 'features')).reshape(self.ctx_num, self.dim_hidden)
            outputs = self.regressor(inputs, params=self.get_subdict(params, 'regressor'))
        else:
            raise ValueError("0 context is sampled!")

        return outputs, 0