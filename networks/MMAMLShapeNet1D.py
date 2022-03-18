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

#   This source code is derived from MMAML-Classification (https://github.com/shaohua0116/MMAML-Classification)
#   Copyright (c) 2018 Tristan Deleu, Risto Vuorio,
#   Copyright (c) 2019 Hexiang Hu, Shao-Hua Sun,
#   licensed under the MIT license,
#   cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import numpy as np
import torch
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from .gated_conv_net import GatedConvModel
from .conv_embedding_model import ConvEmbeddingModel


class MMAMLShapeNet1D(MetaModule):

    def __init__(self, config):
        super(MMAMLShapeNet1D, self).__init__()
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

        self.model = GatedConvModel(
            input_channels=self.img_size[2],
            output_size=config.output_dim,
            use_max_pool=False,
            num_channels=32,
            img_side_len=self.img_size[0],
            condition_type='affine',
            condition_order='low2high',
            verbose=False)
        self.model_parameters = list(self.model.parameters())

        self.embedding_model = ConvEmbeddingModel(
            input_size=np.prod((1, 128, 128)),
            output_size=config.output_dim,
            embedding_dims=[64, 128, 256, 512],
            hidden_size=128,
            num_layers=2,
            convolutional=True,
            num_conv=4,
            num_channels=32,
            rnn_aggregation=(not True),
            embedding_pooling='avg',
            batch_norm=True,
            avgpool_after_conv=True,
            linear_before_rnn=False,
            num_sample_embedding=0,
            sample_embedding_file='embeddings.hdf5',
            img_size=(1, 128, 128),
            verbose=False)
        self.embedding_parameters = list(self.embedding_model.parameters())

        self.optimizers = (torch.optim.Adam(self.model_parameters, lr=config.lr),
                      torch.optim.Adam(self.embedding_parameters, lr=config.lr))
