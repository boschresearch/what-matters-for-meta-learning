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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import save_config
from collections import OrderedDict

from networks.models import NPDecoder, EncoderFC
from networks.bbb import BBBConv2d, BBBLinear, ModuleWrapper, FlattenLayer


def conv_block(in_channels, out_channels, **kwargs):
    return nn.Sequential(OrderedDict([
        ('conv', BBBConv2d(in_channels, out_channels, **kwargs)),
        # ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
        #     track_running_stats=False)),
        ('relu', nn.ReLU()),
        # ('pool', nn.MaxPool2d(2))
    ]))


class BBBEncoder(ModuleWrapper):
    def __init__(self, img_channels, dim_w, device):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ('layer1', conv_block(img_channels, 32, kernel_size=3,
                                  stride=2, padding=1, bias=True, device=device)),
            ('layer2', conv_block(32, 48, kernel_size=3,
                                  stride=2, padding=1, bias=True, device=device)),
            ('pool', nn.MaxPool2d((2, 2))),
            ('layer3', conv_block(48, 64, kernel_size=3,
                                  stride=2, padding=1, bias=True, device=device)),
            ('flatten', FlattenLayer(4096)),
            ('linear', BBBLinear(4096, dim_w, bias=True, device=device))
        ]))


class CNPMR(nn.Module):
    """
    Conditional Neural Process
    """
    def __init__(self, config):
        super(CNPMR, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.agg_mode = config.agg_mode
        self.img_agg = config.img_agg
        self.y_dim = config.output_dim
        self.dim_w = config.dim_w
        self.n_hidden_units_r = config.n_hidden_units_r
        self.dim_r = config.dim_r
        self.dim_z = config.dim_z
        seed = config.seed
        torch.manual_seed(seed)  # make network initialization fixed

        # use same architecture as literatures
        self.encoder_w0 = BBBEncoder(self.img_channels, self.dim_w, device=self.device)

        self.transform_y = nn.Linear(self.label_dim, self.dim_w // 4)

        self.encoder_r = EncoderFC(input_dim=self.dim_w + self.dim_w // 4,
                                   n_hidden_units_r=self.n_hidden_units_r, dim_r=self.dim_r)

        self.r_to_z = nn.Linear(self.dim_r, self.dim_z)

        self.decoder0 = nn.Sequential(
            nn.Linear(self.dim_w + self.dim_z, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.y_dim)
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(256 + self.label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        if self.agg_mode == "baco":
            self.rs_to_mu = nn.Linear(256, 256)
            self.rs_to_var = nn.Linear(256, 256)

        self.mu = nn.Linear(256, 256)
        self.decoder = NPDecoder(aggregate=self.img_agg, output_dim=self.y_dim, task_num=self.task_num, img_channels=self.img_channels, img_size=self.img_size)

    def baco(self, mu, r_sigma):
        """

        :param mu: mean value
        :param r_sigma: variance
        :return:
        """
        ctx_num = mu.shape[1]
        self.r_dim = mu.shape[2]
        mu_z = torch.ones(self.task_num, self.r_dim).to(self.device) * 0.0  # initial mu_z is 0, shape [task_num, r_dim]
        sigma_z = torch.ones(self.task_num, self.r_dim).to(self.device) * 1.0  # initial sigma is 1, shape [task_num, r_dim]

        v = mu - mu_z[:, None, :].repeat(1, ctx_num, 1)
        sigma_inv = 1 / r_sigma
        sigma_z = 1 / (1 / sigma_z + torch.sum(sigma_inv, dim=1))
        mu_z = mu_z + sigma_z * torch.sum(sigma_inv * v, dim=1)
        return mu_z, sigma_z

    def forward(self, batch_train_images, label_train, batch_test_images, test=False):
        """

        :param img_context: context images
        :param img_target: target image
        :param y_target: target label (bar length)
        :return:
        """
        self.test_num = batch_test_images.shape[1]
        self.ctx_num = batch_train_images.shape[1]

        if self.ctx_num:
            batch_train_images = batch_train_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
            x_ctx, _ = self.encoder_w0(batch_train_images)
            x_ctx = x_ctx.reshape(self.task_num, self.ctx_num, self.dim_w)
            label_train = self.transform_y(label_train)
            x = torch.cat([x_ctx, label_train], dim=2)

            rs = self.encoder_r(x)
            # aggregate
            if self.agg_mode == 'mean':
                r = torch.mean(rs, dim=1, keepdim=False)
                z = self.r_to_z(r)[:, None, :].repeat(1, self.test_num, 1)
            elif self.agg_mode == 'max':
                r = torch.max(rs, dim=1, keepdim=False)[0]
                z = self.r_to_z(r)[:, None, :].repeat(1, self.test_num, 1)
            elif self.agg_mode == 'baco':
                mu = self.rs_to_mu(rs)
                log_variance = self.rs_to_var(rs)
                variance = 1e-5 + F.softplus(log_variance)
                r, log_variance = self.baco(mu, variance)
                z = self.r_to_z(r)[:, None, :].repeat(1, self.test_num, 1)
            else:
                raise TypeError("agg_mode is not applicable for CNP, choose from ['mean', 'max', 'baco']")
        else:
            z = torch.ones(self.task_num, self.test_num, self.dim_z).to(self.device) * 0.0

        batch_test_images = batch_test_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
        x_qry, kl = self.encoder_w0(batch_test_images)
        x_qry = x_qry.reshape(self.task_num, self.test_num, self.dim_w)
        x_qry = torch.cat([x_qry, z], dim=-1)

        pr_y_mu = self.decoder0(x_qry)
        pr_y_var = None
        return pr_y_mu, pr_y_var, kl




