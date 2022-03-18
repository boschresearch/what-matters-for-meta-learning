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
from networks.models import ImageEncoder, NPDecoder


class CondNeuralProcess(nn.Module):
    """
    Conditional Neural Process for ShapeNet3D
    """
    def __init__(self, config):
        super(CondNeuralProcess, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2] - 1 if config.task == "shapenet_3d" else self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.agg_mode = config.agg_mode
        self.img_agg = config.img_agg
        self.y_dim = config.output_dim
        seed = config.seed
        torch.manual_seed(seed)  # make network initialization fixed

        self.img_encoder = ImageEncoder(aggregate=self.img_agg, task_num=self.task_num, img_channels=self.img_channels)
        self.task_encoder = nn.Sequential(
            nn.Linear(256 + self.label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        if self.agg_mode == "baco":
            self.latent_mu = nn.Linear(256, 256)
            self.latent_var = nn.Linear(256, 256)

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
            x_ctx = self.img_encoder(batch_train_images)
            x = torch.cat([x_ctx, label_train], dim=2)
            context_features = self.task_encoder(x)
            # aggregate
            if self.agg_mode == 'mean':
                context_features = torch.mean(context_features, dim=1, keepdim=False)
                mu = self.mu(context_features)
                sample_features = mu[:, None, :].repeat(1, self.test_num, 1)
            elif self.agg_mode == 'max':
                context_features = torch.max(context_features, dim=1, keepdim=False)[0]
                mu = self.mu(context_features)
                sample_features = mu[:, None, :].repeat(1, self.test_num, 1)
            elif self.agg_mode == 'baco':
                mu = self.latent_mu(context_features)
                log_variance = self.latent_var(context_features)
                variance = 1e-5 + F.softplus(log_variance)
                mu, log_variance = self.baco(mu, variance)
                mu = self.mu(mu)
                sample_features = mu[:, None, :].repeat(1, self.test_num, 1)
            else:
                raise TypeError("agg_mode is not applicable for CNP, choose from ['mean', 'max', 'baco']")
        else:
            sample_features = torch.ones(self.task_num, self.test_num, 256).to(self.device) * 0.0
            # log_variance = torch.ones(self.task_num, 1, 256).to(self.device) * 1.0

        generated_angles, generated_var = self.decoder(batch_test_images, sample_features)

        kl = 0

        return generated_angles, generated_var, kl




