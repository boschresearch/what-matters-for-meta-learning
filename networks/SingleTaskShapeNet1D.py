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

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.models import EncoderFC, AttnLinear


class SingleTaskShapeNet1D(nn.Module):
    """
    Conditional Neural Process
    """
    def __init__(self, config):
        super(SingleTaskShapeNet1D, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.y_dim = config.output_dim
        self.dim_w = config.dim_w
        self.n_hidden_units_r = config.n_hidden_units_r
        self.dim_r = config.dim_r
        self.dim_z = config.dim_z
        seed = config.seed
        torch.manual_seed(seed)  # make network initialization fixed

        # use same architecture as literatures
        self.encoder_w0 = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4096, self.dim_w)
        )

        self.encoder_r = EncoderFC(input_dim=self.dim_w, n_hidden_units_r=self.n_hidden_units_r, dim_r=self.dim_r)

        self.r_to_z = nn.Linear(self.dim_r, self.dim_z)

        self.decoder0 = nn.Sequential(
            nn.Linear(self.dim_w + self.dim_z, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.y_dim),
            nn.Tanh(),
        )


    def forward(self, batch_train_images, label_train, batch_test_images, test=False):
        """

        :param img_context: context images
        :param img_target: target image
        :param y_target: target label (bar length)
        :return:
        """
        self.test_num = batch_test_images.shape[1]

        batch_test_images = batch_test_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
        x = self.encoder_w0(batch_test_images).reshape(self.task_num, self.test_num, self.dim_w)
        r = self.encoder_r(x)
        z = self.r_to_z(r)
        x = torch.cat([x, z], dim=-1)

        pr_y_mu = self.decoder0(x)
        pr_y_var = None

        kl = 0
        return pr_y_mu, pr_y_var, kl


