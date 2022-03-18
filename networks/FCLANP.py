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
from torch import nn
import torch.nn.functional as F

from networks.models import AttnLinear, ImageEncoder, NPDecoder
from networks.fast_attention import FastAttention
from trainer.losses import LossFunc as losses


class FCLANP(nn.Module):
    """
    Conditional Neural Process using cross attention for ShapeNet3D with Functional Contrastive Learning
    """
    def __init__(self, config):
        super(FCLANP, self).__init__()
        self.device = config.device
        self.img_size = config.img_size
        self.img_channels = self.img_size[2] - 1 if config.task == "shapenet_3d" else self.img_size[2]
        self.task_num = config.tasks_per_batch
        self.label_dim = config.input_dim
        self.agg_mode = config.agg_mode
        self.img_agg = config.img_agg
        self.y_dim = config.output_dim
        self.temperature = config.temperature
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
        self.mu = nn.Linear(256, 256)

        self.decoder = NPDecoder(aggregate=self.img_agg, output_dim=self.y_dim, task_num=self.task_num, img_channels=self.img_channels, img_size=self.img_size)

        # attention block
        h_dim = 256
        n_heads = 8
        self._W_k = nn.ModuleList(
            [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
        )
        self._W_v = nn.ModuleList(
            [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
        )
        self._W_q = nn.ModuleList(
            [AttnLinear(h_dim, h_dim) for _ in range(n_heads)]
        )
        self._W = AttnLinear(n_heads * h_dim, h_dim)
        self.attn = FastAttention(dim_heads=256,
                             # nb_features=nb_features,
                             causal=False)
        self._attention_func = self._multihead_attention
        self.n_heads = n_heads

    def _multihead_attention(self, k, v, q):
        k_all = []
        v_all = []
        q_all = []

        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)

            k_all.append(k_)
            v_all.append(v_)
            q_all.append(q_)

            #out = self._dot_attention(k_, v_, q_)
            #outs.append(out)
        k_all = torch.stack(k_all, dim=1)
        v_all = torch.stack(v_all, dim=1)
        q_all = torch.stack(q_all, dim=1)
        outs = self.attn(q=q_all, k=k_all, v=v_all)
        outs = outs.permute(0,2,3,1).contiguous()
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def forward(self, batch_train_images, label_train, batch_test_images, label_test, test=False):
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
            batch_test_images = batch_test_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
            x_ctx = self.img_encoder(batch_train_images)
            x_tgt = self.img_encoder(batch_test_images)

            x = torch.cat([x_ctx, label_train], dim=2)
            context_features = self.task_encoder(x)

            # attention
            context_features = self._attention_func(x_ctx, context_features, x_tgt)
            mu = self.mu(context_features)
            sample_features = mu
        else:
            sample_features = torch.ones(self.task_num, self.test_num, 256).to(self.device) * 0.0
            # log_variance = torch.ones(self.task_num, self.test_num, 256).to(self.device) * 1.0

        generated_angles, generated_var = self.decoder(batch_test_images, sample_features)
        if not test:
            contra_loss = losses.contrastive_loss_ANP(sample_features, t=self.temperature)
        else:
            contra_loss = 0

        kl = 0
        return generated_angles, generated_var, kl, contra_loss

