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
from collections import OrderedDict

from utils import save_config
from networks.models import NPDecoder, EncoderFC, AttnLinear
from networks.bbb import BBBConv2d, BBBLinear, ModuleWrapper, FlattenLayer


def conv3x3(in_planes, out_planes, stride=1, **kwargs):
    """3x3 convolution with padding"""
    return BBBConv2d(in_planes, out_planes, stride=stride, **kwargs)


def conv1x1(in_planes, out_planes, stride=1, **kwargs):
    """1x1 convolution"""
    return BBBConv2d(in_planes, out_planes, stride=stride, **kwargs)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, **kwargs)
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(inplanes, inplanes, stride, **kwargs),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def conv_block(in_channels, out_channels, **kwargs):
    return nn.Sequential(OrderedDict([
        ('conv', BBBConv2d(in_channels, out_channels, **kwargs)),
        # ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
        #     track_running_stats=False)),
        ('relu', nn.ReLU()),
        # ('pool', nn.MaxPool2d(2))
    ]))


class BBBEncoder(ModuleWrapper):
    def __init__(self, img_channels, device):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ('layer1', conv_block(img_channels, 64, kernel_size=5,
                                  stride=2, padding=2, bias=True, device=device)),
            ('layer2', BasicBlock(64, 64, kernel_size=3, stride=2, padding=1, bias=True, device=device)),
            ('layer3', BasicBlock(64, 64, kernel_size=3, stride=2, padding=1, bias=True, device=device)),
            ('layer4', BasicBlock(64, 64, kernel_size=3, stride=2, padding=1, bias=True, device=device)),
            ('layer5', BasicBlock(64, 64, kernel_size=3, stride=2, padding=1, bias=True, device=device)),
            ('flatten', FlattenLayer(256)),
        ]))


class ANPMRShapeNet3D(nn.Module):
    """
    Conditional Neural Process
    """
    def __init__(self, config):
        super(ANPMRShapeNet3D, self).__init__()
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

        # self.img_encoder = ImageEncoder(aggregate=self.img_agg, task_num=self.task_num, img_channels=self.img_channels)
        self.img_encoder = BBBEncoder(img_channels=self.img_channels, device=self.device)

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

    def pixel_agg(self, x):

        if self.img_agg == "mean":
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
        elif self.img_agg == "max":
            x = nn.AdaptiveMaxPool2d((2, 2))(x)
        elif self.img_agg == "baco":
            x = nn.AdaptiveMaxPool2d((2, 2))(x)
        elif self.img_agg == "reshape":
            x = x.reshape(x.size(0), -1)
        else:
            raise TypeError("Non-valid img_agg!")
        x = x.reshape(x.size(0), -1)
        x = x.view(self.task_num, -1, x.size(1))

        return x

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
            batch_test_images = batch_test_images.reshape(-1, self.img_channels, self.img_size[0], self.img_size[1])
            x_ctx, _ = self.img_encoder(batch_train_images)
            x_tgt, kl = self.img_encoder(batch_test_images)
            x_ctx = self.pixel_agg(x_ctx)
            x_tgt = self.pixel_agg(x_tgt)

            x = torch.cat([x_ctx, label_train], dim=2)
            context_features = self.task_encoder(x)

            # attention
            context_features = self._attention_func(x_ctx, context_features, x_tgt)
            mu = self.mu(context_features)
            sample_features = mu
        else:
            sample_features = torch.ones(self.task_num, self.test_num, 256).to(self.device) * 0.0
            # log_variance = torch.ones(self.task_num, self.test_num, 256).to(self.device) * 1.0
            kl=0

        generated_angles, generated_var = self.decoder(batch_test_images, sample_features)

        return generated_angles, generated_var, kl

