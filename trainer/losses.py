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

import math
import torch
from pytorch_metric_learning.losses import NTXentLoss


class LossFunc():
    def __init__(self, loss_type, task):
        """
        loss_type: [mse, nll]
        task: ["shapenet_3d", "bars", "distractor", "pascal_1d"]
        """
        
        self.loss_type = loss_type
        self.task = task
    
    def calc_loss(self, pr_mu, pr_var, gt_y, test=False):
        if self.loss_type == "mse":
            if self.task == 'distractor':
                loss = torch.sqrt(torch.sum((gt_y - pr_mu) ** 2, dim=-1))
                loss = torch.mean(loss)

            if self.task == "shapenet_3d":
                loss = self.quaternion_loss(gt_y, pr_mu)
            elif self.task == "shapenet_1d":
                if not test:
                    loss = self.azimuth_loss(gt_y, pr_mu)
                if test:
                    loss = self.degree_loss(gt_y, pr_mu)

            elif self.task == "pascal_1d":
                loss = self.mean_square_loss(gt_y, pr_mu)
            return loss

    def quaternion_loss(self, q_gt, q_pr):
        q_pr_norm = torch.sqrt(torch.sum(q_pr ** 2, dim=-1, keepdim=True))
        q_pr = q_pr / q_pr_norm
        pos_gt_loss = torch.abs(q_gt - q_pr).sum(dim=-1)
        neg_gt_loss = torch.abs(-q_gt - q_pr).sum(dim=-1)
        L1_loss = torch.minimum(pos_gt_loss, neg_gt_loss)
        L1_loss = L1_loss.mean()
        return L1_loss

    def azimuth_loss(self, q_gt, q_pr):
        loss = torch.mean(torch.sum((q_gt[..., :2] - q_pr) ** 2, dim=-1))
        return loss

    def degree_loss(self, q_gt, q_pr):
        q_gt = torch.rad2deg(q_gt[..., -1])
        pr_cos = q_pr[..., 0]
        pr_sin = q_pr[..., 1]
        ps_sin = torch.where(pr_sin >= 0)
        ng_sin = torch.where(pr_sin < 0)
        pr_deg = torch.acos(pr_cos)
        pr_deg_ng = -torch.acos(pr_cos) + 2 * math.pi
        pr_deg[ng_sin] = pr_deg_ng[ng_sin]
        pr_deg = torch.rad2deg(pr_deg)
        errors = torch.stack((torch.abs(q_gt - pr_deg), torch.abs(q_gt + 360.0 - pr_deg), torch.abs(q_gt - (pr_deg + 360.0))), dim=-1)
        errors, _ = torch.min(errors, dim=-1)
        losses = torch.mean(errors)
        return losses

    def mean_square_loss(self, q_gt, q_pr):
        loss = torch.mean((q_gt - q_pr)**2)
        return loss

    @staticmethod
    def contrastive_loss(z_1, z_2, t=0.07):
        loss_func = NTXentLoss(temperature=t)
        z = torch.cat((z_1, z_2), dim=0)
        labels = torch.cat((torch.arange(z_1.shape[0]), torch.arange(z_2.shape[0])), dim=0)
        loss = loss_func(z, labels)
        return loss

    @staticmethod
    def contrastive_loss_ANP(z, t=0.07):
        # ind = np.random.permutation(z.shape[1])[:10]
        # z = z[:, ind, :]
        loss_func = NTXentLoss(temperature=t)
        labels = torch.arange(z.shape[0]).reshape(-1, 1).repeat(1, z.shape[1])
        z = z.reshape(-1, z.shape[-1])
        labels = labels.reshape(-1, 1).squeeze()
        loss = loss_func(z, labels)
        return loss