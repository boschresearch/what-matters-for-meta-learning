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
import pickle
import torch
import pickle
import sys
import os
from PIL import Image
from utils import convert_channel_last_np_to_tensor, PascalAugmenter
from dataset import BaseData

"""
   Supporting methods for data handling
"""


class Pascal1D(BaseData):
    """
        Pascal1D dataset for pose estimation

    """
    def __init__(self, path, img_size, seed, aug=[]):
        super().__init__(img_size)
        self.num_classes = 1
        self.alpha = 0.3

        assert set(aug).issubset({"MR", "data_aug", "task_aug"})
        self.aug_list = aug
        if "data_aug" in self.aug_list:
            self.Augmentor = PascalAugmenter()
            self.data_aug = True
        if "task_aug" in self.aug_list:
            self.task_aug = True

        self.x_train, self.y_train = pickle.load(open(os.path.join(path, "train_data_ins.pkl"), 'rb'))
        self.x_val, self.y_val = pickle.load(open(os.path.join(path, "val_data_ins.pkl"), 'rb'))

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.y_train = self.y_train[:, :, -1, None]
        self.x_val, self.y_val = np.array(self.x_val), np.array(self.y_val)
        self.y_val = self.y_val[:, :, -1, None]

        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)

        print(f"Training Set Size = {self.x_train.shape[0]}, Training samples per task: {self.x_train.shape[1]}")
        print(f"Validation Set Size = {self.x_val.shape[0]}, Validation samples per task: {self.x_val.shape[1]}")
        np.random.seed(seed)

    def get_batch(self, source, tasks_per_batch, shot):
        """Get data batch."""
        xs, ys, xq, yq = [], [], [], []
        if source == 'train':
            x, y = self.x_train, self.y_train
        elif source == 'validation':
            x, y = self.x_val, self.y_val

        for _ in range(tasks_per_batch):
            # sample tasks
            classes = np.random.choice(
                range(np.shape(x)[0]), self.num_classes, replace=False)

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []
            for k in list(classes):
                # sample SHOT and QUERY instances
                idx = np.random.choice(
                    range(np.shape(x)[1]),
                    size=shot + shot,  # follow original authors using fixed number for shot and query
                    replace=False)
                x_k = x[k][idx]
                y_k = y[k][idx]

                support_set.append(x_k[:shot])
                query_set.append(x_k[shot:])
                support_sety.append(y_k[:shot])
                query_sety.append(y_k[shot:])

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        xs = np.reshape(
            xs,
            [tasks_per_batch, shot * self.num_classes, *self.img_size])
        xq = np.reshape(
            xq,
            [tasks_per_batch, shot * self.num_classes, *self.img_size])

        if self.data_aug and source == 'train':
            xs = self.Augmentor.generate(xs)
            xq = self.Augmentor.generate(xq)
            # plot image for debug
            # tmp_img = Image.fromarray(xs[0, 0, :, :, 0])
            # tmp_img.show()

        if self.task_aug and source == 'train':
            noise = [0, 0.25, 0.50, 0.75]
            y_noise = np.random.choice(noise, (tasks_per_batch, 1))[:, None, :]
            # y_noise = np.random.uniform(-1, 1, size=(tasks_per_batch, 1))[:, None, :] * 10.0 * self.alpha
            ys += y_noise
            yq += y_noise
            ys %= 1.0
            yq %= 1.0

        xs = xs.astype(np.float32) / 255.0
        xq = xq.astype(np.float32) / 255.0
        ys = ys.astype(np.float32) * 10.0
        yq = yq.astype(np.float32) * 10.0

        xs = convert_channel_last_np_to_tensor(xs)
        xq = convert_channel_last_np_to_tensor(xq)
        return xs, xq, torch.from_numpy(ys).type(torch.FloatTensor), torch.from_numpy(yq).type(torch.FloatTensor)

    def gen_bg(self, config, data="all"):
        pass
