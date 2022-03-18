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
import random
import imgaug.augmenters as iaa
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import torch
from collections import OrderedDict


def convert_channel_last_np_to_tensor(input):
    """input: [task_num, samples_num, height, width, channel]"""
    input = torch.from_numpy(input).type(torch.FloatTensor)
    input = input.permute(0, 1, 4, 2, 3).contiguous()
    return input


def task_augment(batch_train_Q, batch_test_Q, azimuth_only=False):
    # add random noise to each task.
    num_task = batch_train_Q.shape[0]
    q_train, q_test = [], []
    for i in range(num_task):
        noise_azimuth = np.random.randint(-10, 20)
        if azimuth_only:
            noise_ele = 0
        else:
            noise_ele = np.random.randint(-5, 10)
        # adapt train
        r = R.from_quat(batch_train_Q[i])
        e = r.as_euler('ZYX', degrees=True)
        e[:, 0] += noise_ele
        e[:, 2] -= noise_azimuth
        q_train.append(R.from_euler('ZYX', e, degrees=True).as_quat())
        # adapt test
        r = R.from_quat(batch_test_Q[i])
        e = r.as_euler('ZYX', degrees=True)
        e[:, 0] += noise_ele
        e[:, 2] -= noise_azimuth
        q_test.append(R.from_euler('ZYX', e, degrees=True).as_quat())

    q_train = np.array(q_train)
    q_test = np.array(q_test)
    return q_train, q_test


def shuffle_batch(images, R, Q, T):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], R[permutation], Q[permutation], T[permutation]


def convert_index_to_angle(index, num_instances_per_item):
    """
    Convert the index of an image to a representation of the angle
    :param index: index to be converted
    :param num_instances_per_item: number of images for each item
    :return: a biterion representation of the angle
    """
    degrees_per_increment = 360./num_instances_per_item
    angle = index * degrees_per_increment
    angle_radians = np.deg2rad(angle)
    return angle, np.sin(angle_radians), np.cos(angle_radians)


def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()


