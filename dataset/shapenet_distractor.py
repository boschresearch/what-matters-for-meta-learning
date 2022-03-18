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


import os
import numpy as np
import torch
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

from dataset import BaseData
from utils import Augmenter, convert_channel_last_np_to_tensor

"""
   Supporting methods for data handling of Distractor task
"""


def shuffle_batch(images, centers):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], centers[permutation]


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


class AugmenterDistractor(Augmenter):
    def __init__(self):
        super(AugmenterDistractor, self).__init__()

        self.seq = iaa.Sequential(
            [
                self.sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),

                self.sometimes(
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.00, 0.05), size_percent=(0.02, 0.25),
                            per_channel=0.2
                        ),
                    ]),
                ),

            ],
            # do all of the above augmentations in random order
            random_order=True
        )


class ShapeNetDistractor(BaseData):
    """
        Class to handle ShapeNet dataset. Loads from numpy data as saved in data folder.
        categories = ['02691156': airplane [4045], '02828884': bench [1813], '02933112': cabinet[1571], '02958343': cars [3514],
        '02992529': handy [831], '03001627': chairs [6778], '03211117': monitor [1093], '03636649': lamp [2318],
        '03691459': speaker[1597], '04379243': table [8436]]
        test: ['04256520': sofa [3173], '04530566': ship [1939]]

    """
    def __init__(self, path, img_size, num_instances_per_item, train_fraction, val_fraction, seed, aug=[], mode='train', load_test_categ_only=False, test_categ=['04256520', '04530566']):

        """
        Initialize object to handle shapenet data
        :param path: directory of numpy file with preprocessed ShapeNet arrays.
        :param num_instances_per_item: Number of views of each model in the dataset.
        :param train_fraction: Fraction of models used for training.
        :param val_fraction: Fraction of models used for validation.
        :param seed: random seed for selecting data.
        :param mode: indicates either train or test.
        """
        super().__init__(img_size)
        assert set(aug).issubset(set(["data_aug", "task_aug"]))
        self.mode = mode
        self.aug_list = aug
        if "data_aug" in self.aug_list:
            self.Augmentor = AugmenterDistractor()
            self.data_aug = True
        if "task_aug" in self.aug_list:
            self.task_aug = True
            self.num_noise = 16

        self.angle_dimensionality = 3
        self.has_validation_set = True
        if load_test_categ_only:
            train_categories = []
            # data_train = np.zeros((5, 36, 0))
        else:
            train_categories = ['02691156', '02828884', '02933112', '02958343', '02992529', '03001627', '03211117', '03636649', '03691459', '04379243']
            # train_categories = ['02691156']  # used for testing if you don't have enough memory to load the data

        test_categories = test_categ
        # test_categories = ['04530566']  # used for testing if you don't have enough memory to load the data

        for category in train_categories:
            file = os.path.join(path, '{0:s}_multi.npy'.format(category))
            if category == train_categories[0]:  # first time through
                data_train = np.load(file, allow_pickle=True)
            else:
                data_train = np.concatenate((data_train, np.load(file, allow_pickle=True)), axis=0)

        for category in test_categories:
            file = os.path.join(path, '{0:s}_multi.npy'.format(category))
            if category == test_categories[0]:  # first time through
                data_test = np.load(file, allow_pickle=True)
            else:
                data_test = np.concatenate((data_test, np.load(file, allow_pickle=True)), axis=0)

        if load_test_categ_only:
            data_train = data_test

        self.instances_per_item = num_instances_per_item
        self.total_items = data_train.shape[0]

        train_size = (int) (train_fraction * self.total_items)
        val_size = (int) (val_fraction * self.total_items)
        test_size = data_test.shape[0]
        print("Training Set Size = {0:d}".format(train_size))
        print("Validation Set Size = {0:d}".format(val_size))
        print("Test Set Size = {0:d}".format(test_size))
        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        self.test_counter = 0
        np.random.seed(seed)
        np.random.shuffle(data_train)
        #np.random.shuffle(data_test)

        self.train_images, self.train_item_indices, self.train_item_angles, self.train_centers = self.__extract_data(data_train[:train_size])
        self.validation_images, self.validation_item_indices, self.validation_item_angles, self.validation_centers = \
            self.__extract_data(data_train[train_size:train_size + val_size])
        self.test_images, self.test_item_indices, self.test_item_angles, self.test_centers = self.__extract_data(data_test)
        self.train_item_sets = np.max(self.train_item_indices)
        self.validation_item_sets = np.max(self.validation_item_indices)
        self.test_item_sets = np.max(self.test_item_indices)

        # if self.mode == 'test':
        #     self.test_item_permutation = np.random.permutation(self.test_item_sets)
        #     self.test_counter = 0

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    def get_angle_dimensionality(self):
        return self.angle_dimensionality

    def get_has_validation_set(self):
        return self.has_validation_set

    def get_batch(self, source, tasks_per_batch, shot):
        """
        Wrapper function for batching in the model.
        :param source: train, validation or test (string).
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :return: np array representing a batch of tasks.
        """
        self.source = source
        if source == 'train':
            shot = np.random.randint(1, shot + 1)
            return self.__yield_random_task_batch(tasks_per_batch, self.train_images, self.train_item_angles,
                                                  self.train_item_indices, self.train_centers, shot)
        elif source == 'validation':
            return self.__yield_random_task_batch(tasks_per_batch, self.validation_images, self.validation_item_angles,
                                                  self.validation_item_indices, self.validation_centers, shot)
        elif source == 'test':
            self.test_item_permutation = np.random.permutation(self.test_item_sets)
            self.test_counter = 0
            return self.__yield_random_task_batch(tasks_per_batch, self.test_images, self.test_item_angles,
                                                  self.test_item_indices, self.test_centers, shot)

    def __yield_random_task_batch(self, num_tasks_per_batch, images, angles, item_indices, centers, num_train_instances):
        """
        Generate a batch of tasks from image set.
        :param num_tasks_per_batch: number of tasks per batch.
        :param images: images set to generate batch from.
        :param angles: associated angle for each image.
        :param item_indices: indices of each character.
        :param centers: object positions
        :param num_train_instances: number of training images per class.
        :return: a batch of tasks.
        """
        train_images_to_return, test_images_to_return = [], []
        # train_angles_to_return, test_angles_to_return = [], []  # distractor task does not need angles
        train_centers_to_return, test_centers_to_return = [], []
        for _ in range(num_tasks_per_batch):
            images_train, images_test, centers_train, centers_test =\
                self.__generateRandomTask(images, angles, item_indices, centers, num_train_instances)
            train_images_to_return.append(images_train)
            test_images_to_return.append(images_test)
            # train_angles_to_return.append(labels_train)
            # test_angles_to_return.append(labels_test)
            train_centers_to_return.append(centers_train)
            test_centers_to_return.append(centers_test)

        xs = 255 - np.array(train_images_to_return)
        xq = 255 - np.array(test_images_to_return)
        ys = np.array(train_centers_to_return)
        yq = np.array(test_centers_to_return)

        if self.data_aug and self.source == 'train':
            xs = self.Augmentor.generate(xs)
            xq = self.Augmentor.generate(xq)

        # # plot image for debug
        # for _ in range(20):
        #     tmp_img = Image.fromarray(xq[0, _, :, :, 0])
        #     tmp_img.show()

        if self.task_aug and self.source == 'train':
            noise = np.linspace(0, 16, self.num_noise+1)[:-1]
            y_noise = np.random.choice(noise, (num_tasks_per_batch, 2))[:, None, :]
            # y_noise = np.random.uniform(-1, 1, size=(tasks_per_batch, 1))[:, None, :] * 10.0 * self.alpha
            ys += y_noise
            yq += y_noise
            ys %= 128
            yq %= 128

        xs = xs.astype(np.float32) / 255.0
        xq = xq.astype(np.float32) / 255.0
        xs = convert_channel_last_np_to_tensor(xs)
        xq = convert_channel_last_np_to_tensor(xq)

        return xs, xq, torch.from_numpy(ys).type(torch.FloatTensor), torch.from_numpy(yq).type(torch.FloatTensor)

    def __generateRandomTask(self, images, angles, item_indices, centers, num_train_instances):
        """
        Randomly generate a task from image set.
        :param images: images set to generate batch from.
        :param angles: associated angle for each image.
        :param item_indices: indices of each character.
        :param centers: object positions
        :param num_train_instances: number of training images per class.
        :return: tuple containing train and test images and labels for a task.
        """
        if self.source == 'test':
            task_item = self.test_item_permutation[self.test_counter]
            self.test_counter = self.test_counter + 1
        else:
            task_item = np.random.choice(np.unique(item_indices))
        permutation = np.random.permutation(self.instances_per_item)

        # # add random behavior for dataset, i.e. the starting image is not always with 0 degree
        # permutation_angles = permutation + np.random.randint(self.instances_per_item)
        # permutation_angles[np.where(permutation_angles > self.instances_per_item -1)] -= self.instances_per_item

        item_images = images[np.where(item_indices == task_item)[0]][permutation]
        # item_angles = angles[np.where(item_indices == task_item)[0]][permutation]
        item_centers = centers[np.where(item_indices == task_item)[0]][permutation]
        train_images = item_images[:num_train_instances]
        train_centers = item_centers[:num_train_instances]
        if self.mode == 'eval':
            test_images = item_images
            test_centers = item_centers
        else:
            test_images = item_images[num_train_instances:]
            test_centers = item_centers[num_train_instances:]
        # train_images, train_angles, train_centers = item_images[:num_train_instances], item_angles[:num_train_instances], item_centers[:num_train_instances]
        # test_images, test_angles, test_centers = item_images[num_train_instances:], item_angles[num_train_instances:], item_centers[num_train_instances:]
        train_images_to_return, train_centers_to_return = shuffle_batch(train_images, train_centers)
        test_images_to_return, test_centers_to_return = shuffle_batch(test_images, test_centers)
        return train_images_to_return, test_images_to_return, train_centers_to_return, test_centers_to_return

    def __extract_data(self, data):
        """
        Unpack ShapeNet data.
        """
        images, item_indices, item_angles, centers = [], [], [], []
        for item_index, item in enumerate(data):
            for m, instance in enumerate(item):
                images.append(instance[0])
                item_indices.append(item_index)
                item_angles.append(convert_index_to_angle(instance[2], self.instances_per_item))
                centers.append(instance[3])

        images = np.reshape(np.array(images), (len(images), self.image_height, self.image_width, self.image_channels))
        # change from [0-1] to [0-255]
        images = (images * 255).astype(np.uint8)
        indices, angles = np.array(item_indices), np.array(item_angles)
        centers = np.array(centers)
        return images, indices, angles, centers

    def gen_bg(self, config, data="all"):
        pass
