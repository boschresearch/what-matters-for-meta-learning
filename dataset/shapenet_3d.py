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

#   This source code is derived from versa (https://github.com/Gordonjo/versa)
#   Copyright (c) 2018 John Bronskill, Jonathan Gordon, and Matthias Bauer, licensed under the MIT license,
#   cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import numpy as np
import torch
import pickle
import sys
import os
from PIL import Image
from utils import Augmenter, task_augment, convert_channel_last_np_to_tensor
from dataset import BaseData

"""
   Supporting methods for data handling in ShapeNet3D
"""


class ShapeNet3DData(BaseData):
    """
        Class to handle ShapeNet dataset. Loads from numpy data as saved in data folder.

        all_categories = ['02958343', '02691156', '04530566', '02828884', '03211117', '02808440', '03046257', '02871439',
                          '03642806', '04468005', '03790512', '03948459', '04330267', '03797390', '04004475', '03513137',
                          '03761084', '04225987', '02942699', '03938244', '03710193', '04099429', '02773838',
                          '02843684', '03261776', '03085013', '02954340', '03928116', '02924116', '02818832']
        categories_test = ['03928116', '02924116', '02818832']  # ['piano', 'bus', 'bed']

    """
    def __init__(self, path, img_size, num_instances_per_item, train_fraction, val_fraction, seed, aug=[], categ=[], mode='train'):
        super().__init__(img_size)

        # self.angle_dimensionality = 3
        self.mode = mode
        # self.bg_imgs = np.load('./data/bg_imgs.npy').astype(np.str)
        self.bg_imgs = np.load('./data/bg_images.npy')
        assert set(aug).issubset(set(["MR", "data_aug", "task_aug", "azimuth_only"]))
        self.aug_list = aug
        if "data_aug" in self.aug_list:
            self.Augmentor = Augmenter()
            self.data_aug = True
        if "task_aug" in self.aug_list:
            self.task_aug = True
        if "azimuth_only" in self.aug_list:
            self.azimuth_only = True
        else:
            self.azimuth_only = False

        with open(os.path.join(path, 'shapenet3d_azi180ele30_train.pkl'), 'rb') as f:
            data_train = pickle.load(f)
            self.train_images = data_train['images']
            self.train_item_indices = data_train['item_indices']
            self.train_Q = data_train['Q']
        with open(os.path.join(path, 'shapenet3d_azi180ele30_val.pkl'), 'rb') as f:
            data_val = pickle.load(f)
            self.validation_images = data_val['images']
            self.validation_item_indices = data_val['item_indices']
            self.validation_Q = data_val['Q']
        with open(os.path.join(path, 'shapenet3d_azi180ele30_test.pkl'), 'rb') as f:
            data_test = pickle.load(f)
            self.test_images = data_test['images']
            self.test_item_indices = data_test['item_indices']
            self.test_Q = data_test['Q']

        self.instances_per_item = num_instances_per_item

        np.random.seed(seed)

        self.train_item_sets = np.max(self.train_item_indices)
        self.validation_item_sets = np.max(self.validation_item_indices)
        self.test_item_sets = np.max(self.test_item_indices)
        self.train_size = self.train_item_sets + 1
        self.val_size = self.validation_item_sets + 1
        self.test_size = self.test_item_sets + 1
        print("Training Set Size = {0:d}".format(self.train_size))
        print("Validation Set Size = {0:d}".format(self.val_size))
        print("Test Set Size = {0:d}".format(self.test_size))
        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        self.test_counter = 0
        self.test_item_permutation = self.test_rng.permutation(self.test_item_sets+1)
        self.val_item_permutation = self.val_rng.permutation(self.validation_item_sets+1)

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
            train_images, test_images, train_Q, test_Q = \
                self.__yield_random_task_batch(tasks_per_batch, self.train_images, self.train_item_indices, self.train_Q, shot)
            train_images = train_images[..., :3]
            test_images = test_images[..., :3]
            if self.data_aug:
                train_images = self.Augmentor.generate(train_images)
                test_images = self.Augmentor.generate(test_images)
                # plot image for debug
                # tmp_img = Image.fromarray((test_images[0, 0, :, :, :3] * 255).astype(np.uint8))
                # tmp_img.show()
            if self.task_aug and shot:
                train_Q, test_Q = task_augment(train_Q, test_Q, azimuth_only=self.azimuth_only)

        elif source == 'validation':
            train_images, test_images, train_Q, test_Q = \
                self.__yield_random_task_batch(tasks_per_batch, self.validation_images, self.validation_item_indices, self.validation_Q, shot)
            train_images = train_images[..., :3]
            test_images = test_images[..., :3]

        elif source == 'test':
            train_images, test_images, train_Q, test_Q = \
                self.__yield_random_task_batch(tasks_per_batch, self.test_images, self.test_item_indices, self.test_Q, shot)
            train_images = train_images[..., :3]
            test_images = test_images[..., :3]

        train_images = convert_channel_last_np_to_tensor(train_images)
        test_images = convert_channel_last_np_to_tensor(test_images)
        train_Q = torch.from_numpy(train_Q).type(torch.FloatTensor)
        test_Q = torch.from_numpy(test_Q).type(torch.FloatTensor)

        return train_images, test_images, train_Q, test_Q

    def __yield_random_task_batch(self, num_tasks_per_batch, images, item_indices, Q, num_train_instances):
        """
        Generate a batch of tasks from image set.
        :param num_tasks_per_batch: number of tasks per batch.
        :param images: images set to generate batch from.
        :param item_indices: indices of each character.
        :param R: object rotation
        :param Q: object quaternion
        :param R: object translation
        :param num_train_instances: number of training images per class.
        :return: a batch of tasks.
        """
        train_images_to_return, test_images_to_return = [], []
        train_Q_to_return, test_Q_to_return = [], []
        for task in range(num_tasks_per_batch):
            train_images, test_images, train_Q, test_Q =\
                self.__generateRandomTask(images, item_indices, Q, num_train_instances)
            train_images_to_return.append(train_images)
            test_images_to_return.append(test_images)
            train_Q_to_return.append(train_Q)
            test_Q_to_return.append(test_Q)
        return np.array(train_images_to_return), np.array(test_images_to_return), np.array(train_Q_to_return), np.array(test_Q_to_return)

    def __generateRandomTask(self, images, item_indices, Q, num_train_instances):
        """
        Randomly generate a task from image set.
        return: tuple containing train and test images and labels for a task.
        """
        if self.source == 'test':
            if self.test_counter > self.test_item_sets:
                # reset test_counter if exceed the test data size and repeat
                self.test_counter = 0
            task_item = self.test_item_permutation[self.test_counter]
            self.test_counter = self.test_counter + 1
            permutation = self.test_rng.permutation(self.instances_per_item)
            # sanity check
            # print(f"item:{task_item}, index: {permutation[:8]}")
        elif self.source == 'validation':
            if self.test_counter > self.validation_item_sets:
                # reset test_counter if exceed the test data size and repeat
                self.test_counter = 0
            task_item = self.val_item_permutation[self.test_counter]
            self.test_counter = self.test_counter + 1
            permutation = self.val_rng.permutation(self.instances_per_item)
            # sanity check
            # print(f"item:{task_item}, index: {permutation[:8]}")
        else:
            task_item = np.random.choice(np.unique(item_indices))
            permutation = np.random.permutation(self.instances_per_item)

        # # add random behavior for dataset, i.e. the starting image is not always with 0 degree
        # permutation_angles = permutation + np.random.randint(self.instances_per_item)
        # permutation_angles[np.where(permutation_angles > self.instances_per_item -1)] -= self.instances_per_item

        item_images     = images[np.where(item_indices == task_item)[0]][permutation]
        item_Q          = Q[np.where(item_indices == task_item)[0]][permutation]

        train_images, train_Q = item_images[:num_train_instances], item_Q[:num_train_instances]
        if self.mode == 'eval':
            test_images, test_Q = item_images, item_Q
        else:
            test_images, test_Q = item_images[num_train_instances:], item_Q[num_train_instances:]

        train_images_to_return, train_Q_to_return = train_images, train_Q
        test_images_to_return, test_Q_to_return = test_images, test_Q

        return train_images_to_return, test_images_to_return, train_Q_to_return, test_Q_to_return

    def __extract_data(self, data):
        """
        Unpack ShapeNet data.
        """
        images, item_indices, Q = [], [], []
        for item_index, item in enumerate(data):
            for m, instance in enumerate(item):
                images.append(instance[0])
                item_indices.append(item_index)
                Q.append(instance[3])

        images = np.reshape(np.array(images).astype(np.float32)/255, (len(images), self.image_height, self.image_width, self.image_channels))
        indices = np.array(item_indices)
        Q = np.array(Q)
        # change Q to semi-sphere and get unique q representation
        index = Q[:, 1] < 0
        Q[index, :] *= -1

        return images, indices, Q

    def add_random_bg(self, images, item_indicies, task_item):

        item_images = images[np.where(item_indicies == task_item)[0]]
        # change background images
        bg_img_list = np.random.choice(self.bg_imgs.shape[0], item_images.shape[0])
        bg_img = self.bg_imgs[bg_img_list]

        mask = (item_images[..., 3] < 1.0)[..., None]
        item_images[..., :3] = item_images[..., :3] * mask + bg_img * (1 - mask)
        images[np.where(item_indicies == task_item)[0]] = item_images
        # # plot image for debug
        # tmp_img = Image.fromarray((item_images[0, :, :, :3] * 255).astype(np.uint8))
        # tmp_img.show()
        # tmp_img = Image.fromarray((item_images[1, :, :, :3] * 255).astype(np.uint8))
        # tmp_img.show()

    def change_background(self, data_size, images, item_indicies):
        # pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        # images = pool.map(partial(self.add_random_bg, images=images, item_indicies=item_indicies), list(range(data_size)))

        for i in range(data_size):
            self.add_random_bg(images=images, item_indicies=item_indicies, task_item=i)
            print(f"Finishing {i}/{data_size} tasks")
        return images

    def gen_bg(self, config, data="all"):

        if data == "all":
            config.logger.info('\n=========== Generate BG for all data ============')
            self.change_background(self.train_size, self.train_images, self.train_item_indices)
            self.change_background(self.val_size, self.validation_images, self.validation_item_indices)
            self.change_background(self.test_size, self.test_images, self.test_item_indices)
        elif data == "train":
            config.logger.info('\n============= Regenerate BG for Training Data ==============')
            self.change_background(self.train_size, self.train_images, self.train_item_indices)
        else:
            raise TypeError("Wrong data type for generating random background, check gen_bg(data=**)!")

    def generate_and_save_data(self,):
        # print("Number of cpu : ", multiprocessing.cpu_count())
        # print('\n=========== Generate BG for all data ============')

        train_image = self.change_background(self.train_size, self.train_images, self.train_item_indices)
        with open('./data/ShapeNet3D_azi180ele30/shapenet3d_azi180ele30_train.pkl', 'wb') as f:
            pickle.dump({"images": train_image, "item_indices": self.train_item_indices, "Q": self.train_Q}, f)
        val_image = self.change_background(self.val_size, self.validation_images, self.validation_item_indices)
        with open('./data/ShapeNet3D_azi180ele30/shapenet3d_azi180ele30_val.pkl', 'wb') as f:
            pickle.dump({"images": val_image, "item_indices": self.validation_item_indices, "Q": self.validation_Q}, f)
        test_image = self.change_background(self.test_size, self.test_images, self.test_item_indices)
        with open('./data/ShapeNet3D_azi180ele30/shapenet3d_azi180ele30_test.pkl', 'wb') as f:
            pickle.dump({"images": test_image, "item_indices": self.test_item_indices, "Q": self.test_Q}, f)

        print('\n=========== Finished ============')


if __name__ == "__main__":

    data = ShapeNet3DData(path='./data/ShapeNet3D_azi180ele30_w_bg',
                          img_size=[64, 64, 4],
                          train_fraction=0.8,
                          val_fraction=0.2,
                          num_instances_per_item=30,
                          seed=42,)

    data.generate_and_save_data()

    # with open('./segmentation_shapenet3d_azi360ele30_test.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # images = data["images"]
    # task_num = images.shape[0]
    # for i in range(task_num):
    #     item_images = images[i]
    #     # plot image for debug
    #     tmp_img = Image.fromarray((item_images[:, :, :3] * 255).astype(np.uint8))
    #     tmp_img.show()
