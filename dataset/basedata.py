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

from abc import abstractmethod


class BaseData:
    def __init__(self, img_size):
        self.img_size = img_size
        self.image_height = img_size[0]
        self.image_width = img_size[1]
        self.image_channels = img_size[2]
        self.data_aug = False
        self.task_aug = False

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    @abstractmethod
    def get_batch(self, source, tasks_per_batch, shot):
        raise NotImplementedError

    @abstractmethod
    def gen_bg(self, config, data="all"):
        raise NotImplementedError

