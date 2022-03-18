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

#!/usr/bin/python3

import os
import json
import numpy as np


def generate_label_path(flag):
    # flag: str, ["train", "test", "val"]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.dirname(dir_path)
    data_path = os.path.join(root_path, "data")
    dir_folder = os.path.join(data_path, flag)
    label_path = os.path.join(dir_folder, "labels")
    return label_path


def generate_label_list(path):
    """
    :param path: path to labels folder
    :param flag: str, ["train", "test", "val"]
    :return: lable list
    """
    label_list = []
    for label_file in os.listdir(path):
        with open(os.path.join(path, label_file), "r") as f:
            p = json.load(f)
            for k in p.keys():
                label = float(p[k])
                label_list.append(label)
    return label_list


def cal_label_mu_sigma(flag="train"):
    label_path = generate_label_path(flag)
    # add all bar length to label_list
    label_list = generate_label_list(label_path)
    label = np.asarray(label_list, dtype=np.double)
    mu = label.mean()
    sigma = label.std()
    result = {}
    result["mu"] = mu
    result["sigma"] = sigma
    with open(os.path.join(os.path.dirname(label_path), "label_mu_sigma.txt"), "w") as f:
        json.dump(result, f)
    return


if __name__ == "__main__":
    cal_label_mu_sigma("train")
    cal_label_mu_sigma("test")
    cal_label_mu_sigma("val")
