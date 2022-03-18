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

import os.path
import numpy as np
import random
import torch
import imgaug
import argparse

from evaluator.model_evaluator import ModelEvaluator
from trainer.losses import LossFunc
from dataset import ShapeNet3DData, ShapeNetDistractor, Pascal1D, ShapeNet1D
from configs.config import Config


"""
Evaluation on different tasks and return statistical results
"""


def evaluate(config):
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    imgaug.seed(config.seed)

    import importlib
    module = importlib.import_module(f"networks.{config.method}")
    np_class = getattr(module, config.method)
    model = np_class(config)
    model = model.to(config.device)

    checkpoint = config.checkpoint
    if checkpoint:
        config.logger.info("load weights from " + checkpoint)
        model.load_state_dict(torch.load(checkpoint))

    # load dataset
    if config.task == 'shapenet_3d':
        data = ShapeNet3DData(path='./data/ShapeNet3D_azi180ele30',
                              img_size=config.img_size,
                              train_fraction=0.8,
                              val_fraction=0.2,
                              num_instances_per_item=30,
                              seed=42,
                              aug=config.aug_list,
                              mode='eval')
    elif config.task == 'pascal_1d':
        data = Pascal1D(path='./data/Pascal1D',
                        img_size=config.img_size,
                        seed=42,
                        aug=config.aug_list)
    elif config.task == 'shapenet_1d':
        data = ShapeNet1D(path='./data/ShapeNet1D',
                        img_size=config.img_size,
                        seed=42,
                        data_size=config.data_size,
                        aug=config.aug_list)

    elif config.task == 'distractor':
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  train_fraction=0.8,
                                  val_fraction=0.2,
                                  num_instances_per_item=36,
                                  seed=42,
                                  load_test_categ_only=True,
                                  aug=config.aug_list,
                                  mode='eval')
    else:
        raise NameError("dataset doesn't exist, check dataset name!")

    loss = LossFunc(loss_type=config.loss_type, task=config.task)

    if 'MAML' not in config.method:
        evaluator = ModelEvaluator(model=model, loss=loss, config=config, data=data)
    else:
        raise NameError(f"method name:{config.method} is not valid!")

    evaluator.evaluate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
