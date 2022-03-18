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
import torch
import imgaug
import argparse

from evaluator.model_evaluator import ModelEvaluator
from trainer.losses import LossFunc
from dataset.refinement import ShapeNet1DRefinement, ShapeNetDistractor
from configs.config import Config

"""
    Refinement used to refine SingleTask models
"""

def refine(config):
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

    optimizer_name = config.optimizer
    if config.weight_decay:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config.lr, weight_decay=config.beta)
    else:
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config.lr)

    # load dataset
    if config.task == 'shapenet_1d':
        data = ShapeNet1DRefinement(path='./data/ShapeNet1D',
                                    img_size=config.img_size,
                                    seed=42,
                                    data_size=config.data_size,
                                    aug=config.aug_list)
    elif config.task == 'distractor':
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  num_instances_per_item=36,
                                  seed=42,
                                  aug=config.aug_list)
    else:
        raise NameError("Choose wrong dataset for refinement, check dataset name!")

    loss = LossFunc(loss_type=config.loss_type, task=config.task)

    if 'MAML' not in config.method:
        trainer = ModelEvaluator(model=model, loss=loss, config=config, data=data, optimizer=optimizer)
    else:
        raise NameError(f"method name:{config.method} is not valid for refinement!")

    trainer.refine()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    for ctx_num in range(1, config.max_ctx_num + 1):
        config.max_ctx_num = ctx_num
        config.save_path = f"results/{config.mode}/{config.method}/{config.timestamp}_{config.task}_datasize_{config.data_size}_{config.agg_mode}_{config.img_agg}{config.loss_type}_{config.aug_list}_seed_{config.seed}/ctx_num{config.max_ctx_num}"
        config.create_dirs()
        refine(config)


if __name__ == "__main__":
    main()
