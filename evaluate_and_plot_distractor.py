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
import math
from pathlib import Path
import torch
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, img_as_ubyte
from transforms3d.euler import quat2euler
import argparse
import random
import imgaug
from trainer.losses import LossFunc
from dataset import ShapeNet3DData, ShapeNetDistractor, Pascal1D, ShapeNet1D
from configs.config import Config

"""
Evaluate distractor task with plotting the results
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(__file__)
root_path = file_path.parent
method = "CNPDistractor"

# categories = ['04256520', '04530566']  # ['sofa', 'watercraft']
categories = ['04530566']

labels = {k: int(v) for (v, k) in enumerate(categories)}


def plot_image_and_center(shot_images, gt_centers, generated_mu, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    shot_images = shot_images.cpu().numpy()
    generated_mu = generated_mu.cpu().numpy()
    gt_centers = gt_centers.cpu().numpy()

    # plot context angles
    for i in range(shot_num):
        gen_center = (generated_mu[i, 0], generated_mu[i, 1])
        gt_center = (gt_centers[i, 0], gt_centers[i, 1])

        fig = plt.figure(figsize=(2, 2))
        ax = plt.subplot()
        ax.axis('off')

        im = ax.imshow(shot_images[i].squeeze(), origin='upper', cmap='gray')
        ax.plot(*gen_center, 'bo', markersize=7)
        ax.plot(*gt_center, marker='o', markersize=7, color='green')
        patch = patches.Rectangle((0, 0), 128, 128, transform=ax.transData)
        im.set_clip_path(patch)
        fig.tight_layout()
        plt.savefig(f"{output_path}/{i}")
        plt.close()


def evaluate(device, config):
    loss_func = LossFunc(loss_type=config.loss_type, task=config.task)
    # load dataset
    if config.task == 'distractor':
        # used to plot only test category
        data = ShapeNetDistractor(path='./data/distractor',
                                  img_size=config.img_size,
                                  train_fraction=0.8,
                                  val_fraction=0.2,
                                  num_instances_per_item=36,
                                  seed=42,
                                  aug=config.aug_list,
                                  mode='eval',
                                  load_test_categ_only=True,
                                  test_categ=categories)
    else:
        raise NameError("dataset doesn't exist, check dataset name!")

    import importlib
    module = importlib.import_module(f"networks.{config.method}")
    np_class = getattr(module, config.method)
    model = np_class(config)
    model = model.to(config.device)

    checkpoint = config.checkpoint
    if checkpoint:
        config.logger.info("load weights from " + checkpoint)
        model.load_state_dict(torch.load(checkpoint))
    # model.eval()
    test_iteration = 0

    loss_all = []
    latent_z_list = []
    with torch.no_grad():
        data.gen_bg(config)
        while test_iteration < config.val_iters:
            source = 'test'
            ctx_x, qry_x, ctx_y, qry_y = \
                data.get_batch(source=source, tasks_per_batch=config.tasks_per_batch, shot=config.max_ctx_num)
            ctx_x = ctx_x.to(config.device)
            qry_x = qry_x.to(config.device)
            ctx_y = ctx_y.to(config.device)
            qry_y = qry_y.to(config.device)

            pr_mu, pr_var, sample_z = model(ctx_x, ctx_y, qry_x)
            latent_z_list.append(sample_z)
            loss = loss_func.calc_loss(pr_mu, pr_var, qry_y, test=True)
            loss_all.append(loss.item())

            images_to_generate = qry_x
            centers_to_generate = qry_y
            path_save_image = os.path.join(config.save_path, "image")
            if not os.path.exists(path_save_image):
                os.makedirs(path_save_image)
            output_path = os.path.join(path_save_image, 'output_{0:02d}'.format(test_iteration))
            os.makedirs(output_path)
            plot_image_and_center(1.0 - images_to_generate[0], centers_to_generate[0], pr_mu[0], data.get_image_height(), data.get_image_width(), output_path)

            test_iteration += 1

    with open(os.path.join(config.save_path, 'losses_all.txt'), 'w') as f:
        np.savetxt(f, loss_all, delimiter=',', fmt='%.4f')
    config.logger.info('Results have been saved to {}'.format(config.save_path))
    config.logger.info('================= Evaluation finished =================\n')
    return latent_z_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file")
    args = parser.parse_args()
    config = Config(args.config)
    path = config.save_path
    i = 15
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    imgaug.seed(config.seed)
    config.max_ctx_num = i
    config.save_path = path + f'/context_num_{i}'
    latent_z_list = evaluate(device, config)

