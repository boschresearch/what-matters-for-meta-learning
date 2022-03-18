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
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
import argparse
import random
import imgaug
from trainer.losses import LossFunc
from dataset import ShapeNet3DData
from configs.config import Config


"""
Evaluate shapenet3d task with plotting the results
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_angle_from_sincos(generated_angles):
    angle_sin = generated_angles[..., 0]
    angle_cos = generated_angles[..., 1]
    a_acos = np.arccos(angle_cos)
    angles = np.where(angle_sin < 0, np.rad2deg(-a_acos) % 360, np.rad2deg(a_acos))
    return angles


def cal_angle_from_quatt(quaternions):
    r = R.from_quat(quaternions)
    eulers = r.as_euler('ZYX', degrees=True)
    return eulers


def plot_image_and_angle(shot_images, gt_y, pr_y, image_height, image_width, output_path):
    shot_num = shot_images.size(0)
    shot_images = shot_images.cpu().numpy()
    pr_y = pr_y.cpu().numpy()
    gt_y = gt_y.cpu().numpy()
    pr_y = cal_angle_from_quatt(pr_y)
    gt_y = cal_angle_from_quatt(gt_y)

    shot_images = shot_images.transpose(0, 2, 3, 1)

    # plot context angles
    for i in range(shot_num):
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.rcParams['savefig.dpi'] = 64
        plt.rcParams.update({'font.size': 35})
        plt.axis('off')
        fig = plt.figure()
        ax = plt.subplot()
        ax.axis('off')

        # im = ax.imshow(shot_images[i].squeeze(), origin='upper', vmin=0, vmax=1.0)
        im = ax.imshow(shot_images[i].squeeze(), origin='upper')
        plt.text(0.01 * image_width, image_height + 4, f"gt: {gt_y[i][2].round(0), gt_y[i][0].round(0)}", color='green')
        plt.text(0.01 * image_width, image_height + 10, f"pr: {pr_y[i][2].round(0), pr_y[i][0].round(0)}", color='blue')
        patch = patches.Rectangle((0, 0), 128, 128, transform=ax.transData)
        im.set_clip_path(patch)
        plt.savefig(f"{output_path}/{i}", bbox_inches='tight')
        plt.close()


def evaluate(device, config):
    loss_func = LossFunc(loss_type=config.loss_type, task=config.task)
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
    test_iteration = 0

    loss_all = []
    latent_z_list = []
    with torch.no_grad():
        # data.gen_bg(config)
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
            plot_image_and_angle(images_to_generate[0], centers_to_generate[0], pr_mu[0], data.get_image_height(), data.get_image_width(), output_path)


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


    # for i in range(config.max_ctx_num, config.max_ctx_num + 1):  # use loop for testing different number of contexts
    i = 15

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    imgaug.seed(config.seed)
    config.max_ctx_num = i
    config.save_path = path + f'/context_num_{i}'
    latent_z_list = evaluate(device, config)

