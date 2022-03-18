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

#   This source code is derived from MMAML-Classification (https://github.com/shaohua0116/MMAML-Classification)
#   Copyright (c) 2018 Tristan Deleu, Risto Vuorio,
#   Copyright (c) 2019 Hexiang Hu, Shao-Hua Sun,
#   licensed under the MIT license,
#   cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import math
import sys
import json
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np
import torch


class MMAMLTrainer(object):
    def __init__(self, meta_learner, data, log_interval,
                 save_interval, model_type, config):

        self.best_loss = {'validation': 10000, 'test': 10000}
        self.config = config
        self.start_iter = 1
        self.iterations = config.iterations
        assert config.save_path is not None
        self.save_path = config.save_path
        # setup visualization writer instance
        self.writer = SummaryWriter(self.save_path, max_queue=10)

        self._meta_learner = meta_learner
        self.data = data
        self._log_interval = log_interval
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = config.save_path

    def run(self, is_training):

        # generate random bg before starting training
        self.data.gen_bg(self.config)

        self.config.logger.info('\n================== Start training ===================')
        # self.start_time = time.time()
        assert (self.iterations + 1) > self.start_iter
        for iter in range(self.start_iter, self.iterations + 1):

            if iter % self.config.bg_gen_freq == 0:
                self.data.gen_bg(self.config, data="train")

            self.config.logger.info("Iter {}".format(iter))
            self._train_iter(iter)

            if iter % self.config.val_freq == 0:
                self._validate_iter(iter, source='validation')
                if self.config.task != 'pascal_1d':
                    self._validate_iter(iter, source='test')

            # save weights for each 1000 epochs
            if iter % 1000 == 0:
                self.save_intermediate_model(iter)

    def _train_iter(self, iter, is_training=True):
        test = False
        ctx_x, qry_x, ctx_y, qry_y = \
            self.data.get_batch(source='train', tasks_per_batch=self.config.tasks_per_batch,
                                shot=self.config.max_ctx_num)
        ctx_x = ctx_x.to(self.config.device)
        qry_x = qry_x.to(self.config.device)
        ctx_y = ctx_y.to(self.config.device)
        qry_y = qry_y.to(self.config.device)

        (pre_train_measurements, adapted_params, embeddings
         ) = self._meta_learner.adapt(ctx_x, ctx_y)
        post_val_measurements = self._meta_learner.step(
            adapted_params, embeddings, qry_x, qry_y, is_training, test=test)

        if self.writer is not None:
            self.writer.add_scalar('Loss/train', post_val_measurements['loss'], iter)
            self.config.logger.info(f"Train Iteration {iter} loss: {post_val_measurements['loss']:.4f}\n")

        del ctx_x, qry_x, ctx_y, qry_y

    def _validate_iter(self, iter, source):
        """
        Validate model on validation data and save visual results for checking
        :return: a dict of model's output
        """
        test = True
        losses = []
        for i in range(self.config.val_iters):

            ctx_x, qry_x, ctx_y, qry_y = \
                self.data.get_batch(source=source, tasks_per_batch=self.config.tasks_per_batch,
                                    shot=self.config.max_ctx_num)

            ctx_x = ctx_x.to(self.config.device)
            qry_x = qry_x.to(self.config.device)
            ctx_y = ctx_y.to(self.config.device)
            qry_y = qry_y.to(self.config.device)

            (pre_train_measurements, adapted_params, embeddings
             ) = self._meta_learner.adapt(ctx_x, ctx_y, num_updates=self.config.test_num_steps)
            post_val_measurements = self._meta_learner.step(
                adapted_params, embeddings, qry_x, qry_y, is_training=False, test=test)

            losses.append(post_val_measurements['loss'])


        loss = np.mean(losses)
        std = np.std(losses)
        self.writer.add_scalar(f'Loss/{source}', loss, iter)
        self.config.logger.info(f"{source} {iter} loss: {loss.item():.4f}")

        if loss < self.best_loss[source]:
            self.best_loss[source] = loss
            best_model_step = iter
            self.config.logger.info(f"save best {source} model epoch : {best_model_step}\n")
            torch.save(self._meta_learner.state_dict(), f"{self.config.save_path}/models/best_{source}_model.pt")
            with open(os.path.join(self.config.save_path, f"best_{source}_error.txt"), "a") as f:
                f.write(f"Best Step: {iter} \n")
                f.write(f"Best {source} Loss: \n{str(loss)}\n")
                f.write(f"Best {source} Loss std: \n{str(std)}\n")
        del ctx_x, qry_x, ctx_y, qry_y

    def compute_confidence_interval(self, value):
        """
        Compute 95% +- confidence intervals over tasks
        change 1.960 to 2.576 for 99% +- confidence intervals
        """
        return np.std(value) * 1.960 / np.sqrt(len(value))

    def train(self):
        self.run(is_training=True)

    def eval(self):
        self.run(is_training=False)

    def write_tensorboard(self, pre_val_measurements, pre_train_measurements,
                          post_val_measurements, post_train_measurements,
                          iteration, embedding_grads_mean=None):
        for key, value in pre_val_measurements.items():
            self._writer.add_scalar(
                '{}/before_update/meta_val'.format(key), value, iteration)
        for key, value in pre_train_measurements.items():
            self._writer.add_scalar(
                '{}/before_update/meta_train'.format(key), value, iteration)
        for key, value in post_train_measurements.items():
            self._writer.add_scalar(
                '{}/after_update/meta_train'.format(key), value, iteration)
        for key, value in post_val_measurements.items():
            self._writer.add_scalar(
                '{}/after_update/meta_val'.format(key), value, iteration)
        if embedding_grads_mean is not None:
            self._writer.add_scalar(
                'embedding_grads_mean', embedding_grads_mean, iteration)

    def log_output(self, pre_val_measurements, pre_train_measurements,
                   post_val_measurements, post_train_measurements,
                   iteration, embedding_grads_mean=None):
        log_str = 'Iteration: {} '.format(iteration)
        for key, value in sorted(pre_val_measurements.items()):
            log_str = (log_str + '{} meta_val before: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(pre_train_measurements.items()):
            log_str = (log_str + '{} meta_train before: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(post_train_measurements.items()):
            log_str = (log_str + '{} meta_train after: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(post_val_measurements.items()):
            log_str = (log_str + '{} meta_val after: {:.3f} '
                                 ''.format(key, value))
        if embedding_grads_mean is not None:
            log_str = (log_str + 'embedding_grad_norm after: {:.3f} '
                       ''.format(embedding_grads_mean))
        print(log_str)

    def save_intermediate_model(self, iter):
        torch.save(self._meta_learner.state_dict(), f"{self.config.save_path}/models/model_intermediate.pt")
        self.config.logger.info(f"save intermediate model iter: {iter}")
