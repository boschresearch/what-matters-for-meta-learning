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
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluator.base_evaluator import BaseEvaluator


class ModelEvaluator(BaseEvaluator):
    """ Evaluator """

    def __init__(self, model, loss, config, data, optimizer=None):
        super().__init__(model=model, loss=loss, config=config, optimizer=optimizer)
        self.loss = loss
        self.data = data

    def refine(self):

        self.config.logger.info('\n================== Start training ===================')

        assert (self.iterations + 1) > self.start_iter
        for iter in range(self.start_iter, self.iterations + 1):

            if iter % self.config.bg_gen_freq == 0:
                self.data.gen_bg(self.config, data="train")

            self.config.logger.info("Iter {}".format(iter))
            self._refine_iter(iter)

            if iter % self.config.val_freq == 0:
                self._validate_iter(source='validation')
                if self.config.task != 'pascal_1d':
                    source = 'test'
                    loss, std = self._validate_iter(source='test')
                    if loss < self.best_loss[source]:
                        self.best_loss[source] = loss
                        best_model_step = iter
                        self.config.logger.info(f"save best {source} model epoch : {best_model_step}\n")
                        torch.save(self.model.state_dict(), f"{self.config.save_path}/models/best_{source}_model.pt")
                        with open(os.path.join(self.config.save_path, f"best_{source}_error.txt"), "a") as f:
                            f.write(f"Best Step: {iter} \n")
                            f.write(f"Best {source} Loss: \n{str(loss)}\n")
                            f.write(f"Best {source} Loss std: \n{str(std)}\n")

        torch.save(self.model.state_dict(), f"{self.config.save_path}/models/model_end_{iter}.pt")
        self.config.logger.info('models have been saved to {}'.format(self.config.save_path))
        self.config.logger.info('================= Training finished =================\n')

    def _refine_iter(self, iter):

        self.model.train()
        self.optimizer.zero_grad()

        ctx_x, qry_x, ctx_y, qry_y = \
            self.data.get_batch(source='refine_train', tasks_per_batch=self.config.tasks_per_batch, shot=self.config.max_ctx_num)

        ctx_x = ctx_x.to(self.config.device)
        qry_x = qry_x.to(self.config.device)
        ctx_y = ctx_y.to(self.config.device)
        qry_y = qry_y.to(self.config.device)

        pr_mu, pr_var, kl = self.model(None, None, ctx_x)  # refine use train data
        losses = self.loss.calc_loss(pr_mu, pr_var, ctx_y)
        losses += kl * self.config.beta

        losses.backward()
        self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar('Loss/train', losses, iter)
            self.config.logger.info(f"Train Iteration {iter} loss: {losses.item():.4f}\n")

        if not math.isfinite(losses.item()):
            self.config.logger.info("Loss is {}, stopping training".format(losses.item()))
            sys.exit(1)
        del ctx_x, qry_x, ctx_y, qry_y

    def evaluate(self):

        self.config.logger.info('\n================== Start Evaluation ===================')
        # self.start_time = time.time()
        val_losses, val_std = [], []
        test_losses, test_std = [], []

        for ctx_num in range(1, self.config.max_ctx_num + 1):
            loss, std = self._validate_iter(source='validation', max_ctx_num=ctx_num)
            val_losses.append(loss)
            val_std.append(std)
            if self.config.task != 'pascal_1d':
                loss, std = self._validate_iter(source='test', max_ctx_num=ctx_num)
                test_losses.append(loss)
                test_std.append(std)

        index = list(range(1, self.config.max_ctx_num + 1))
        val_result = np.column_stack((index, val_losses, val_std))
        np.savetxt(f"{self.config.save_path}/val_losses.txt", val_result, fmt='%1.4f')
        if self.config.task != 'pascal_1d':
            test_result = np.column_stack((index, test_losses, test_std))
            np.savetxt(f"{self.config.save_path}/test_losses.txt", test_result, fmt='%1.4f')

        torch.save(self.model.state_dict(), f"{self.config.save_path}/models/model.pt")
        self.config.logger.info('models have been saved to {}'.format(self.config.save_path))
        self.config.logger.info('================= Evaluation finished =================\n')

        self._plot_loss_vs_ctx_num(val_losses, val_std, test_losses, test_std)

    def evaluate_one_task(self):
        self.config.logger.info('\n================== Start Evaluation ===================')
        test_losses, test_std = [], []

        for ctx_num in range(1, self.config.max_ctx_num + 1):
            loss, std = self._validate_iter(source='test', max_ctx_num=ctx_num)
            test_losses.append(loss)
            test_std.append(std)

        index = list(range(1, self.config.max_ctx_num + 1))

        test_result = np.column_stack((index, test_losses, test_std))
        np.savetxt(f"{self.config.save_path}/test_losses.txt", test_result, fmt='%1.4f')

        torch.save(self.model.state_dict(), f"{self.config.save_path}/models/model.pt")
        self.config.logger.info('Results have been saved to {}'.format(self.config.save_path))
        self.config.logger.info('================= Evaluation finished =================\n')

        self._plot_loss_vs_ctx_num_one_task(test_losses, test_std)

    def _validate_iter(self, source, max_ctx_num=0):
        """
        Validate model on validation data (intra-category) and test data (novel category)
        """
        self.model.eval()
        with torch.no_grad():
            losses = []
            if source == 'test':
                # reset test_counter and reset seed
                self.data.test_counter = 0
                self.data.test_rng.seed(42)
            elif source == 'validation':
                # reset test_counter and reset seed
                self.data.test_counter = 0
                self.data.val_rng.seed(42)
            for i in range(self.config.val_iters):
                ctx_x, qry_x, ctx_y, qry_y = \
                    self.data.get_batch(source=source, tasks_per_batch=self.config.tasks_per_batch, shot=max_ctx_num)
                ctx_x = ctx_x.to(self.config.device)
                qry_x = qry_x.to(self.config.device)
                ctx_y = ctx_y.to(self.config.device)
                qry_y = qry_y.to(self.config.device)
                if self.config.contrastive:
                    pr_mu, pr_var, _, _ = self.model(ctx_x, ctx_y, qry_x, qry_y, test=True)
                else:
                    pr_mu, pr_var, _ = self.model(ctx_x, ctx_y, qry_x, test=True)
                loss = self.loss.calc_loss(pr_mu, pr_var, qry_y, test=True)
                losses.append(loss.view(1))

            loss = torch.mean(torch.cat(losses))
            std    = torch.std(torch.cat(losses))
            self.config.logger.info(f"{source} loss: {loss.item():.4f}")
            self.config.logger.info(f"{source} std: {std.item():.4f}")

            del ctx_x, qry_x, ctx_y, qry_y
            return loss.item(), std.item()

    def _save_checkpoint(self, epoch):
        d = {
             'epoch': epoch,
             'model': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict()
        }
        filename = os.path.abspath(os.path.join(self.save_dir, 'checkpoint-epoch{}.pth'.format(epoch)))
        torch.save(d, filename)

    def _resume_checkpoint(self, resume_path, optimizer=None):
        """
        Resume from saved checkpoints
        """
        ckpt = torch.load(resume_path)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.start_epoch = ckpt['epoch']
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])

    def _plot_loss_vs_ctx_num(self, val_losses, val_std, test_losses, test_std):
        filename = f"{self.config.save_path}/loss_vs_ctx_num.png"
        index = list(range(1, self.config.max_ctx_num + 1))
        val_losses = np.array(val_losses)
        val_std = np.array(val_std)
        test_losses = np.array(test_losses)
        test_std = np.array(test_std)

        # plt.ylim(0, 8)

        plt.plot(index, val_losses, label='val')
        plt.fill_between(index, val_losses - val_std, val_losses + val_std, alpha=0.1)
        if self.config.task != 'pascal_1d':
            plt.plot(index, test_losses, label='test')
            plt.fill_between(index, test_losses - test_std, test_losses + test_std, alpha=0.1)
            # single_loss = np.loadtxt("results/refinement/SingleTaskShapeNet1D/2021-10-01_12-50-09_shapenet_1d_datasize_large__mse_['data_aug']_seed_2578/loss_vs_ctx.txt")
            # plt.plot(index, single_loss, label='refinement')
            # single_loss = np.loadtxt("results/refinement/SingleTaskDistractor/2021-10-03_22-47-35_distractor_datasize_None__maxmse_['data_aug']_seed_2578/loss_vs_ctx.txt")
            # plt.plot(index, single_loss, label='refinement')

        plt.legend(loc='best')
        plt.xlabel('ctx_num')
        plt.ylabel('error(pixel)')
        plt.savefig(filename)
        plt.clf()

    def _plot_loss_vs_ctx_num_one_task(self, test_losses, test_std):
        filename = f"{self.config.save_path}/loss_vs_ctx_num.png"
        index = list(range(1, self.config.max_ctx_num + 1))
        test_losses = np.array(test_losses)
        test_std = np.array(test_std)

        # plt.ylim(0, 8)

        plt.plot(index, test_losses, label='test')
        plt.fill_between(index, test_losses - test_std, test_losses + test_std, alpha=0.1)
        # single_loss = np.loadtxt("results/refinement/SingleTaskShapeNet1D/2021-10-01_12-50-09_shapenet_1d_datasize_large__mse_['data_aug']_seed_2578/loss_vs_ctx.txt")
        # plt.plot(index, single_loss, label='refinement')
        # single_loss = np.loadtxt("results/refinement/SingleTaskDistractor/2021-10-03_22-47-35_distractor_datasize_None__maxmse_['data_aug']_seed_2578/loss_vs_ctx.txt")
        # plt.plot(index, single_loss, label='refinement')

        plt.legend(loc='best')
        plt.xlabel('ctx_num')
        plt.ylabel('error(pixel)')
        plt.savefig(filename)
        plt.clf()
