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
import torch
from trainer.base_trainer import BaseTrainer


class ModelTrainer(BaseTrainer):
    """ Trainer """

    def __init__(self, model, loss, optimizer, config, data):
        super().__init__(model=model, loss=loss, optimizer=optimizer, config=config)
        self.loss = loss
        self.data = data

    def train(self):
        self.config.logger.info('\n================== Start training ===================')
        # self.start_time = time.time()
        assert (self.iterations + 1) > self.start_iter
        for iter in range(self.start_iter, self.iterations + 1):

            if iter % self.config.bg_gen_freq == 0 and self.config.gen_bg:
                self.data.gen_bg(self.config, data="train")
                # pass

            self.config.logger.info("Iter {}".format(iter))
            self._train_iter(iter)

            if iter % self.config.val_freq == 0:
                self._validate_iter(iter, source='validation')
                if self.config.task != 'pascal_1d':
                    self._validate_iter(iter, source='test')

            # save weights for each 1000 epochs
            if iter % 1000 == 0:
                self.save_intermediate_model(iter)

        torch.save(self.model.state_dict(), f"{self.config.save_path}/models/model_end_{iter}.pt")
        self.config.logger.info('models have been saved to {}'.format(self.config.save_path))
        self.config.logger.info('================= Training finished =================\n')

    def _train_iter(self, iter):

        self.model.train()
        self.optimizer.zero_grad()

        ctx_x, qry_x, ctx_y, qry_y = \
            self.data.get_batch(source='train', tasks_per_batch=self.config.tasks_per_batch, shot=self.config.max_ctx_num)

        ctx_x = ctx_x.to(self.config.device)
        qry_x = qry_x.to(self.config.device)
        ctx_y = ctx_y.to(self.config.device)
        qry_y = qry_y.to(self.config.device)

        if self.config.contrastive:
            pr_mu, pr_var, kl, contra_loss = self.model(ctx_x, ctx_y, qry_x, qry_y)
        else:
            pr_mu, pr_var, kl = self.model(ctx_x, ctx_y, qry_x)

        losses = self.loss.calc_loss(pr_mu, pr_var, qry_y)
        losses += kl * self.config.beta

        if self.config.contrastive:
            losses += contra_loss * self.config.contrastive_rate

        losses.backward()
        self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar('Loss/train', losses, iter)
            self.config.logger.info(f"Train Iteration {iter} loss: {losses.item():.4f}\n")

        if not math.isfinite(losses.item()):
            self.config.logger.info("Loss is {}, stopping training".format(losses.item()))
            sys.exit(1)
        del ctx_x, qry_x, ctx_y, qry_y

    def _validate_iter(self, iter, source):
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
                    self.data.get_batch(source=source, tasks_per_batch=self.config.tasks_per_batch,
                                        shot=self.config.max_ctx_num)
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
            self.writer.add_scalar(f'Loss/{source}', loss, iter)
            self.config.logger.info(f"{source} {iter} loss: {loss.item():.4f}")

            if loss < self.best_loss[source]:
                self.best_loss[source] = loss
                best_model_step = iter
                self.config.logger.info(f"save best {source} model epoch : {best_model_step}\n")
                torch.save(self.model.state_dict(), f"{self.config.save_path}/models/best_{source}_model.pt")
                with open(os.path.join(self.config.save_path, f"best_{source}_error.txt"), "a") as f:
                    f.write(f"Best Step: {iter} \n")
                    f.write(f"Best {source} Loss: \n{str(loss)}\n")
                    f.write(f"Best {source} Loss std: \n{str(std)}\n")
            del ctx_x, qry_x, ctx_y, qry_y

    def save_intermediate_model(self, iter):
        torch.save(self.model.state_dict(), f"{self.config.save_path}/models/model_intermediate.pt")
        self.config.logger.info(f"save intermediate model iter: {iter}")

    def _save_checkpoint(self, epoch):
        d = {
             'epoch': epoch,
             'model': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict()
        }
        filename = os.path.abspath(os.path.join(self.save_dir,
                                                'checkpoint-epoch{}.pth'.format(epoch)))
        torch.save(d, filename)

    def _resume_checkpoint(self, resume_path, optimizer=None):
        ckpt = torch.load(resume_path)
        self.model.load_state_dict(ckpt['model'], strict=True)
        self.start_epoch = ckpt['epoch']
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
