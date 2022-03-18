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

#   This source code is derived from Model-Agnostic Meta-Learning (https://github.com/tristandeleu/pytorch-maml)
#   Copyright (c) 2019 Tristan Deleu, licensed under the MIT license,
#   cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

from collections import OrderedDict
from torchmeta.utils import gradient_update_parameters
from utils import tensors_to_device, compute_accuracy
from trainer.base_trainer import BaseTrainer


class ModelAgnosticMetaLearning(BaseTrainer):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    """
    def __init__(self, model, config, data, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, test_num_adaptation_steps=1, scheduler=None,
                 loss_function=F.mse_loss, device=None):
        super().__init__(model=model, loss=loss_function, optimizer=optimizer, config=config)

        self.model = model.to(device=device)
        self.config = config
        self.data = data
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.test_num_adaptation_steps = test_num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_outer_loss(self, ctx_x, ctx_y, qry_x, qry_y, num_adaptation_steps=5, test=False):

        test_targets = qry_y
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'mean_pre_loss': 0.
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        mean_pre_loss = torch.tensor(0., device=self.device)  # prediction loss without kl, used for validation
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(ctx_x, ctx_y, qry_x, qry_y)):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                num_adaptation_steps=num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']

            with torch.set_grad_enabled(self.model.training):
                test_logits, kl = self.model(test_inputs, params=params)
                outer_loss = self.loss_function.calc_loss(test_logits, None, test_targets, test=test)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_pre_loss += outer_loss
                mean_outer_loss += outer_loss + self.config.beta * kl

        mean_outer_loss.div_(num_tasks)
        mean_pre_loss.div_(num_tasks)

        results['mean_outer_loss'] = mean_outer_loss.item()
        results['mean_pre_loss'] = mean_pre_loss

        return mean_outer_loss, results

    def adapt(self, inputs, targets, num_adaptation_steps=1, step_size=0.1, first_order=False):
        params = None

        results = {'inner_losses': np.zeros(
            (num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits, _ = self.model(inputs, params=params)
            inner_loss = self.loss_function.calc_loss(logits, None, targets)
            results['inner_losses'][step] = inner_loss.item()

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                                                step_size=step_size, params=params,
                                                # first_order=(not self.model.training) or first_order,  #TODO: first_order during evaluation??
                                                first_order=first_order
                                                )

        return params, results

    def train(self):
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

        outer_loss, results = self.get_outer_loss(ctx_x, ctx_y, qry_x, qry_y, num_adaptation_steps=self.num_adaptation_steps)

        outer_loss.backward()
        self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar('Loss/train', outer_loss, iter)
            self.config.logger.info(f"Train Iteration {iter} loss: {outer_loss.item():.4f}\n")

        if not math.isfinite(outer_loss.item()):
            self.config.logger.info("Loss is {}, stopping training".format(outer_loss.item()))
            sys.exit(1)
        del ctx_x, qry_x, ctx_y, qry_y
        return results

    def _validate_iter(self, iter, source):
        """
        Validate model on validation data and save visual results for checking
        :return: a dict of model's output
        """
        self.model.eval()

        losses = []
        for i in range(self.config.val_iters):
            ctx_x, qry_x, ctx_y, qry_y = \
                self.data.get_batch(source=source, tasks_per_batch=self.config.tasks_per_batch,
                                    shot=self.config.max_ctx_num)
            ctx_x = ctx_x.to(self.config.device)
            qry_x = qry_x.to(self.config.device)
            ctx_y = ctx_y.to(self.config.device)
            qry_y = qry_y.to(self.config.device)

            outer_loss, results = self.get_outer_loss(ctx_x, ctx_y, qry_x, qry_y,
                                                      num_adaptation_steps=self.test_num_adaptation_steps, test=True)

            losses.append(results['mean_pre_loss'])

        loss = torch.mean(torch.stack(losses))
        std  = torch.std(torch.stack(losses))
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


MAMLTrainer = ModelAgnosticMetaLearning

