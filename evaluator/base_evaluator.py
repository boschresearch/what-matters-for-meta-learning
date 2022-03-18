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

import torch
from abc import abstractmethod
from torch.utils.tensorboard.writer import SummaryWriter


class BaseEvaluator:
    """
    Base Evaluator
    """
    def __init__(self, model, loss, config, optimizer=None):
        self.best_loss = {'validation': 10000, 'test': 10000}
        self.config = config
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.start_iter = 1
        self.iterations = config.iterations
        assert config.save_path is not None
        self.save_path = config.save_path

        self.writer = SummaryWriter(self.save_path, max_queue=10)

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def _save_checkpoint(self, epoch):
        raise NotImplementedError

    def _resume_checkpoint(self, resume_path):
        raise NotImplementedError
