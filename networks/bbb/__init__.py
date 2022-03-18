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

#   This source code is derived from PyTorch-BayesianCNN (https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
#   Copyright (c) 2019 Kumar Shridhar, licensed under the MIT license,
#   cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from .BBBConv import BBBConv2d
from .BBBLinear import BBBLinear
from .misc import ModuleWrapper, FlattenLayer