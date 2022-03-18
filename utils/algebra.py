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

import math
import numpy as np



def calc_function(x, y, angle):
    """given point (x, y) and angle, calculate linear function y = kx + b, return b"""

    k = math.tan(angle)
    b = y - k * x
    return b


def calc_mean_var(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    return mean, std


if __name__ == "__main__":
    x = [8.0853, 7.8592, 8.0334, 8.0864, 8.1560]
    mean, std = calc_mean_var(x)
    print("mean: ", mean)
    print("std: ", std)

