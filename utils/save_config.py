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
import json


def save_config(dir_path, name, data):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, name), "w") as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(',', ':'))