# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/bin/bash

set -e
set -x

# virtualenv -p python3 .
# source ./bin/activate

apt-get update
apt-get install libcusolver10

pip install -r smurf/requirements.txt
python -m smurf.smurf_augmentation_test
python -m smurf.smurf_net_test
python -m smurf.smurf_utils_test
python -m smurf.end_to_end_test
