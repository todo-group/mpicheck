#!/usr/bin/env python3

import os
import torch

# print(os.environ['CUDA_VISIBLE_DEVICES'])
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())

