#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: test.py
# author: twh
# time: 2021/3/11 20:24
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch


# path="F:/Lab/AMD_CL/origin/1/01-A-0001-V1-OCT/王季萍000.jpg"
# img=Image.open(path)
# img_path=path.split('/')
# print(img_path)
# cropped=img.crop((500,0,1008,435))
# plt.imshow(cropped)
# plt.show()

X = torch.rand(2,3,3)
Y = torch.rand(500, 1)
print(X)