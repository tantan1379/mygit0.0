#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: dataset.py
# author: twh
# time: 2021/3/11 21:30
from torch.utils import data
import os


class AMD_CL(data.Dataset):
    def __init__(self, root, resize, mode):
        super(AMD_CL, self).__init__()
        self.root = root
        self.resize = resize
        self.images, self.labels = self.load_csv('images&labels.csv')
        assert (mode == "train" or mode == "val" or mode == "test"), "invalid mode input"
        if mode == "train":
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.images))]
