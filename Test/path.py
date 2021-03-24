#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: path.py
# author: twh
# time: 2021/3/23 17:35
import torch
import os
import glob
import xlrd

name = []
number = []
res = []
full = []
mypath = r"F:\Lab\AMD_CL\origin"
for file in os.listdir(mypath):
    for patch in os.listdir(os.path.join(mypath, file)):
        name += [patch]

for x in name:
    number += [x[6:9]]

print(sorted(number))
workbook = xlrd.open_workbook(r"F:\Lab\AMD_CL\0305AMD应答标注.xls")
sheet = workbook.sheet_by_index(0)
cols = sheet.col_values(0)[2:]
for x in number:
    for y in cols:
        if y not in number:
            print(int(y))

print(res)
