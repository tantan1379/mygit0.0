import os
import glob
import shutil

mypath = "F:\\Lab\\2021文献整理\\2015-2016实验室文献整理\\眼科相关"
despath = "F:\\Lab\\2021文献整理\\paper(ophthalmology)"
paper = []

paper+=glob.glob(os.path.join(mypath,"*/*/*.pdf"))

print(paper)

for p in paper:
    filepath = os.path.join(despath,os.path.split(p)[-1])
    shutil.copyfile(p,filepath)

