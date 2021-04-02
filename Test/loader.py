import pandas as pd
import os

root = "C:\\Users\\TRT\\Desktop\\testset\\"
files = []

# for file in os.listdir(root):
#     files.append(root+file)
image_folders = list(map(lambda x: root + x, os.listdir(root)))

for file in os.listdir(root):
    files+=[os.path.join(root,file)]
print(image_folders)
print("")
print(files)