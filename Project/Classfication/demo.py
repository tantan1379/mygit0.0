import glob
import os
import pandas as pd
from dataset.dataloader import *
# all_images = []
# image_folders = list(map(lambda x: root + x, os.listdir(root)))


# for f in image_folders:
#     # print(f)
#     all_images += glob.glob(os.path.join(f,"*.jpg"))

# print(image_folders)
# print("")
# print(all_images)

root = "C:\\Users\\TRT\\Desktop\\testset\\"
images,labels,all_files = [],[],[]

for file in os.listdir(root):
    print(file)
    images+=glob.glob(os.path.join(root,file,"*.jpg"))
for image in images:
    labels.append(int(image.split(os.sep)[-2]))
all_files=pd.DataFrame({"image":images,"label":labels})

images=[]
for _,row in all_files.iterrows():
    images.append((row["image"],row["label"]))

print(images)