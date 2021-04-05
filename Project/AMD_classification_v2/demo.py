import glob
import os
import pandas as pd
from dataset.dataloader import *
from config import config
from torch.utils.data import DataLoader
from cv2 import cv2
from PIL import Image
# all_images = []
# image_folders = list(map(lambda x: root + x, os.listdir(root)))


# for f in image_folders:
#     # print(f)
#     all_images += glob.glob(os.path.join(f,"*.jpg"))

# print(image_folders)
# print("")
# print(all_images)

# --------------------------------------

# root = "C:\\Users\\TRT\\Desktop\\testset\\"
# images,labels,all_files = [],[],[]

# for file in os.listdir(root):
#     print(file)
#     images+=glob.glob(os.path.join(root,file,"*.jpg"))
# for image in images:
#     labels.append(int(image.split(os.sep)[-2]))
# all_files=pd.DataFrame({"image":images,"label":labels})
# images=[]
# for _,row in all_files.iterrows():
#     images.append((row["image"],row["label"]))

# print(images)

# --------------------------------------

# train_data_list = get_files(config.train_data, "train")
# train_dataloader = DataLoader(ChaojieDataset(train_data_list), batch_size=config.batch_size, shuffle=True,
#                                pin_memory=True, num_workers=0)
# for iter,(image,label) in enumerate(train_dataloader):
#     print(image)

# ---------------------------------------
img = "F:\\Lab\\AMD_CL\\split\\val\\2\\高章红_000007.jpg"
img_1 = Image.open(img).convert("RGB")
img_r1 = img_1.resize((int(config.img_height * 1.5), int(config.img_weight * 1.5)),Image.ANTIALIAS)
print(img_1.size)
# print(img_r1.size)
img_2 = cv2.imread(img)
print(img_2.shape)