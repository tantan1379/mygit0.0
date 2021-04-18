from skimage import feature, exposure
import cv2
import numpy as np


image = cv2.imread('gamma.jpg')
fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
cv2.imshow('hog_image', hog_image)
cv2.imshow('img', image)
cv2.imshow('hog', hog_image_rescaled)
cv2.waitKey(0)


# img = cv2.imread('gamma.jpg',0)
# # img = img.resize(600,300)
# img = np.float32(img) / 255.0  # 归一化​
# # 计算x和y方向的梯度
# gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
# gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
# # 计算合梯度的幅值和方向（角度）
# mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
# cv2.imshow("gx",gx)
# cv2.imshow("gy",gy)
# cv2.imshow("mag",mag)
# cv2.imshow("angle",angle)
# cv2.waitKey(0)