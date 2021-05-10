import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image

root = "F:\\dicom\\MC1\\293\\605\\2319\\00000046"
root1 = "F:\\dicom\\MC19\\18753\\26436\\124209\\00000001"
img = sitk.ReadImage(root1)
img_array = sitk.GetArrayFromImage(img)
# img_out = sitk.GetImageFromArray(img_array)
# print(img_array.shape)
