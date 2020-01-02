# https://www.kaggle.com/adkarhe/dicom-images
import os
import pydicom
import glob
import PIL.Image
import matplotlib.pyplot as plt


input_dir = 'D:\Halimeh\Datasets\Kaggle\siim-medical-images\dicom_dir'
output_dir = './'

fig, axs = plt.subplots(2, 5, figsize=(20, 10))
test_list = [os.path.basename(x) for x in glob.glob(input_dir + './*.dcm')]
for f, ax in zip(test_list, axs.flatten()):
    dc = pydicom.read_file(os.path.join(input_dir, f))
    print('type dc', type(dc))
    img = dc.pixel_array
    print('type img', type(img), img.shape)
    img_mem = PIL.Image.fromarray(img)
    print('type img_mem', type(img_mem))
    # ax.imshow(img_mem)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.show()
