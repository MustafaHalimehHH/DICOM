# https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data
import os
import numpy
import pydicom
import cv2
import tqdm
import matplotlib.pyplot as plt


def show_dcm_info(file_path, dataset):
    print('Filename....', file_path)
    print('Storage Type....', dataset.SOPClassUID)

    patient_name = dataset.PatientName
    display_name = patient_name.family_name + ", " + patient_name.given_name
    print('Patient Name....', display_name)
    print('Patient ID....', dataset.PatientID)
    print('Patient Age....', dataset.PatientAge)
    print('Patient Sex....', dataset.PatientSex)
    print('Modality....', dataset.Modality)
    print('Body Part....', dataset.BodyPartExamined)
    # print('View Position....', dataset.ViewPosition)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print('Image Size.... {rows:d} x {cols:d}, {size:d} bytes'.format(rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print('Pixel Spacing....', dataset.PixelSpacing)


def plot_pixel_data(dataset, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


input_dir = 'D:\Halimeh\Datasets\Kaggle\siim-medical-images\dicom_dir'
i = 1
num_to_plot = 5
for file_name in os.listdir(input_dir):
    # print(file_name)
    file_path = os.path.join(input_dir, file_name)
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(file_path, dataset)
    plot_pixel_data(dataset)
    if i > num_to_plot:
        break
    i = i + 1