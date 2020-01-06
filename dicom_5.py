# https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
import numpy
import os
import math
import pandas
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


INPUT_FOLDER = 'D:\Halimeh\Datasets\Kaggle\siim-medical-images\dicom_dir'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
print('patients', patients)


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = numpy.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    image = numpy.stack([s.pixel_array for s in slices])
    image = image.astype(numpy.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(numpy.float64)
            image[slice_number] = image[slice_number].astype(numpy.int16)
        image[slice_number] += numpy.int16(intercept)
    return numpy.array(image, dtype=numpy.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    spacing = numpy.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=numpy.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = numpy.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def plot_3d(image, threshold=-300):
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def largest_label_volume(img, bg=-1):
    vals, counts = numpy.unique(img, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[numpy.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    binary_image = numpy.array(image > -320, dtype=numpy.int8) + 1
    labels = measure.label(binary_image)
    background_labels = labels[0, 0, 0]
    binary_image[background_labels == labels] = 2
    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None:
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1
    binary_image = 1 - binary_image
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:
        binary_image[labels != l_max] = 0
    return binary_image


def normalize(image):
    image = (image - (-1e3)) / (4e2 - (-1e3))
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def zero_centring(image):
    image = image - 25e-2
    return image