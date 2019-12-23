# https://github.com/ben-heil/DICOM-CNN
import sys
import argparse
import keras
import keras.backend as k
import skimage.exposure
import os
import numpy
import math
import random
import time
import matplotlib.pyplot as plt
import dicom


pos_dir = ''
neg_dir = ''
pos_test_dir = ''
neg_test_dir = ''


def images_count(dir_name):
    for root, dir, files in os.walk('./' + dir_name):
        scans = [file for file in files if file != 'desktop.ini']
        return len(scans)
    print(dir_name + ' is not a valid folder')


def read_scan(scan_num, dir_name):
    for root, dir, files in os.walk('./', dir_name):
        scans = [file for file in files if file != 'desktop.ini']
        print('Reading image ' + scans[scan_num] + ' from ' + dir_name)
        data = numpy.load('./' + dir_name + '/' + scans[scan_num])
        data = skimage.exposure.equalize_adapthist(data)
        return data
    print(dir_name + ' is not a valid folder')


def read_validation_images(pos_dir, neg_dir):
    labels = []
    patient_data = []
    for i in range(0, images_count(pos_dir)):
        data = read_scan(i, pos_dir)
        labels.append(1)
        patient_data.append(data)
    for i in range(0, images_count(neg_dir)):
        data = read_scan(i, neg_dir)
        labels.append(0)
        patient_data.append(data)
    patient_data = numpy.stack(patient_data).astype(float)
    labels = numpy.stack(labels)
    return patient_data, labels


def read_batch(batch_size, pos_dir, neg_dir):
    labels = []
    patient_data = []
    pos_len = images_count(pos_dir)
    neg_len = images_count(neg_dir)
    pos_start = random.randint(0, (pos_len - (batch_size // 2)) - 1)
    neg_start = random.randint(0, (neg_len - (batch_size // 2)) - 1)

    for i in range(batch_size // 2):
        data = read_scan(pos_start + i, pos_dir)
        labels.append(1)
        patient_data.append(data)
        data = read_scan(neg_start + i, neg_dir)
        labels.append(0)
        patient_data.append(data)

    patient_data = numpy.stack(patient_data).astype(float)
    return patient_data, numpy.stack(labels)


def norm_read_all(pos_dir, neg_dir):
    labels = []
    patient_data = []
    pos_len = images_count(pos_dir)
    neg_len = images_count(neg_dir)

    for i in range(pos_len):
        data = read_scan(i, pos_dir)
        labels.append(1)
        data = skimage.exposure.equalize_adapthist(data)
        patient_data.append(data)
    for i in range(neg_len):
        data = read_scan(i, neg_dir)
        labels.append(0)
        data = skimage.exposure.equalize_adapthist(data)
        patient_data.append(data)

    patient_data = numpy.stack(patient_data).astype(float)
    return patient_data, numpy.stack(labels)


def read_test(pos_test_dir, neg_test_dir):
    labels = []
    patient_data = []
    pos_len = images_count(pos_test_dir)
    neg_len = images_count(neg_test_dir)

    for i in range(pos_len):
        data = read_scan(i, pos_test_dir)
        labels.append(1)
        data = skimage.exposure.equalize_adapthist(data)
        patient_data.append(data)
    for j in range(neg_len):
        data = read_scan(j, neg_test_dir)
        labels.append(0)
        data = skimage.exposure.equalize_adapthist(data)
        patient_data.append(data)

    patient_data = numpy.stack(patient_data).astype(float)
    return patient_data, numpy.stack(labels)


def main():
    parser = argparse.ArgumentParser(description='Convolutional Neural Net')
    parser.add_argument('mode', help='specify whether to train or test the model')
    args = parser.parse_args()
    try:
        print('Model Loading')
        model = keras.models.load_model('./CNN.save')
    except:
        print('saved file not found, reinstantiating...')
        model = keras.Sequential([
            keras.layers.Conv2D(16, 15, input_shape=(512, 512, 1)),
            keras.layers.MaxPooling2D(pool_size=(4)),
            keras.layers.Conv2D(16, 7),
            keras.layers.Activation('tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(16, 7),
            keras.layers.Activation('tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(16, 7),
            keras.layers.Activation('tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(4)),
            keras.layers.Conv2D(16, 3),
            keras.layers.Activation('tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(4)),
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation='tanh'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.SGD(lr=1e-2), metrics=['acc'])
        log = keras.callbacks.CSVLogger('KerasLog.csv', append=True)

        if args.mode.lower() == 'test':
            print('Start Testing')
            test_images, test_labels = read_test(pos_test_dir, neg_test_dir)
            print(model.evaluate(test_images.reshape(40, 512, 512, 1), test_labels))
        elif args.mode.lower() == 'train':
            print('Reading Validation Images')
            batch_size = 16
            val_images, val_labels = read_validation_images(pos_dir, neg_dir)
            best_loss = float('inf')
            for i in range(1000):
                batch, labels = read_batch(batch_size)
                history = model.fit(x=batch.reshape(batch_size, 512, 512, 1), y=labels, epochs=5, validation_data=(val_images.reshape(16, 512, 512, 1), val_labels), callbacks=[log])
                curr_loss = history.history['val_loss'][-1]
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    print('Saving model with loss of ' + str(curr_loss))
                    model.save('./CNN.save')


if __name__ == '__main__':
    main()