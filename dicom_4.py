# https://www.kaggle.com/akensert/inceptionv3-prev-resnet50-keras-baseline-model/notebook
import os
import sys
import math
import numpy
import pydicom
import cv2
import tensorflow
import keras
from sklearn.model_selection import ShuffleSplit

test_images_dir = ''
train_images_dir = ''


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = numpy.clip(img, img_min, img_max)
    return img


def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = numpy.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)
    return bsb_img


def window_with_correction(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.Pixelrepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = numpy.clip(img, img_min, img_max)
        return img


def window_without_correction(dcm, window_center, window_width):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = numpy.clip(img, img_min, img_max)
    return img


def window_testing(img, window):
    brain_img = window(img, 40, 80)
    subdural_img = window(img, 80, 200)
    soft_img = window(img, 40, 380)
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = numpy.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)
    return bsb_img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 1), img_dir=train_images_dir, *args, **kwargs):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return len(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index+1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):
        if self.labels is not None:
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > numpy.random.rand(len(keep_prob)))
            self.indices = numpy.arange(len(self.list_IDs))[keep]
            numpy.random.shuffle(self.indices)
        else:
            self.indices = numpy.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = numpy.empty((self.batch_size, *self.img_size))
        if self.labels is not None:
            Y = numpy.empty((self.batch_size, 6), dtype=numpy.float32)
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = None
                Y[i,] = self.labels.loc[ID].values
            return X, Y
        else:
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = None
            return X


def weighted_log_loss(y_true, y_pred):
    class_weights = numpy.array([2., 1., 1., 1., 1., 1.])
    eps = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, eps, 1.0 - eps)
    out = - (y_true * keras.backend.log(y_pred) * class_weights + (1.0 - y_true) * keras.backend.log(1.0 - y_pred) * class_weights)
    return keras.backend.mean(out, axis=-1)


def _normalized_weighted_average(arr, weights=None):
    if weights is not None:
        scl = keras.backend.sum(weights)
        weights = keras.backend.expand_dims(weights, axis=1)
        return keras.backend.sum(keras.backend.dot(arr, weights), axis=1) /scl
    return keras.backend.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    class_weights = keras.backend.variable([2., 1., 1., 1., 1., 1.])
    eps = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, eps, 1.0 - eps)
    loss = - (y_true * keras.backend.log(y_pred) + (1.0 - y_true) * keras.backend.log(1.0 - y_pred))
    loss_samples = _normalized_weighted_average(loss, class_weights)
    return keras.backend.mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    class_weights = [2., 1., 1., 1., 1., 1.]
    eps = 1e-7
    preds = numpy.clip(preds, eps, 1 - eps)
    loss = trues * numpy.log(preds) + (1 - trues) * numpy.log(1 - preds)
    loss_samples = numpy.average(loss, axis=1, weights=class_weights)
    return - loss_samples.mean()


class PredictionCheckpoint(keras.callbacks.Callback):
    def __init__(self, test_df, valid_df, test_images_dir=test_images_dir, valid_images_dir=train_images_dir, batch_size=32, input_size=(224, 224, 3)):
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = train_images_dir
        self.batch_size = batch_size
        self.input_size = input_size

    def on_train_begin(self, logs=None):
        self.test_predictions = []
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.test_predictions.append(self.model.predict_generator(
            DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir), verbose=2
        )[:len(self.test_df)])


class MyDeepModel:
    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3, decay_rate=1.0, decay_stpes=1, weights='imagenet', verbose=1):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_stpes
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        out = keras.layers.Dense(6, activation='sigmoid', name='dense_output')
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])

    def fit_and_predict(self, train_df, valid_df, test_df):
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, float(epoch / self.decay_steps)))
        self.model.fit_generator(DataGenerator(
            train_df.index,
            train_df,
            self.batch_size,
            self.input_dims,
            train_images_dir
        ),
        epochs=self.num_epochs,
        verbose=self.verbose,
        use_multiprocessing=True,
        workers=4,
        callbacks=[pred_history, scheduler])