'''
This is an example of a fully convolutional network prediction pipeline.
The data used is from the Histopathologic Cancer Detection playground challenge on Kaggle (link:https://www.kaggle.com/c/histopathologic-cancer-detection)

This module loads the data into memory and applies preprocessing (random rotations and cropping to 64x64).

Author: Peter Zeglen

'''

import cv2 as cv
import numpy as np
import pandas as pd
from os import listdir

np.random.seed(0)

train_size = 49000
validation_size = 1000
d = pd.read_csv("data/train_labels.csv")

#Labels: binary 0, 1 (represents positive case)
labels = d['label'].values[:(train_size + validation_size)]

#Training images: each is associated with a label
images = np.stack([np.float32(cv.imread("data/train/" + d_id + ".tif"))/255.
                   for d_id in d['id'][:(train_size + validation_size)]], axis=0)
print("Training images read in")

#This is the unlabeled set. The entire folder is read in.
test_images = np.stack([np.float32(cv.imread("data/test/" + im_name))/255.
                        for im_name in listdir("data/test/")[:100]], axis=0)
test_size = test_images.shape[0]
print("Test images read in")


#A random rotation is applied to each image
def rotated_cropped(im, rotation, x, y):
    M = cv.getRotationMatrix2D((y, x), rotation, 1)
    rot_im = cv.warpAffine(im, M, (96, 96))[(x - 32):(x + 32), (y - 32):(y + 32), :]
    return rot_im


def get_batch(batch_size=50):
    #Angle of rotation
    angles = np.random.randint(0, 360, size=batch_size)
    inds = np.random.randint(0, train_size, size=batch_size)
    
    #To avoid trying to cut outside of an image, we select the cropping area based on the
    #chosen rotation.
    grid_size = 48 - np.sqrt(2) * np.maximum(np.abs(np.sin(np.radians(angles + 45))),
                                             np.abs(np.cos(np.radians(angles + 45)))) * 32
    #Random cropping
    center_x = np.int32(np.random.uniform(size=batch_size) * 2 * grid_size + 48 - grid_size)
    center_y = np.int32(np.random.uniform(size=batch_size) * 2 * grid_size + 48 - grid_size)

    batch_images = np.stack([
        rotated_cropped(images[ind, :, :, :], rot, x, y) for rot, ind, x, y
        in zip(angles.tolist(), inds.tolist(), center_x.tolist(), center_y.tolist())
    ], axis=0)

    return batch_images, labels[inds]


def get_validation(batch_size=50):
    choice = np.minimum(batch_size, validation_size)
    return images[train_size:(train_size + choice), 16:80, 16:80, :], \
           labels[train_size:(train_size + choice)]


def get_test(batch_size=50):
    inds = np.random.randint(0, test_size, size=batch_size)
    return test_images[inds, 16:80, 16:80, :]


def get_train(batch_size=50):
    inds = np.random.randint(0, train_size, size=batch_size)
    return images[inds, 16:80, 16:80, :]