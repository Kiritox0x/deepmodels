import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

CIFAR_PATH = 'cifar-10-batches-py'
DATA_BATCH_PREFIX = 'data_batch_'
TEST_BATCH_PATH = 'test_batch'


def unpickle(filepath):
    with open(filepath, 'rb') as fr:
        d = pickle.load(fr, encoding="bytes")
    return d


def load():
    for file in os.listdir(CIFAR_PATH):
        if file.startswith(DATA_BATCH_PREFIX):
            d = unpickle(CIFAR_PATH + '/' + file)
            try:
                x_train = np.concatenate((x_train, d[b'data']), axis=0)
            except NameError:
                x_train = d[b'data']
            try:
                y_train = y_train + d[b'labels']
            except NameError:
                y_train = d[b'labels']

    c = unpickle(CIFAR_PATH + '/' + TEST_BATCH_PATH)
    x_test = c[b'data']
    y_test = c[b'labels']

    x_train = x_train.reshape(-1, 3, 32, 32)
    x_train = np.float32(x_train)

    x_test = x_test.reshape(-1, 3, 32, 32)
    x_test = np.float32(x_test)

    mean_r = np.mean(x_train[:, 0])
    mean_g = np.mean(x_train[:, 1])
    mean_b = np.mean(x_train[:, 2])

    x_train[:, 0, :, :] = x_train[:, 0, :, :] - mean_r
    x_train[:, 1, :, :] = x_train[:, 1, :, :] - mean_g
    x_train[:, 2, :, :] = x_train[:, 2, :, :] - mean_b

    x_test[:, 0, :, :] = x_test[:, 0, :, :] - mean_r
    x_test[:, 1, :, :] = x_test[:, 1, :, :] - mean_g
    x_test[:, 2, :, :] = x_test[:, 2, :, :] - mean_b

    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))

    y_train_onehot = np.eye(np.max(y_train) + 1)[y_train]
    y_test_onehot = np.eye(np.max(y_test) + 1)[y_test]

    return x_train, y_train_onehot, x_test, y_test_onehot


def display(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()
