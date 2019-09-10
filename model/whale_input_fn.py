"""
Create the input data pipeline:
Four possible channels: grey scale(bw), Red, Green, Blue
I ensemble these four channel results for final submission.
"""

import keras
import numpy as np
import random
import pickle
import os
from os.path import isfile
from lapjv import lapjv
from PIL import Image as pil_image
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D
from keras.layers import Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from numpy import zeros, newaxis


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def expand_path(p):
    TRAIN = 'data/train_test_rgb/'
    TEST = 'data/test_rgb/'
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p

def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img

def read_cropped_image_bw(p):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    global h2p
    filename = 'data/metadata/h2p_pickle'
    infile = open(filename,'rb')
    h2p = pickle.load(infile)
    infile.close()
    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    #size_x, size_y = p2size[p]
    #read bw images
    resize_shape = (384, 384)
    img_shape = (384, 384, 1)
    img = read_raw_image(p).convert('L')
    img = img.resize(resize_shape)
    img = img_to_array(img)
    img = img.reshape(img.shape[:-1])
    img = img.reshape(img_shape)
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training_bw(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image_bw(p)


def read_for_validation_bw(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image_bw(p)

def read_cropped_image_red(p):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    global h2p
    filename = 'data/metadata/h2p_pickle'
    infile = open(filename,'rb')
    h2p = pickle.load(infile)
    infile.close()

    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    #size_x, size_y = p2size[p]
    resize_shape = (384, 384)
    img_shape = (384, 384, 1)
    img = read_raw_image(p).convert('RGB')
    img = img.resize(resize_shape)
    img = img_to_array(img)
    #read the red images
    img2= img[:,:,0]
    img = img2[:,:,newaxis]
    img = img.reshape(img.shape[:-1])
    img = img.reshape(img_shape)
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training_red(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image_red(p)

def read_for_validation_red(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image_red(p)

def read_cropped_image_green(p):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    global h2p
    filename = 'data/metadata/h2p_pickle'
    infile = open(filename,'rb')
    h2p = pickle.load(infile)
    infile.close()

    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    #size_x, size_y = p2size[p]
    resize_shape = (384, 384)
    img_shape = (384, 384, 1)
    img = read_raw_image(p).convert('RGB')
    img = img.resize(resize_shape)
    img = img_to_array(img)
    #read the green images
    img2= img[:,:,1]
    img = img2[:,:,newaxis]
    img = img.reshape(img.shape[:-1])
    img = img.reshape(img_shape)
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training_green(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image_green(p)

def read_for_validation_green(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image_green(p)

def read_cropped_image_blue(p):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    global h2p
    filename = 'data/metadata/h2p_pickle'
    infile = open(filename,'rb')
    h2p = pickle.load(infile)
    infile.close()

    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    #size_x, size_y = p2size[p]
    resize_shape = (384, 384)
    img_shape = (384, 384, 1)
    img = read_raw_image(p).convert('RGB')
    img = img.resize(resize_shape)
    img = img_to_array(img)
    #read the blue images
    img2= img[:,:,2]
    img = img2[:,:,newaxis]
    img = img.reshape(img.shape[:-1])
    img = img.reshape(img_shape)
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training_blue(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image_blue(p)


def read_for_validation_blue(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image_blue(p)


class TrainingData(Sequence):

    def __init__(self, score, steps=1000, batch_size=32, channel='bw'):
        """
        @param score: LAP score matrix for the picture matching
        @param steps: the number of epoch we are planning with this score matrix
        Need w2ts, h2p and t2i
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        self.channel = channel
        global w2ts

        for ts in w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2

        if self.channel == 'bw':
            for i in range(0, size, 2):
                a[i, :, :, :] = read_for_training_bw(self.match[j][0])
                b[i, :, :, :] = read_for_training_bw(self.match[j][1])
                c[i, 0] = 1  # This is a match
                a[i + 1, :, :, :] = read_for_training_bw(self.unmatch[j][0])
                b[i + 1, :, :, :] = read_for_training_bw(self.unmatch[j][1])
                c[i + 1, 0] = 0  # Different whales
                j += 1
        if self.channel == 'red':
            for i in range(0, size, 2):
                a[i, :, :, :] = read_for_training_red(self.match[j][0])
                b[i, :, :, :] = read_for_training_red(self.match[j][1])
                c[i, 0] = 1  # This is a match
                a[i + 1, :, :, :] = read_for_training_red(self.unmatch[j][0])
                b[i + 1, :, :, :] = read_for_training_red(self.unmatch[j][1])
                c[i + 1, 0] = 0  # Different whales
                j += 1
        if self.channel == 'green':
            for i in range(0, size, 2):
                a[i, :, :, :] = read_for_training_green(self.match[j][0])
                b[i, :, :, :] = read_for_training_green(self.match[j][1])
                c[i, 0] = 1  # This is a match
                a[i + 1, :, :, :] = read_for_training_green(self.unmatch[j][0])
                b[i + 1, :, :, :] = read_for_training_green(self.unmatch[j][1])
                c[i + 1, 0] = 0  # Different whales
                j += 1
        if self.channel == 'blue':
            for i in range(0, size, 2):
                a[i, :, :, :] = read_for_training_blue(self.match[j][0])
                b[i, :, :, :] = read_for_training_blue(self.match[j][1])
                c[i, 0] = 1  # This is a match
                a[i + 1, :, :, :] = read_for_training_blue(self.unmatch[j][0])
                b[i + 1, :, :, :] = read_for_training_blue(self.unmatch[j][1])
                c[i + 1, 0] = 0  # Different whales
                j += 1

        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0: return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        x, _, _ = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=64, verbose=1, channel='bw'):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        self.channel = channel
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=K.floatx())

        if self.channel=='bw':
            for i in range(size): a[i, :, :, :] = read_for_validation_bw(self.data[start + i])
        if self.channel=='red':
            for i in range(size): a[i, :, :, :] = read_for_validation_red(self.data[start + i])
        if self.channel=='green':
            for i in range(size): a[i, :, :, :] = read_for_validation_green(self.data[start + i])
        if self.channel=='blue':
            for i in range(size): a[i, :, :, :] = read_for_validation_blue(self.data[start + i])

        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size
