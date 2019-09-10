"""Utility functions for training"""

import keras
import numpy as np
import pickle
import random
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D
from keras.layers import Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from tqdm import tqdm_notebook as tqdm
from model.whale_input_fn import *
from model.whale_model import *


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m

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
        img_shape = (384, 384, 1)
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



def compute_score(branch_model, head_model, train, verbose=1, channel='bw' ):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose, channel=channel), max_queue_size=12, workers=6,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
    score = score_reshape(score, features)
    return features, score

def make_steps(step, ampl, train, branch_model, head_model, chan='bw'):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2hs, w2ts, h2p, t2i, steps, img_shape, features, score, histories
    img_shape = (384, 384, 1)

    filename = 'data/metadata/w2hs_pickle'
    infile = open(filename,'rb')
    w2hs = pickle.load(infile)
    infile.close()

    filename = 'data/metadata/h2p_pickle'
    infile = open(filename,'rb')
    h2p = pickle.load(infile)
    infile.close()

    # shuffle the training pictures
    random.shuffle(train)

    # Map whale id to the list of associated training picture hash value
    w2ts = {}
    for w, hs in w2hs.items():
        for h in hs:
            if h in train:
                if w not in w2ts: w2ts[w] = []
                if h not in w2ts[w]: w2ts[w].append(h)
    for w, ts in w2ts.items(): w2ts[w] = np.array(ts)

    # Map training picture hash value to index in 'train' array
    t2i = {}
    for i, t in enumerate(train): t2i[t] = i

    # Compute the match score for each picture pair
    features, score = compute_score(branch_model, head_model, train, channel=chan)

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=24,channel=chan),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=1).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)
