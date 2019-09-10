"""Train the model"""

import argparse
import os
import random

import pickle
import sys
from lapjv import lapjv
from os.path import isfile

from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D
from keras.layers import Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
from itertools import islice
import time
import datetime
start_t = time.time()
#print("start time:", start_t)
print("Kernel start time:",datetime.datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing model weights etc.")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    #set the random seed
    random.seed(42)

    # Reading Files and Labels for training and testing
    global img_shape, resize_shape, w2ts, t2i, w2hs, steps, features, score, histories

    args = parser.parse_args()
    TRAIN_DF = os.path.join(args.data_dir, 'train_test_label.csv')
    SUB_Df = os.path.join(args.data_dir, 'sample_submission.csv')
    TRAIN = os.path.join(args.data_dir,'train_test_rgb/')
    TEST = os.path.join(args.data_dir,'test_rgb/')
    P2H = os.path.join(args.data_dir,'metadata/p2h.pickle')
    P2SIZE = os.path.join(args.data_dir,'metadata/p2size.pickle')
    BB_DF = os.path.join(args.data_dir,'metadata/bounding_boxes.csv')

    tagged = dict([(p, w) for _, p, w in pd.read_csv(TRAIN_DF).to_records()])
    submit = [p for _, p, _ in pd.read_csv(SUB_Df).to_records()]
    join = list(tagged.keys()) + submit

    img_shape = (384, 384, 1)  # The image shape used by the model
    resize_shape = (384, 384)
    total_whale_files = list(tagged.keys()) + submit
    print("tagged whales: ",len(tagged))
    print("submitted whales: ",len(submit))

    # Load all needed input dict, and prepare shuffled train_set
    filename = 'data/metadata/h2p_pickle'
    infile = open(filename,'rb')
    h2p = pickle.load(infile)
    infile.close()

    filename = 'data/metadata/t2i_pickle'
    infile = open(filename,'rb')
    t2i = pickle.load(infile)
    infile.close()

    filename = 'data/metadata/train_set_pickle'
    infile = open(filename,'rb')
    train_set = pickle.load(infile)
    infile.close()

    filename = 'data/metadata/w2ts_pickle'
    infile = open(filename,'rb')
    w2ts = pickle.load(infile)
    infile.close()

    filename = 'data/metadata/p2h_pickle'
    infile = open(filename,'rb')
    p2h = pickle.load(infile)
    infile.close()

    filename = 'data/metadata/w2hs_pickle'
    infile = open(filename,'rb')
    w2hs = pickle.load(infile)
    infile.close()

    train = []  # A list of training image ids
    for hs in w2hs.values():
        if len(hs) > 1:
            train += hs
    random.shuffle(train)
    train_set = set(train)

    from model.whale_input_fn import *
    from model.whale_model import *
    from model.whale_training import *


    #Model Training with red channel images
    img_shape = (384, 384, 1)
    resize_shape = (384, 384)
    histories = []
    steps = 0
    channel='red'
    tmp = keras.models.load_model('experiments/happy_whale_red_rgbtest_train02.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002, img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    set_lr(model, 4e-6)
    for _ in range(4): make_steps(8, 0.25, train, branch_model, head_model, chan=channel)
    model.save('experiments/happy_whale_red_rgbtest_train03.model')

    #Model Training with green channel images
    histories = []
    steps = 0
    channel='blue'
    tmp = keras.models.load_model('experiments/happy_whale_green_rgbtest_train02.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002,img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    set_lr(model, 4e-6)
    for _ in range(4): make_steps(8, 0.25, train, branch_model, head_model, chan=channel)
    model.save('experiments/happy_whale_green_rgbtest_train03.model')

    #Model Training with blue channel images
    histories = []
    steps = 0
    channel='blue'
    tmp = keras.models.load_model('experiments/happy_whale_blue_rgbtest_train02.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002,img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    set_lr(model, 4e-6)
    for _ in range(4): make_steps(8, 0.25, train, branch_model, head_model, chan=channel)
    model.save('experiments/happy_whale_blue_rgbtest_train03.model')

    #Model Training with grey scale images
    histories = []
    steps = 0
    channel='blue'
    tmp = keras.models.load_model('experiments/happy_whale_bw_rgbtest_train02.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002,img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    set_lr(model, 4e-6)
    for _ in range(4): make_steps(8, 0.25, train, branch_model, head_model, chan=channel)
    model.save('experiments/happy_whale_bw_rgbtest_train03.model')

    start_e = time.time()
    print("run time: ", (start_e-start_t)/60)
    print("Kernel end time:",datetime.datetime.now())
