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


    #Model Prediction with red channel images
    img_shape = (384, 384, 1)
    resize_shape = (384, 384)
    histories = []
    steps = 0
    channel='red'
    tmp = keras.models.load_model('experiments/happy_whale_red_rgbtest_train03.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002, img_shape=img_shape)
    model.set_weights(tmp.get_weights())

    # Find elements from training sets not 'new_whale'
    h2ws = {}
    new_whale ='new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i = {}
    for i, h in enumerate(known): h2i[h] = i

    fknown = branch_model.predict_generator(FeatureGen(known, channel=channel), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit, channel=channel), max_queue_size=20, workers=10, verbose=0)
    score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fsubmit)

    score_blue = score
    filename = 'experiments/score_red_test_train03'
    outfile = open(filename,'wb')
    pickle.dump(score,outfile)
    outfile.close()


    #Model Training with green channel images
    histories = []
    steps = 0
    channel='blue'
    tmp = keras.models.load_model('experiments/happy_whale_green_rgbtest_train03.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002,img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    # Find elements from training sets not 'new_whale'
    h2ws = {}
    new_whale ='new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i = {}
    for i, h in enumerate(known): h2i[h] = i

    fknown = branch_model.predict_generator(FeatureGen(known, channel=channel), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit, channel=channel), max_queue_size=20, workers=10, verbose=0)
    score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fsubmit)

    score_blue = score
    filename = 'experiments/score_green_test_train03'
    outfile = open(filename,'wb')
    pickle.dump(score,outfile)
    outfile.close()


    #Model Prediction with blue channel images
    histories = []
    steps = 0
    channel='blue'
    tmp = keras.models.load_model('experiments/happy_whale_blue_rgbtest_train03.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002,img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    # Find elements from training sets not 'new_whale'
    h2ws = {}
    new_whale ='new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i = {}
    for i, h in enumerate(known): h2i[h] = i

    fknown = branch_model.predict_generator(FeatureGen(known, channel=channel), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit, channel=channel), max_queue_size=20, workers=10, verbose=0)
    score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fsubmit)

    score_blue = score
    filename = 'experiments/score_blue_test_train03'
    outfile = open(filename,'wb')
    pickle.dump(score,outfile)
    outfile.close()


    #Model Prediction with grey scale images
    histories = []
    steps = 0
    channel='blue'
    tmp = keras.models.load_model('experiments/happy_whale_bw_rgbtest_train03.model')
    ampl = 100.0
    ampl = max(1.0, 100 ** -0.1 * ampl)
    model, branch_model, head_model = build_model(64e-5, 0.0002,img_shape=img_shape)
    model.set_weights(tmp.get_weights())
    # Find elements from training sets not 'new_whale'
    h2ws = {}
    new_whale ='new_whale'
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i = {}
    for i, h in enumerate(known): h2i[h] = i

    fknown = branch_model.predict_generator(FeatureGen(known, channel=channel), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit, channel=channel), max_queue_size=20, workers=10, verbose=0)
    score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fsubmit)

    score_bw = score
    filename = 'experiments/score_bw_test_train03'
    outfile = open(filename,'wb')
    pickle.dump(score,outfile)
    outfile.close()

    # Maximum Probablity Selection from 4 channels
    score =np.zeros((score_bw.shape[0],score_bw.shape[1]))
    for nn in range(score_bw.shape[0]):
        for mm in range(score_bw.shape[1]):
            score[nn,mm]=np.max([score_blue[nn,mm],score_green[nn,mm],score_red[nn,mm],score_bw[nn,mm]])

    score_max_proba = score
    filename = 'experiments/score_max_test_bw_rgb'
    outfile = open(filename,'wb')
    pickle.dump(score,outfile)
    outfile.close()
    print(score.sum(),score.mean(),score.max(),score.std())


    # Generate the subsmission file.
    def prepare_submission(threshold, filename):
        """
        Generate a Kaggle submission file.
        @param threshold the score given to 'new_whale'
        @param filename the submission file name
        """
        vtop = 0
        vhigh = 0
        pos = [0, 0, 0, 0, 0, 0]
        with open(filename, 'wt', newline='\n') as f:
            f.write('Image,Id\n')
            for i, p in enumerate(tqdm(submit)):
                t = []
                s = set()
                a = score[i, :]
                for j in list(reversed(np.argsort(a))):
                    h = known[j]
                    if a[j] < threshold and new_whale not in s:
                        pos[len(t)] += 1
                        s.add(new_whale)
                        t.append(new_whale)
                        if len(t) == 5: break;
                    for w in h2ws[h]:
                        assert w != new_whale
                        if w not in s:
                            if a[j] > 1.0:
                                vtop += 1
                            elif a[j] >= threshold:
                                vhigh += 1
                            s.add(w)
                            t.append(w)
                            if len(t) == 5: break;
                    if len(t) == 5: break;
                if new_whale not in s: pos[5] += 1
                assert len(t) == 5 and len(s) == 5
                f.write(p + ',' + ' '.join(t[:5]) + '\n')
        return vtop, vhigh, pos

    prepare_submission(0.999, 'submission_max_test_bw_rgb_train03.csv')
    end_t = time.time()
    print("Prediction time (hrs): ", (end_t - start_t) / 3600.)
