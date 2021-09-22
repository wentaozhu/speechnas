# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-06-05)

import os
import sys
import time
import torch
import random
import argparse

sys.path.insert(0, '.')

import dataloader.utils.kaldi_io as kaldi_io
from config.global_enum import CheckpointKey

#################################################
# Adding kaldi tools to shell path,

# Select kaldi,
os.environ['KALDI_ROOT'] = './kaldi'
# Add kaldi tools to path,
path = os.popen('echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src'
                '/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src'
                '/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src'
                '/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin'
                '/:$KALDI_ROOT/src/lmbin/')
os.environ['PATH'] = path.readline().strip() + ':' + os.environ['PATH']
path.close()


def get_cosine_eer(model, model_name, dataset='voxceleb1_test'):
    output = './embeddings_cosine/%s/%s-%s' % (
        model_name, time.strftime("%Y%m%d-%H%M%S"), str(random.random())[2:10]
    )

    if not os.path.exists('./embeddings_cosine'):
        os.mkdir('./embeddings_cosine')
    if not os.path.exists('./embeddings_cosine/%s' % model_name):
        os.mkdir('./embeddings_cosine/%s' % model_name)
    if not os.path.exists(output):
        os.mkdir(output)

    data_split = './dataset/speech_data/%s/split1order' % dataset
    input_feats_path = 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ' \
                       'scp:%s/%s/feats.scp ark:- |' % (data_split, 1)
    output_vectors_path = 'ark:| copy-vector ark:- ark,scp:%s/xvector.ark,%s/xvector.scp ' \
                          % (output, output)

    model.eval()
    with torch.no_grad():
        with kaldi_io.open_or_fd(input_feats_path, "rb") as r, \
                kaldi_io.open_or_fd(output_vectors_path, 'wb') as w:
            data_idx = 0
            while True:
                data_idx += 1
                key = kaldi_io.read_key(r)
                if not key:
                    break
                # if data_idx % 1000 == 0:
                #     print("Process {0}th utterance for key {1}".format(data_idx, key))
                feats = kaldi_io.read_mat(r)
                try:
                    embedding = model.extract_embedding(feats)
                except:
                    model = model.module
                    embedding = model.extract_embedding(feats)
                kaldi_io.write_vec_flt(w, embedding.numpy(), key=key)
    params = model.get_param()
    madds = model.get_madds()
    process = os.popen('metric/metric_eer/cos_score.sh %s' % output)
    output = process.read().strip().split('\n')
    process.close()

    return float(output[0]), float(output[1]), float(output[2]), params, madds
