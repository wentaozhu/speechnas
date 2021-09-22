# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-06-05)

import sys
import torch
import argparse

sys.path.insert(0, '')
sys.path.insert(0, 'subtools/pytorch')
sys.path.insert(0, 'subtools/pytorch/libs')

import utils
import dataloader.utils.kaldi_io as kaldi_io
from nas_core.global_enum import CheckpointKey

import os

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

# Parse
parser = argparse.ArgumentParser(description="Extract embeddings form a piece of feats.scp or pipeline")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--use_gpu", type=str, default='true')
parser.add_argument("--gpu_id", type=str, default="", help="Specify a fixed gpu.")
parser.add_argument("--gpu_num", type=int, default="", help="GPUs.")
parser.add_argument("--ckpt_path", type=str, help="The model checkpoint.")
args = parser.parse_args()
args.output = './embeddings/%s/%s' % (args.model_name, args.dataset)
print(args)

# if not os.path.exists('./embeddings'):
#     os.mkdir('./embeddings')
# if not os.path.exists('./embeddings/%s' % args.model_name):
#     os.mkdir('./embeddings/%s' % args.model_name)
# if not os.path.exists(args.output):
#     os.mkdir(args.output)

if args.gpu_num == 8:
    data_split = './dataset/speech_data/%s/split8order' % args.dataset
elif args.gpu_num == 4:
    data_split = './dataset/speech_data/%s/split4order' % args.dataset
else:
    raise NotImplementedError

input_feats_path = 'ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ' \
                   'scp:%s/%s/feats.scp ark:- |' % (data_split, int(args.gpu_id) + 1)
output_vectors_path = 'ark:| copy-vector ark:- ark,scp:%s/xvector.%s.ark,%s/xvector.%s.scp ' \
                      % (args.output, int(args.gpu_id) + 1, args.output, int(args.gpu_id) + 1)


def extract_embeddings():
    # model define
    if args.model_name == 'dtdnn_base_rightDataset':
        from network.dtdnnss_searched import DtdnnssBase_v1
        model = DtdnnssBase_v1(num_class=7323)
        args.ckpt_path = './checkpoint/best_model.pth.rightDataset'
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')[CheckpointKey.ModelState], strict=False)
    else:
        raise NotImplementedError

    # load params
    model = utils.select_model_device(model, args.use_gpu, gpu_id=args.gpu_id)
    model.eval()

    print('load params done')
    # extract embeddings
    with kaldi_io.open_or_fd(input_feats_path, "rb") as r, \
            kaldi_io.open_or_fd(output_vectors_path, 'wb') as w:
        data_idx = 0
        while True:
            data_idx += 1
            key = kaldi_io.read_key(r)
            if not key:
                break
            if data_idx % 1000 == 0:
                print("Process {0}th utterance for key {1}".format(data_idx, key))
            feats = kaldi_io.read_mat(r)
            embedding = model.extract_embedding(feats)
            kaldi_io.write_vec_flt(w, embedding.numpy(), key=key)


if __name__ == '__main__':
    extract_embeddings()
