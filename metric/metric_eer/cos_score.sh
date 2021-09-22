#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.


# cmd.sh
export train_cmd="run.pl"
export cuda_cmd="run.pl"

# path.sh
kaldi_repo=./kaldi/egs/voxceleb/v2
export KALDI_ROOT=./kaldi
export PATH=$kaldi_repo/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$kaldi_repo:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

set -e


data=./dataset/speech_data
voxceleb1_trials=$data/voxceleb1_test/trials

nnet_dir=$1
stage=9

if [ $stage -le 9 ]; then
  # Cosine similarity
  mkdir -p $nnet_dir/scores
  cat $voxceleb1_trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-normalize-length scp:$nnet_dir/xvector.scp ark:- |" \
      "ark:ivector-normalize-length scp:$nnet_dir/xvector.scp ark:- |" \
      $nnet_dir/scores/scores_voxceleb_test.cos

  eer=`compute-eer <(metric/metric_eer/prepare_for_eer.py $voxceleb1_trials $nnet_dir/scores/scores_voxceleb_test.cos) 2> /dev/null`
  mindcf1=`metric/metric_eer/compute_min_dcf.py --c-miss 10 --p-target 0.01 $nnet_dir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  mindcf2=`metric/metric_eer/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores/scores_voxceleb_test.cos $voxceleb1_trials 2> /dev/null`
  echo "$eer"
  echo "$mindcf1"
  echo "$mindcf2"
fi
