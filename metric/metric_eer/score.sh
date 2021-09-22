#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

model_name=$1
echo "Model Scoring: $model_name"

# cmd.sh
export train_cmd="run.pl"
export cuda_cmd="run.pl"

# path.sh
kaldi_repo=/home/lushun/code/kaldi/egs/voxceleb/acoustic_repo
export KALDI_ROOT=/home/lushun/code/kaldi
export PATH=$kaldi_repo/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$kaldi_repo:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

set -e


data=/share/ai_platform/lushun/dataset/speech_data

voxceleb1_trials=$data/voxceleb1_test/trials

nnet_dir=./embeddings/$model_name
exp=$nnet_dir
stage=10
nj=$2

echo "combining train xvectors"
train_emb=$nnet_dir/train
for j in $(seq $nj); do cat $train_emb/xvector.$j.scp; done >$train_emb/xvector.scp || exit 1;

echo "combining test xvectors"
test_emb=$nnet_dir/voxceleb1_test
for j in $(seq $nj); do cat $test_emb/xvector.$j.scp; done >$test_emb/xvector.scp || exit 1;

if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/train/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/train/xvector.scp \
    $nnet_dir/train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnet_dir/train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/train/xvector.scp ark:- |" \
    ark:$data/train/utt2spk $nnet_dir/train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/train/log/plda.log \
    ivector-compute-plda ark:$data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/train/xvector.scp ark:- | transform-vec $nnet_dir/train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/train/plda || exit 1;
  echo "stage 10 is ok!"
  #exit 2
fi

if [ $stage -le 11 ]; then
  $train_cmd $exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/train/mean.vec scp:$nnet_dir/voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/train/mean.vec scp:$nnet_dir/voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $exp/scores_voxceleb1_test || exit 1;
  echo "stage 11 is ok!"
  #exit 2
fi

if [ $stage -le 12 ]; then
  echo "stage 12 is begin!"
  eer=`compute-eer <(metric/metric_eer/prepare_for_eer.py $voxceleb1_trials $exp/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`metric/metric_eer/compute_min_dcf.py --p-target 0.01 $exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`metric/metric_eer/compute_min_dcf.py --p-target 0.001 $exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  echo "stage 12 is ok!"
  #exit 2
fi
