#!/bin/sh

# define parameters
Model=etdnn_base_rightDataset
Dataset=(train voxceleb1_test)
Output=./embeddings/$Model
GPU=(0 1 2 3 4 5 6 7)
GPU_NUM=8

## create output direction
if [ ! -d "./embeddings" ]; then
  mkdir "./embeddings"
fi
if [ -d "$Output" ]; then
  rm -fr "$Output"
fi
mkdir "$Output"
mkdir "$Output/log"
mkdir "$Output/train"
mkdir "$Output/voxceleb1_test"

# extract embeddings
echo "Extract embeddings by model: $Model"
for((i=0; i<${#Dataset[*]}; i++)); do
  for((j=0; j<${#GPU[*]}; j++)); do
    nohup python -u metric/metric_eer/score.py --model_name $Model --dataset ${Dataset[i]} --gpu_id ${GPU[j]} \
     --gpu_num $GPU_NUM > ./embeddings/$Model/log/${Dataset[i]}_split$((j+1)).log  2>&1 &
  done;
done;

# wait for embeddings
wait

# score
./metric/metric_eer/score.sh $Model $GPU_NUM