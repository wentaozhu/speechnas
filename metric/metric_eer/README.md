## Benchamrk

### Config
1. Specify your KALDI_ROOT at line 25 of score.py, line 20 of score.sh. <br />
2. Specify your kaldi_repo path at line 19 of score.sh. <br />  
3. Define your model and specify the checkpoint path in the extract_embeddings() of score.py.<br />
4. Change the name of the "Model" in auto_run.sh. <br />

### Extract Embeddings and Scoring
```shell
bash ./metric/metric_eer/auto_run.sh
```