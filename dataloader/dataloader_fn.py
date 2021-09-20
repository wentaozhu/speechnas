#! -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

sys.path.append("./")

from dataloader.utils.prefetcher import DataPrefetcher
from dataloader.voxceleb import get_vexceleb_dataset

fn_map = {
    "voxceleb": get_vexceleb_dataset,
}


def define_dataloader(data_name, train_batch_size, test_batch_size, dataset_type, dataset_root=None, **kwargs):
    if data_name not in fn_map.keys():
        raise KeyError('Unknown Dataset Name!')
    dataloader_fn = fn_map[data_name]
    loader = dataloader_fn(train_batch_size, test_batch_size, dataset_root, dataset_type, **kwargs)
    is_use_cuda = kwargs["config_model"].is_use_cuda
    is_use_prefetcher = kwargs["config_model"].is_use_prefetcher
    if is_use_cuda is True and is_use_prefetcher is True:
        print("Data Prefetcher Enable ...")
        loader = DataPrefetcher(loader)
    return loader


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='For DDP local rank')
#     parser.add_argument('--local_rank',
#                         default=0,
#                         type=int,
#                         help='node rank for distributed training')
#     dataloader = define_dataloader("vexceleb", 64, 64, "train", "/share/ai_platform/lushun/dataset/speech_data/csv_file")
#     print(dataloader)
