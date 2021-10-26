from __future__ import print_function, division

import torch

"""
this code is modified from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py#L46
"""


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
            if not first:
                yield input, target
            else:
                first = False
            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        yield input, target

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader)
