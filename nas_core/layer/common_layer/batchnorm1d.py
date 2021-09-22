import torch.nn as nn

from nas_core.global_enum import Activation
from nas_core.layer.utils import _act


class BatchNorm1d_(nn.Module):
    def __init__(self, num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 act=Activation.ReLU,
                 is_folded=True):
        super(BatchNorm1d_, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.act_type = act
        self.act = _act(self.act_type)
        self.is_folded = is_folded

        self.bn = nn.BatchNorm1d(num_features=self.num_features,
                                 eps=self.eps,
                                 momentum=self.momentum,
                                 affine=self.affine,
                                 track_running_stats=self.track_running_stats)

    def forward(self, inputs):
        result_bn = self.bn(inputs)
        result = self.act(result_bn)
        return result

    @property
    def multiply_adds(self):
        if self.is_folded is True:
            # when using, bn's weight can be folded into conv, so we don't need to calc BN's MADD
            return 0
        else:
            # TODO 不考虑fold方式，需要计算bn的计算量
            madds = self.num_features * 2
            return madds

    @property
    def params(self):
        if self.is_folded is True:
            # when using, bn's weight can be folded into conv, so we don't need to calc BN's params
            return 0
        else:
            # TODO 不考虑fold方式，需要计算bn的参数量
            params = self.num_features * 2
            return params

    @property
    def latency(self):
        return None
