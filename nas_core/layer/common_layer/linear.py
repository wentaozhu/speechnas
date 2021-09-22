import torch.nn as nn

from nas_core.layer.utils import _act
from nas_core.global_enum import Activation


class Linear_(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act=Activation.ReLU, is_folded=True):
        super(Linear_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.act_type = act
        self.is_folded = is_folded
        self.linear = nn.Linear(in_features=self.in_features,
                                out_features=self.out_features,
                                bias=self.bias)
        self.act = _act(self.act_type)

    def forward(self, inputs):
        result_linear = self.linear(inputs)
        result = self.act(result_linear)
        return result

    @property
    def multiply_adds(self):
        result = self.in_features * self.out_features
        return result

    @property
    def params(self):
        params = self.in_features * self.out_features
        # TODO 不考虑fold方式，需要计算bias的参数量
        if self.bias is True and self.is_folded is False:
            params += self.out_features
            # print("%d" % self.out_features)
        return params
