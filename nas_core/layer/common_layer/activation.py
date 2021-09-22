from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F


class Identity_(nn.Module):
    """
    skip connect
    """

    def __init__(self):
        super(Identity_, self).__init__()

    def forward(self, inputs):
        return inputs


class HSwish_(nn.Module):
    """
    Hard Swish nonlinearity
    """

    def __init__(self, inplace=True):
        super(HSwish_, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid_(nn.Module):
    """
    Hard Sigmoid nonlinearity
    """

    def __init__(self, inplace=True):
        super(HSigmoid_, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out
