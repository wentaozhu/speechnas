from __future__ import print_function, division

import torch.nn as nn


def _padding(kernel_size, padding_type, dilation):
    # some loop import bug outside, so place import inner
    from nas_core.global_enum import Padding
    if isinstance(kernel_size, int):
        if padding_type == Padding.SAME:
            padding = (kernel_size // 2) * dilation
            return padding
        elif padding_type == Padding.VALID:
            padding = 0
            return padding
        else:
            raise Exception('_padding() Unknown padding type!')
    elif isinstance(kernel_size, tuple):
        if padding_type == Padding.SAME:
            padding_0 = (kernel_size[0] // 2) * dilation
            padding_1 = (kernel_size[1] // 2) * dilation
            return padding_0, padding_1
        elif padding_type == Padding.VALID:
            padding = 0
            return padding
        else:
            raise Exception('_padding() Unknown padding type!')


def _pair(n):
    if isinstance(n, tuple) or isinstance(n, list):
        return n
    elif isinstance(n, int) or n is None:
        return n, n
    else:
        raise Exception('Unknown shape type!')


def _act(act_type, num_parameters=0):
    # some loop import bug outside, so place import inner
    from nas_core.global_enum import Activation
    from nas_core.layer.common_layer.activation import Identity_, HSigmoid_, HSwish_
    if act_type is None or act_type == Activation.Identity:
        return Identity_()
    if act_type == Activation.ReLU:
        result = nn.ReLU(inplace=True)
        return result
    if act_type == Activation.LeakyReLU:
        result = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        return result
    if act_type == Activation.Sigmoid:
        result = nn.Sigmoid()
        return result
    if act_type == Activation.ReLU6:
        result = nn.ReLU6(inplace=True)
        return result
    if act_type == Activation.HardSigmoid:
        result = HSigmoid_(inplace=True)
        return result
    if act_type == Activation.HardSwish:
        result = HSwish_(inplace=True)
        return result
    if act_type == Activation.Tanh:
        result = nn.Tanh()
        return result
    if act_type == Activation.PReLU:
        result = nn.PReLU(num_parameters)
        return result
    raise Exception("_act: Go here ..., maybe don\'t define some act")
