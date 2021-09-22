import torch
import torch.nn as nn

from nas_core.global_enum import Activation, Padding
from nas_core.layer.common_layer.batchnorm1d import BatchNorm1d_
from nas_core.layer.common_layer.conv1d import Conv1d_
from nas_core.layer.utils import _act


class TDNN_(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=Padding.SAME,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act=Activation.ReLU,
                 reduction=4,
                 is_use_skip_connect=False,
                 is_folded=False,
                 in_shape=None,
                 out_shape=None
                 ):
        super(TDNN_, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.act_type = act
        self.is_use_skip_connect = is_use_skip_connect
        self.is_folded = is_folded
        self.padding = padding
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.conv = Conv1d_(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            act=Activation.Identity,
            is_folded=is_folded,
            in_shape=in_shape,
            out_shape=out_shape
        )
        self.bn = BatchNorm1d_(num_features=self.out_channels,
                               momentum=0.1,
                               affine=True,
                               act=Activation.Identity,
                               is_folded=is_folded)
        if self.act_type == Activation.PReLU:
            self.act = _act(self.act_type, num_parameters=self.out_channels)
        else:
            self.act = _act(self.act_type)

    def forward(self, inputs):
        result = self.conv(inputs)
        result = self.bn(result)
        result = self.act(result)
        if self.is_use_skip_connect is True and self.in_channels == self.out_channels:
            result = result + inputs
        return result

    @property
    def multiply_adds(self):
        madds = self.conv.multiply_adds
        madds += self.bn.multiply_adds
        return madds

    @property
    def params(self):
        params = self.conv.params
        params += self.bn.params
        if self.act_type == Activation.PReLU:
            params += self.out_channels
        return params


class StatisticsPooling(nn.Module):
    """ An usual mean [+ stddev] poolling layer"""

    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-6):
        super(StatisticsPooling, self).__init__()
        self.stddev = stddev
        self.input_dim = input_dim
        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim
        self.eps = eps
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        params: inputs  格式[batch-size, len-frames, channels]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        counts = inputs.shape[2]
        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev:
            if self.unbiased and counts > 1:
                counts = counts - 1
            var = torch.sum((inputs - mean) ** 2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean


if __name__ == "__main__":
    op = TDNN_(in_channels=2, out_channels=32, kernel_size=3, dilation=1, bias=False, is_use_se=True, in_shape=8, out_shape=8)
    inputs = torch.ones([2, 2, 200])
    result = op(inputs)

    print(result)
    print(op.multiply_adds)
    print(op.params)
