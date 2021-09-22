import torch
import torch.nn as nn

from config.global_enum import Activation, Padding
from nas_core.layer.common_layer.batchnorm1d import BatchNorm1d_
from nas_core.layer.common_layer.conv1d import Conv1d_
from nas_core.layer.utils import _act


class TransitLayer(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 bias=False,
                 is_folded=False,
                 act=Activation.ReLU,
                 in_shape=None,
                 out_shape=None):
        super(TransitLayer, self).__init__()
        self.bn = BatchNorm1d_(num_features=in_channels,
                               momentum=0.1,
                               affine=True,
                               act=Activation.Identity,
                               is_folded=is_folded)
        self.act = _act(act, in_channels)
        self.conv = Conv1d_(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=Padding.SAME,
            dilation=1,
            groups=1,
            bias=bias,
            act=Activation.Identity,
            is_folded=is_folded,
            in_shape=in_shape,
            out_shape=out_shape
        )

    def forward(self, x):
        x = self.act(self.bn(x))
        x = self.conv(x)
        return x

    @property
    def multiply_adds(self):
        madds = self.bn.multiply_adds
        madds += self.conv.multiply_adds
        return madds

    @property
    def params(self):
        params = self.bn.params
        params += self.conv.params
        return params


class DenseLayer(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 bias=True,
                 is_folded=False,
                 act=Activation.ReLU,
                 in_shape=None,
                 out_shape=None):
        super(DenseLayer, self).__init__()
        self.conv = Conv1d_(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=Padding.SAME,
            dilation=1,
            groups=1,
            bias=bias,
            act=Activation.Identity,
            is_folded=is_folded,
            in_shape=in_shape,
            out_shape=out_shape
        )
        self.bn = BatchNorm1d_(num_features=out_channels,
                               momentum=0.1,
                               affine=False,
                               act=Activation.Identity,
                               is_folded=is_folded)
        self.act = _act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    @property
    def multiply_adds(self):
        madds = self.conv.multiply_adds
        # madds += self.bn.multiply_adds
        return madds

    @property
    def params(self):
        # params = self.bn.params
        params = self.conv.params
        return params


class DTDNN_(nn.Module):
    def __init__(self, in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=Padding.SAME,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act=Activation.ReLU,
                 is_folded=False,
                 in_shape=None,
                 out_shape=None
                 ):
        super(DTDNN_, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.act_type = act
        self.is_folded = is_folded
        self.padding = padding
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.bn1 = BatchNorm1d_(num_features=self.in_channels,
                                momentum=0.1,
                                affine=True,
                                act=Activation.Identity,
                                is_folded=is_folded)
        self.act1 = _act(self.act_type)
        self.conv1 = Conv1d_(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=1,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=self.groups,
            bias=self.bias,
            act=Activation.Identity,
            is_folded=is_folded,
            in_shape=in_shape,
            out_shape=out_shape
        )

        self.bn2 = BatchNorm1d_(num_features=self.mid_channels,
                                momentum=0.1,
                                affine=True,
                                act=Activation.Identity,
                                is_folded=is_folded)
        self.act2 = _act(self.act_type)
        self.conv2 = Conv1d_(
            in_channels=self.mid_channels,
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

    def forward(self, inputs):
        result = self.conv1(self.act1(self.bn1(inputs)))
        result = self.conv2(self.act2(self.bn2(result)))
        result = torch.cat([inputs, result], 1)
        return result

    @property
    def multiply_adds(self):
        madds = self.bn1.multiply_adds
        madds += self.conv1.multiply_adds
        madds = self.bn2.multiply_adds
        madds += self.conv2.multiply_adds
        return madds

    @property
    def params(self):
        params = self.bn1.params
        params += self.conv1.params
        params += self.bn2.params
        params += self.conv2.params
        return params


if __name__ == "__main__":
    inputs = torch.ones([2, 128, 8])
    op = DTDNN_(in_channels=128, mid_channels=128, out_channels=64, kernel_size=3,
                dilation=1, bias=False, in_shape=8, out_shape=8)
    result = op(inputs)

    print(result.shape)
    print(op.multiply_adds)
    print(op.params)
