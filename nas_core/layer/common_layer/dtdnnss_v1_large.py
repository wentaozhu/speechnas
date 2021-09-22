import torch
import torch.nn as nn
import torch.nn.functional as F

from nas_core.layer.common_layer.dtdnnss_v1 import HighOrderStatsPool
from nas_core.global_enum import Activation, Padding
from nas_core.layer.common_layer.batchnorm1d import BatchNorm1d_
from nas_core.layer.common_layer.conv1d import Conv1d_
from nas_core.layer.common_layer.linear import Linear_
from nas_core.layer.utils import _act


class StatsSelect(nn.Module):

    def __init__(self, in_out_channels, reduction_channels, num_branches, is_folded=False):
        """
        # TODO 该版本的reduction channels可搜索
        """
        super(StatsSelect, self).__init__()
        self.is_folded = is_folded
        self.gather = HighOrderStatsPool()
        self.linear1 = Linear_(in_features=in_out_channels * 4,
                               out_features=reduction_channels,  # TODO here
                               bias=True,
                               act=Activation.Identity,
                               is_folded=self.is_folded)
        self.linear2 = nn.ModuleList()
        for _ in range(num_branches):
            self.linear2.append(Linear_(in_features=reduction_channels,  # TODO and here
                                        out_features=in_out_channels,
                                        bias=True,
                                        act=Activation.Identity,
                                        is_folded=self.is_folded))

    def forward(self, x_a, x_b, x_c=None, x_d=None):  # TODO no change for now
        if x_c is None:
            x = [x_a, x_b]
        elif x_d is None:
            x = [x_a, x_b, x_c]
        else:
            x = [x_a, x_b, x_c, x_d]
        f = torch.cat([_x.unsqueeze(dim=1) for _x in x], dim=1)
        x = torch.sum(f, dim=1)
        x = self.linear1(self.gather(x))
        s = []
        for linear in self.linear2:
            s.append(linear(x).unsqueeze(dim=1))
        s = torch.cat(s, dim=1)
        s = F.softmax(s, dim=1).unsqueeze(dim=-1)
        return torch.sum(f * s, dim=1)

    @property
    def multiply_adds(self):
        madds = self.linear1.multiply_adds
        madds += sum([linear.multiply_adds for linear in self.linear2])
        return madds

    @property
    def params(self):
        params = self.linear1.params
        params += sum([linear.params for linear in self.linear2])
        return params


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


class DTDNNSS_Large_(nn.Module):
    def __init__(self, in_channels,
                 mid_channels,
                 reduction_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=Padding.SAME,
                 dilation=None,
                 groups=1,
                 bias=False,
                 act=Activation.PReLU,
                 is_folded=False,
                 in_shape=None,
                 out_shape=None,
                 ):
        super(DTDNNSS_Large_, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.reduction_channels = reduction_channels
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
        self.act1 = _act(self.act_type, num_parameters=self.in_channels)
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
        self.act2 = _act(self.act_type, num_parameters=self.mid_channels)
        self.conv2 = nn.ModuleList()
        for _dilation in self.dilation:
            self.conv2.append(
                Conv1d_(
                    in_channels=self.mid_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=_dilation,
                    groups=self.groups,
                    bias=self.bias,
                    act=Activation.Identity,
                    is_folded=is_folded,
                    in_shape=in_shape,
                    out_shape=out_shape
                )
            )
        self.select = StatsSelect(in_out_channels=self.out_channels,
                                  reduction_channels=self.reduction_channels,
                                  num_branches=len(self.dilation),
                                  is_folded=self.is_folded)

    def forward(self, inputs):
        result = self.conv1(self.act1(self.bn1(inputs)))
        result = self.act2(self.bn2(result))
        if len(self.dilation) == 2:
            result = self.select(self.conv2[0](result), self.conv2[1](result))
        elif len(self.dilation) == 3:
            result = self.select(self.conv2[0](result), self.conv2[1](result), self.conv2[2](result))
        else:
            result = self.select(
                self.conv2[0](result),
                self.conv2[1](result),
                self.conv2[2](result),
                self.conv2[3](result)
            )
        result = torch.cat([inputs, result], 1)
        return result

    @property
    def multiply_adds(self):
        madds = self.bn1.multiply_adds
        madds += self.conv1.multiply_adds
        madds += self.bn2.multiply_adds
        madds += sum([conv.multiply_adds for conv in self.conv2])
        madds += self.select.multiply_adds
        return madds

    @property
    def params(self):
        params = self.bn1.params
        params += self.in_channels  # TODO 这里算的是第1个PReLU的参数
        params += self.conv1.params
        params += self.bn2.params
        params += self.mid_channels  # TODO 这里算的是第2个PReLU的参数
        params += sum([conv.params for conv in self.conv2])
        params += self.select.params
        return params