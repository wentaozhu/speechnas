import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.global_enum import Activation, Padding
from nas_core.layer.common_layer.batchnorm1d import BatchNorm1d_
from nas_core.layer.common_layer.conv1d import Conv1d_
from nas_core.layer.utils import _act


def high_order_statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    norm = (x - mean.unsqueeze(dim=dim)) / std.clamp(min=eps).unsqueeze(dim=dim)
    skewness = norm.pow(3).mean(dim=dim)
    kurtosis = norm.pow(4).mean(dim=dim)
    stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class HighOrderStatsPool(nn.Module):

    def forward(self, x):
        return high_order_statistics_pooling(x)


class StatsSelect(nn.Module):

    def __init__(self, channels, branches, null=False, reduction=1):
        super(StatsSelect, self).__init__()
        self.gather = HighOrderStatsPool()
        self.linear1 = nn.Linear(channels * 4, channels // reduction)
        self.linear2 = nn.ModuleList()
        if null:
            branches += 1
        for _ in range(branches):
            self.linear2.append(nn.Linear(channels // reduction, channels))
        self.channels = channels
        self.branches = branches
        self.null = null
        self.reduction = reduction

    # def forward(self, x):
    def forward(self, x_a, x_b):
        x = [x_a, x_b]

        f = torch.cat([_x.unsqueeze(dim=1) for _x in x], dim=1)
        x = torch.sum(f, dim=1)
        x = self.linear1(self.gather(x))
        s = []
        for linear in self.linear2:
            s.append(linear(x).unsqueeze(dim=1))
        s = torch.cat(s, dim=1)
        s = F.softmax(s, dim=1).unsqueeze(dim=-1)
        if self.null:
            s = s[:, :-1, :, :]
        return torch.sum(f * s, dim=1)

    def extra_repr(self):
        return 'channels={}, branches={}, reduction={}'.format(
            self.channels, self.branches, self.reduction
        )


class DTDNNSS_(nn.Module):
    def __init__(self, in_channels,
                 mid_channels,
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
                 null=False,
                 reduction=2
                 ):
        super(DTDNNSS_, self).__init__()
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
        self.act1 = _act(self.act_type, self.in_channels)
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
        self.act2 = _act(self.act_type, self.mid_channels)
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
        self.select = StatsSelect(out_channels, len(dilation), null=null, reduction=reduction)
        # self._initialize_weights()

    def _initialize_weights(self):
        # TODO 初始化权值
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
                # m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        result = self.conv1(self.act1(self.bn1(inputs)))
        result = self.act2(self.bn2(result))
        # result = self.select([conv(result) for conv in self.conv2])
        result = self.select(self.conv2[0](result), self.conv2[1](result))
        result = torch.cat([inputs, result], 1)
        return result

    @property
    def multiply_adds(self):
        madds = self.bn1.multiply_adds
        madds += self.conv1.multiply_adds
        madds = self.bn2.multiply_adds
        madds += sum([conv.multiply_adds for conv in self.conv2])
        return madds

    @property
    def params(self):
        params = self.bn1.params
        params += self.conv1.params
        params += self.bn2.params
        params += sum([conv.params for conv in self.conv2])
        return params


if __name__ == "__main__":
    inputs = torch.ones([2, 128, 8])
    op = DTDNNSS_(in_channels=128, mid_channels=128, out_channels=64, kernel_size=3,
                  dilation=(1, 3), bias=False, in_shape=8, out_shape=8)
    result = op(inputs)

    print(result.shape)
    print(op.multiply_adds)
    print(op.params)
