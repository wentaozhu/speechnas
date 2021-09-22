import torch
import torch.nn as nn
import torch.nn.functional as F

from nas_core.global_enum import Activation, Padding
from nas_core.layer.common_layer.batchnorm1d import BatchNorm1d_
from nas_core.layer.common_layer.conv1d import Conv1d_
from nas_core.layer.common_layer.linear import Linear_
from nas_core.layer.utils import _act


class TransitLayer(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 bias=False,
                 is_folded=False,
                 act=Activation.PReLU,
                 in_shape=None,
                 out_shape=None):
        super(TransitLayer, self).__init__()
        self.in_channels = in_channels
        self.bn = BatchNorm1d_(num_features=in_channels,
                               momentum=0.1,
                               affine=True,
                               act=Activation.Identity,
                               is_folded=is_folded)
        self.act = _act(act, num_parameters=in_channels)
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
        params += self.in_channels
        return params


class DenseLayer(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 bias=True,
                 is_folded=False,
                 act=Activation.Identity,
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
        # self.act = _act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    @property
    def multiply_adds(self):
        madds = self.conv.multiply_adds
        madds += self.bn.multiply_adds
        return madds

    @property
    def params(self):
        params = self.conv.params
        params += self.bn.params
        return params


class HighOrderStatsPool(nn.Module):

    def forward(self, x):
        return self._high_order_statistics_pooling(x)

    def _high_order_statistics_pooling(self, x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
        mean = x.mean(dim=dim)
        std = x.std(dim=dim, unbiased=unbiased)
        norm = (x - mean.unsqueeze(dim=dim)) / std.clamp(min=eps).unsqueeze(dim=dim)
        skewness = norm.pow(3).mean(dim=dim)
        kurtosis = norm.pow(4).mean(dim=dim)
        stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
        if keepdim:
            stats = stats.unsqueeze(dim=dim)
        return stats


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

    def forward(self, x_a, x_b):  # TODO no change for now
        x = [x_a, x_b]
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


class DTDNNSS_V1_(nn.Module):
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
        super(DTDNNSS_V1_, self).__init__()
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
        result = self.select(self.conv2[0](result), self.conv2[1](result))
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


if __name__ == "__main__":
    inputs = torch.ones([2, 768, 200])
    op = DTDNNSS_V1_(in_channels=768, mid_channels=128, reduction_channels=32, out_channels=64, kernel_size=3,
                  dilation=(1, 3), bias=False, in_shape=200, out_shape=200)
    result = op(inputs)

    print(result.shape)
    print(op.multiply_adds)
    print(op.params)
