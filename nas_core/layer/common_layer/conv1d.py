import torch.nn as nn

from nas_core.global_enum import Activation, Padding
from nas_core.layer.utils import _padding, _act


class Conv1d_(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=Padding.SAME,
                 dilation=1,
                 groups=1,
                 bias=True,
                 act=Activation.ReLU,
                 is_folded=True,
                 in_shape=None,
                 out_shape=None
                 ):
        super(Conv1d_, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.act_type = act
        self.is_folded = is_folded
        self.padding = _padding(self.kernel_size, self.padding_type, self.dilation)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias
        )
        self.act = _act(act_type=self.act_type)

    def forward(self, inputs):
        result_conv = self.conv(inputs)
        result = self.act(result_conv)
        return result

    @property
    def multiply_adds(self):
        if self.out_shape is None:
            raise Exception('Conv1d: You must init out_shape to tell the Conv1d\' output shape!')
        grouped_in_channel = self.in_channels // self.groups
        grouped_out_channel = self.out_channels // self.groups
        each_group_multiply_adds = self.out_shape * self.kernel_size * grouped_in_channel * grouped_out_channel
        multiply_adds = each_group_multiply_adds * self.groups
        return multiply_adds

    @property
    def params(self):
        grouped_in_channel = self.in_channels // self.groups
        grouped_out_channel = self.out_channels // self.groups
        each_group_params = self.kernel_size * grouped_in_channel * grouped_out_channel
        params = each_group_params * self.groups
        # TODO 不考虑fold方式，需要计算bias的参数量
        if self.bias is True and self.is_folded is False:
            params += self.out_channels
        return params


if __name__ == "__main__":
    import torch

    op = Conv1d_(in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 dilation=2,
                 bias=True,
                 is_folded=False,
                 in_shape=8,
                 out_shape=8)
    inputs = torch.ones([1, 1, 6])
    op.conv.weight.data.fill_(1)
    result = op(inputs)

    print(result)
    print(op.multiply_adds)
    print(op.params)
