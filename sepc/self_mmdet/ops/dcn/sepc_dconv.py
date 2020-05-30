import torch
import torch.nn as nn
from mmdet.ops.dcn import DeformConv
from mmdet.ops.dcn import deform_conv
from torch.nn.modules.utils import _pair


class sepc_conv(DeformConv):
    def __init__(self, *args, part_deform=False, **kwargs):
        # 会用父类 DeformConv 的初始化函数进行初始化
        """
        这里列举一下传递进来的参数吧:
        :param in_channels: type->int: conv的输入通道数
        :param out_channels: type->int: conv的输出通道数
        :param kernel_size: type->int: conv的核的大小尺寸
        :param dilation: type->int: 空洞大小
        :param groups: type->int: 组卷积组数
        :param padding: type->int: conv进行padding的大小
        PS: 上面的全都是 nn.Conv2d 需要的基本参数
        :param part_deform: type->bool: 是否使用 deformable conv
        """
        super(sepc_conv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        # 如果pconv选择进行deformable conv的时候会初始化一个 Conv2d -> 是那个卷积计算 offset 的 conv
        # outchannel是 deformable_groups * 2 * kernel_size * kernel_size PS: deformable_groups是deformable预测offset时候的
        # 分组, 但是预测offset时候你分组了，就会产生多组offset，也就要给多组conv提供offset啊, 所以 groups 和 deformable_groups要
        # 保持一致?
        if self.part_deform:
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deformable_groups * 2 * self.kernel_size[0] *
                self.kernel_size[1],
                kernel_size=self.kernel_size,
                stride=_pair(self.stride),
                padding=_pair(self.padding),
                bias=True)
            self.init_offset()
        # 初始化 bias 的参数为 0
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):
        # 预测 offset 的 conv 的 weights 和 bias 都要初始化为 0
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        # i 是输入的层次, start_level 是开始使用 deformable conv 的层次，论文中最底层是不使用 deformable conv
        # 一是认为不需要对齐，二是会节省很大的计算消耗，所以如果是低层或者就设置为不使用 sepc 只是普通的 pconv 情形下返回普通的 conv2d
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                              dilation=self.dilation, groups=self.groups)
        # 获取 offset
        offset = self.conv_offset(x)
        # 使用 deform_conv 利用 offset 进行卷积
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups) + self.bias.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)
        # PS: 无论是普通的 conv2d 还是 deformable conv 都用 self.weight 初始化，初始化方法是 stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv) ，但是在 sepc 中会重新进行初始化... 为啥不一步到位...
