import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import auto_fp16
from mmdet.models.registry import NECKS
from sepc.self_mmdet.ops.dcn.sepc_dconv import sepc_conv
from torch.nn import init as init

# PS: Pconv是在FPN后接入的哦


# 这些在声明的时候会注入到mmdetection的Registry中，所以放在哪里声明都是可以的，只要能引入对应的Registry
@NECKS.register_module
class SEPC(nn.Module):

    def __init__(self,
                 in_channels=[256] * 5,
                 out_channels=256,
                 num_outs=5,
                 pconv_deform=False,
                 lcconv_deform=False,
                 iBN=False,
                 Pconv_num=4,

                 ):
        """
        SEPC模块的初始化函数
        :param in_channels: type->list: 输入进来的每个特征层次的通道数
        :param out_channels: type->int: 特征层次输出的通道数
        :param num_outs: type->int: 输出的特征图的数目
        :param pconv_deform: type->bool: 在 pconv 中是否使用 deformable conv
        :param lcconv_deform: type->bool: 在分类和方框回归网络中是否使用 deformable conv
        :param iBN: type->func: integrated BN函数. (对应于论文3.2部分，会对多个level进行共享的BN)
        :param Pconv_num: type->int: Pconv 的数量
        """
        super(SEPC, self).__init__()
        # 注册类内变量
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5
        self.fp16_enabled = False
        self.iBN = iBN
        self.Pconvs = nn.ModuleList()

        # 添加Pconv模块
        for i in range(Pconv_num):
            self.Pconvs.append(PConvModule(in_channels[i], out_channels, iBN=self.iBN, part_deform=pconv_deform))

        # Extra head部分的注册
        self.lconv = sepc_conv(256, 256, kernel_size=3, dilation=1, part_deform=lcconv_deform)
        self.cconv = sepc_conv(256, 256, kernel_size=3, dilation=1, part_deform=lcconv_deform)
        self.relu = nn.ReLU()
        # BN层的注册，但是其是会在后续共享的
        if self.iBN:
            self.lbn = nn.BatchNorm2d(256)
            self.cbn = nn.BatchNorm2d(256)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        # 初始化 Extra head 部分的权重，Pconv部分的权重会交给 PconvModule自己去进行初始化
        for str in ["l", "c"]:
            m = getattr(self, str + "conv")
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x = inputs
        # pconv处理
        for pconv in self.Pconvs:
            x = pconv(x)
        # extra head 处理
        cls = [self.cconv(level, item) for level, item in enumerate(x)]
        loc = [self.lconv(level, item) for level, item in enumerate(x)]
        # BN处理 (使用 iBN 模块进行 BN 操作处理)
        if self.iBN:
            cls = iBN(cls, self.cbn)
            loc = iBN(loc, self.lbn)
        # 进行激活函数，获取输出(将作为交给最后的 cls_head 和 bbox_head 的 input)
        outs = [[self.relu(s), self.relu(l)] for s, l in zip(cls, loc)]
        # 返回输出
        return tuple(outs)


class PConvModule(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 kernel_size=[3, 3, 3],
                 dilation=[1, 1, 1],
                 groups=[1, 1, 1],
                 iBN=False,
                 part_deform=False,

                 ):
        """
        Pconv 模块的实现
        :param in_channels: type->int: 输入的特征的通道数
        :param out_channels: type->int: 输出的特征通道数
        :param kernel_size: type->list: 卷积核的大小
        :param dilation: type->list: 卷积核空洞的大小
        :param groups: type->list: 组数
        :param iBN: type->func: integrated BN函数
        :param part_deform: type->bool: 是否使用 SEPC (在pconv中使用deformable)
        """
        super(PConvModule, self).__init__()

        #     assert not (bias and iBN)
        self.iBN = iBN
        self.Pconv = nn.ModuleList()
        # 构建 pconv 模块
        self.Pconv.append(
            sepc_conv(in_channels, out_channels, kernel_size=kernel_size[0], dilation=dilation[0], groups=groups[0],
                      padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2, part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels, out_channels, kernel_size=kernel_size[1], dilation=dilation[1], groups=groups[1],
                      padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2, part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels, out_channels, kernel_size=kernel_size[2], dilation=dilation[2], groups=groups[2],
                      padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2, stride=2, part_deform=part_deform))

        if self.iBN:
            self.bn = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.init_weights()

    # 初始化 pconv 的权重
    def init_weights(self):
        for m in self.Pconv:
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # 按照标准配置来说，x输入进来是 5 个 level 的
        next_x = []
        # level是当前要进行卷积的三个层次(实验默认是三个层次)的最低层次编号，feature是三个层次中的最低层次的特征(PS: 其实并没有必要要将
        # 这个东西拿出来... 有 level 和 x 就够了)
        # 下面循环就是将每3个(首尾是两个)通过 conv 或者 deformable conv 进行特征处理然后相加来得到多个 level 最终输出的特征信息
        for level, feature in enumerate(x):

            temp_fea = self.Pconv[1](level, feature)
            if level > 0:
                temp_fea += self.Pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.upsample_bilinear(self.Pconv[0](level, x[level + 1]),
                                                size=[temp_fea.size(2), temp_fea.size(3)])
            next_x.append(temp_fea)
        # 如果选择进行 integrated BN 那么进行处理
        if self.iBN:
            next_x = iBN(next_x, self.bn)
        # 进行激活函数的处理
        next_x = [self.relu(item) for item in next_x]
        return next_x


def iBN(fms, bn):
    """
    integrated batch normalization函数处理方法
    :param fms: 输入进入要进行 iBN 的数据
    :param bn: 进行 iBN 的 BN 层
    :return:
    """
    # 首先获取输入进来的每个 level 的向量的平面空间大小(长宽)
    sizes = [p.shape[2:] for p in fms]
    # 获取 mini-batch size 和 channel size
    n, c = fms[0].shape[0], fms[0].shape[1]
    # 将所有特征层次上的特征展开为 mini-batch size * channel size * 1 * (WH) 的向量，并在最后一维上进行拼接
    #     即形成 mini-batch size * channel size * 1 * (sum(WH for W, H in sizes))
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    # 对上面的内容进行 BN 操作
    # 对 2d-BN 来说无论是对平面上进行 BN 还是展开之后进行 BN 都是效果上相同的，没有其他影响的，具体说明见论文笔记后面的补充
    fm = bn(fm)
    # 将 BN 处理后的特征向量重新按照不同 level 上的大小进行分割
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    # 复原为原本的形状返回
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]
