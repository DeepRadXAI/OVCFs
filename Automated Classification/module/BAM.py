import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，用于深度学习和张量操作
from torch import nn  # 从 PyTorch 中导入神经网络模块
from torch.nn import init  # 从 PyTorch 中导入初始化模块，用于权重初始化

# 自动填充函数，用于根据内核大小、填充和扩张比率自动计算填充值
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        # 如果扩张比率大于1，根据扩张比率调整内核大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 如果没有指定填充值，自动计算填充值为内核大小的一半
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# 定义一个展平层，将输入的多维张量展平为二维张量
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# 定义通道注意力模块 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        # 自适应平均池化，将输入特征图的空间维度缩小到 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 定义全连接层的通道数
        gate_channels = [channel]  # 输入通道数
        gate_channels += [channel // reduction] * num_layers  # 缩减后的通道数
        gate_channels += [channel]  # 恢复到原始通道数

        # 使用 nn.Sequential 定义通道注意力模块的层次
        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())  # 展平层
        for i in range(len(gate_channels) - 2):
            # 添加全连接层、批量归一化层和激活函数
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        # 添加最后的全连接层，将通道恢复到原始大小
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, channels, height, width)
        # 先通过自适应平均池化，输出形状为 (batch_size, channels, 1, 1)
        res = self.avgpool(x)
        # 通过通道注意力网络，输出形状为 (batch_size, channels)
        res = self.ca(res)
        # 调整维度以匹配输入特征图的形状，并在空间维度上扩展
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res

# 定义空间注意力模块 (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        # 使用 nn.Sequential 定义空间注意力模块的层次
        self.sa = nn.Sequential()
        # 第一个 1x1 卷积层用于减少通道数
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            # 3x3 卷积层，使用扩张卷积，并附加批量归一化和激活函数
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, 
                                                        padding=autopad(3, None, dia_val), dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        # 最后的 1x1 卷积层将输出通道数减少为 1
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, channels, height, width)
        # 通过空间注意力网络，输出形状为 (batch_size, 1, height, width)
        res = self.sa(x)
        # 扩展为与输入相同的形状
        res = res.expand_as(x)
        return res

# 定义 BAMBlock 模块，结合通道和空间注意力
class BAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        # 初始化通道注意力和空间注意力模块
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数来计算注意力权重

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积层使用 He 初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为 0
            elif isinstance(m, nn.BatchNorm2d):  # 对批量归一化层初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对全连接层使用正态分布初始化
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, channels, height, width)
        b, c, _, _ = x.size()
        # 计算空间注意力输出
        sa_out = self.sa(x)
        # 计算通道注意力输出
        ca_out = self.ca(x)
        # 将空间和通道注意力相加并通过 Sigmoid 激活函数计算注意力权重
        weight = self.sigmoid(sa_out + ca_out)
        # 结合输入和注意力权重
        out = (1 + weight) * x
        return out

# 测试代码块
if __name__ == '__main__':
    # 创建 BAMBlock 实例，输入通道数为 512
    input = torch.randn(32, 512, 7, 7)  # 随机生成输入张量，形状为 (32, 512, 7, 7)
    bam = BAMBlock(channel=512, reduction=16, dia_val=2)
    output = bam(input)
    # 打印输出张量的形状，验证输出是否正确
    print(output.shape)
