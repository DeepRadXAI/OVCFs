import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        """
        SKAttention 初始化函数
        Args:
            channel (int): 输入和输出通道数。
            kernels (list): 多尺度卷积核的大小列表。
            reduction (int): 通道数缩减的比例因子。
            group (int): 深度卷积的组数。
            L (int): 计算降维的最小通道数。
        """
        super().__init__()

        # 计算缩减后的通道数，保证其不小于 L。
        self.d = max(L, channel // reduction)
        
        # 初始化多个卷积操作，每个卷积操作的卷积核大小由 kernels 列表决定。
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),  # 深度卷积
                    ('bn', nn.BatchNorm2d(channel)),  # 批归一化
                    ('relu', nn.ReLU())  # ReLU 激活函数
                ]))
            )

        # 线性层，用于将通道数降维为 d。
        self.fc = nn.Linear(channel, self.d)

        # 初始化多个线性层，用于将降维后的特征映射回原通道数。
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))

        # softmax 层，用于计算不同尺度特征的注意力权重。
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        bs, c, _, _ = x.size()  # 获取输入张量的形状信息
        conv_outs = []

        ### 多尺度特征提取
        for conv in self.convs:
            conv_outs.append(conv(x))  # 使用不同的卷积核对输入进行卷积操作
        feats = torch.stack(conv_outs, 0)  # 将不同卷积核的输出在第一个维度上堆叠，形状为 (k, bs, channel, h, w)

        ### 特征融合
        U = sum(conv_outs)  # 将所有尺度的特征进行相加，形状为 (bs, c, h, w)

        ### 通道数缩减
        S = U.mean(-1).mean(-1)  # 对空间维度进行平均，得到形状为 (bs, c) 的张量
        Z = self.fc(S)  # 通过全连接层进行通道数缩减，得到形状为 (bs, d) 的张量

        ### 计算注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)  # 通过线性层将降维后的特征映射回原通道数，形状为 (bs, c)
            weights.append(weight.view(bs, c, 1, 1))  # 调整形状为 (bs, channel, 1, 1)
        attention_weights = torch.stack(weights, 0)  # 将所有的注意力权重在第一个维度上堆叠，形状为 (k, bs, channel, 1, 1)
        attention_weights = self.softmax(attention_weights)  # 使用 softmax 进行归一化，得到最终的注意力权重

        ### 加权融合特征
        V = (attention_weights * feats).sum(0)  # 将注意力权重与对应的多尺度特征相乘并相加，得到最终的加权特征
        return V

# 示例运行
if __name__ == '__main__':
    input = torch.randn(1, 64, 64, 64)  # 创建一个随机输入张量，形状为 (1, 64, 64, 64)
    sk_attention = SKAttention(channel=64, reduction=8)  # 初始化 SKAttention 模块
    output = sk_attention(input)  # 通过 SKAttention 模块进行前向传播
    print(output.shape)  # 打印输出张量的形状，确认其与输入形状相同
