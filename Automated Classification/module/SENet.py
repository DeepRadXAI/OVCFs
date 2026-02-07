import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化SELayer类。
        参数:
        channel (int): 输入特征图的通道数。
        reduction (int): 用于减少通道数的缩减率，默认为16。它用于在全连接层中压缩特征的维度。
        """
        super(SELayer, self).__init__()
        # 自适应平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层序列，包含两个线性变换和中间的ReLU激活函数
        self.fc = nn.Sequential(
            # 第一个线性层，将通道数从 'channel' 缩减到 'channel // reduction'
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLU激活函数，用于引入非线性
            nn.ReLU(inplace=True),
            # 第二个线性层，将通道数从 'channel // reduction' 恢复到 'channel'
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid激活函数，将输出限制在(0, 1)之间
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播函数。
        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channel, height, width)。
        返回:
        Tensor: 经过通道注意力调整后的输出张量，形状与输入相同。
        """
        # 获取输入张量的形状
        b, c, h, w = x.size()
        # Squeeze：通过全局平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        y = self.avg_pool(x).view(b, c)
        # Excitation：通过全连接层序列，对压缩后的特征进行处理
        y = self.fc(y).view(b, c, 1, 1)
        # 通过扩展后的注意力权重 y 调整输入张量 x 的每个通道
        return x * y.expand_as(x)
        
# 示例用法
if __name__ == "__main__":
    # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
    x = torch.randn(4, 64, 32, 32)
    # 创建一个SELayer实例，通道数为64
    se_layer = SELayer(channel=64)
    # 通过SELayer调整输入特征
    y = se_layer(x)
    # 打印输出张量的形状，应该与输入相同
    print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])
