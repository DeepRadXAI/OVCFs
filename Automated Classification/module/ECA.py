import torch
from torch import nn

class ECA_layer(nn.Module):
    """构建一个 ECA 模块。

    参数:
        channel: 输入特征图的通道数
        k_size: 自适应选择的一维卷积核大小
    """
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        
        # 全局平均池化层，用于将每个通道的空间信息压缩成一个值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 一维卷积层，用于捕捉通道之间的交互信息
        # 1. 输入通道数为1，因为经过全局平均池化后，每个特征图都变成了1x1
        # 2. 输出通道数为1，因为我们不想改变通道数量，只是调整权重
        # 3. kernel_size=k_size，指定卷积核的大小
        # 4. padding=(k_size - 1) // 2，用于保持卷积后的张量长度与输入一致
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        
        # Sigmoid激活函数，将输出的范围限制在(0, 1)之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数，定义数据流经过该模块的处理步骤。

        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        返回:
        Tensor: 经过ECA模块处理后的输出张量。
        """
        
        # 使用全局平均池化将每个通道的空间维度 (H, W) 压缩到 1x1
        # 输出张量的形状将变为 [batch_size, channels, 1, 1]
        y = self.avg_pool(x)
        
        # 去掉最后一个维度，并交换第二个和第三个维度
        # y.squeeze(-1) 的形状是 [batch_size, channels, 1]
        # y.transpose(-1, -2) 交换后的形状是 [batch_size, 1, channels]
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 通过一维卷积处理，卷积核大小是 k_size
        # 形状保持 [batch_size, 1, channels]，内容经过一维卷积核处理
        y = self.conv(y)
        
        # 再次交换维度，恢复原始的通道顺序
        # y.transpose(-1, -2) 将形状从 [batch_size, 1, channels] 变为 [batch_size, channels, 1]
        y = y.transpose(-1, -2)
        
        # 恢复被去掉的维度，将形状从 [batch_size, channels, 1] 变为 [batch_size, channels, 1, 1]
        y = y.unsqueeze(-1)
        
        # 使用 Sigmoid 激活函数将输出限制在 (0, 1) 之间
        y = self.sigmoid(y)
        
        # 将输入张量 x 与处理后的权重 y 相乘，进行通道加权
        # expand_as 确保 y 的形状与 x 匹配，以便逐元素相乘
        return x * y.expand_as(x)

# 示例用法
if __name__ == "__main__":
    # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
    x = torch.randn(4, 64, 32, 32)
    # 创建一个 ECA 模块实例，通道数为 64
    eca = ECA_layer(channel=64)
    # 通过 ECA 模块调整输入特征
    y = eca(x)
    # 打印输出张量的形状，应该与输入相同
    print(y.shape)  # 输出: torch.Size([4, 64, 32, 32])
