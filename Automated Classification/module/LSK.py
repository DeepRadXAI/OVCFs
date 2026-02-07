import torch
import torch.nn as nn

class LSKNet(nn.Module):
    def __init__(self, dim):
        """
        LSKblock 初始化函数
        Args:
            dim (int): 输入通道的数量。
        """
        super().__init__()
        # 第一个卷积层，使用 5x5 的卷积核，通道数与输入一致，采用深度可分离卷积（groups=dim）。
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 第二个卷积层，使用 7x7 的卷积核，采用深度可分离卷积，并设置扩展卷积（dilation=3）和较大的填充（padding=9）来扩大感受野。
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 第三个卷积层，使用 1x1 的卷积核，将通道数减少一半（dim -> dim//2）。
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        # 第四个卷积层，使用 1x1 的卷积核，将通道数减少一半（dim -> dim//2）。
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 用于通道间注意力机制的卷积层，接收 2 个通道的数据并输出 2 个通道，卷积核大小为 7x7。
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 最后一个卷积层，使用 1x1 的卷积核，将通道数恢复到原始输入的维度（dim//2 -> dim）。
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        Returns:
            Tensor: 输出张量，形状与输入相同。
        """
        # 通过第一个卷积层，获取局部特征（5x5 卷积）。
        attn1 = self.conv0(x)
        # 通过第二个卷积层，获取更大范围的空间特征（7x7 扩展卷积）。
        attn2 = self.conv_spatial(attn1)
        # 通过第三个卷积层，减少通道数（1x1 卷积，dim -> dim//2）。
        attn1 = self.conv1(attn1)
        # 通过第四个卷积层，减少通道数（1x1 卷积，dim -> dim//2）。
        attn2 = self.conv2(attn2)
        # 将两个不同感受野的特征在通道维度上进行拼接，形成一个新的张量。
        attn = torch.cat([attn1, attn2], dim=1)
        # 对拼接后的特征图计算通道维度的平均值特征（avg_attn）。
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # 对拼接后的特征图计算通道维度的最大值特征（max_attn）。
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 将平均值特征和最大值特征在通道维度上进行拼接。
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # 通过注意力机制卷积层，生成两个通道的注意力权重。
        sig = self.conv_squeeze(agg).sigmoid()
        # 将两个不同感受野的特征分别乘以相应的注意力权重。
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 通过最后一个卷积层，将通道数恢复到原始输入的维度。
        attn = self.conv(attn)
        # 将原始输入与注意力加权后的特征相乘，得到增强后的输出。
        return x * attn

# 示例运行，输入张量形状为 (1, 64, 64, 64)，输出形状与输入相同。
if __name__ == '__main__':
    block = LSKNet(64).cuda()  # 初始化 LSKblock 模块，并将其移动到 GPU 上。
    input = torch.rand(1, 64, 64, 64).cuda()  # 创建一个随机输入张量，并移动到 GPU 上。
    output = block(input)  # 通过 LSKblock 模块进行前向传播。
    print(input.size(), output.size())  # 打印输入和输出的张量形状，确认它们相同。
