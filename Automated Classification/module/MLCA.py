import math, torch
from torch import nn
import torch.nn.functional as F

# 定义多尺度局部上下文注意力模块 (MLCA)
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()

        # 初始化参数
        # in_size: 输入的通道数
        # local_size: 局部池化尺寸，默认为5
        # gamma 和 b 用于计算卷积核的大小
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        # 根据 ECA 方法计算卷积核大小 k
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1  # 保证 k 是奇数，以便对称填充
        # 定义两个 1D 卷积，用于全局和局部的注意力计算
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # 局部和全局注意力的加权参数
        self.local_weight = local_weight
        # 定义自适应平均池化，用于局部和全局特征提取
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 进行局部和全局的自适应平均池化
        local_arv = self.local_arv_pool(x)  # 局部特征池化
        global_arv = self.global_arv_pool(local_arv)  # 从局部特征中进一步提取全局特征
        # 获取输入和池化后的特征的形状
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
        # 将局部特征重新排列为 (b, 1, local_size*local_size*c) 以便于通过 1D 卷积
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        # 将全局特征重新排列为 (b, 1, c)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
        # 通过局部卷积计算局部注意力
        y_local = self.conv_local(temp_local)
        # 通过全局卷积计算全局注意力
        y_global = self.conv(temp_global)
        # 将局部注意力重新排列回原始形状 (b, c, local_size, local_size)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        # 将全局注意力重新排列回 (b, c, 1, 1)
        y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)
        # 应用 sigmoid 激活函数，将注意力权重映射到 (0, 1) 区间
        att_local = y_local_transpose.sigmoid()
        # 将全局注意力池化到局部特征的大小
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        # 根据局部和全局的加权参数，融合两种注意力，调整到输入的空间维度
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
        # 将输入特征与注意力权重相乘，得到加权后的输出
        x = x * att_all
        return x

# 测试代码块
if __name__ == '__main__':
    # 创建 MLCA 模块实例，输入通道数为 256
    attention = MLCA(in_size=256)
    # 随机生成输入张量，形状为 (2, 256, 16, 16)
    inputs = torch.randn((2, 256, 16, 16))
    # 将输入张量传入 MLCA 模块，计算输出
    result = attention(inputs)
    # 打印输出张量的形状
    print(result.size())
