import torch
import torch.nn as nn

# 定义一个 h_sigmoid 类，继承自 nn.Module。
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        # 调用父类的构造函数。
        super(h_sigmoid, self).__init__()
        # 使用 ReLU6 作为激活函数，inplace 参数用于节省内存。
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        # 实现 h-sigmoid 激活函数：f(x) = ReLU6(x + 3) / 6
        return self.relu(x + 3) / 6

# 定义一个 h_swish 类，继承自 nn.Module。
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        # 调用父类的构造函数。
        super(h_swish, self).__init__()
        # 使用 h_sigmoid 作为 h-swish 的一部分。
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        # 实现 h-swish 激活函数：f(x) = x * h_sigmoid(x)
        return x * self.sigmoid(x)

# 定义一个 CoordAtt 类，继承自 nn.Module。
# 该类实现了一种结合坐标信息的注意力机制。
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        # 调用父类的构造函数。
        super(CoordAtt, self).__init__()
        # 使用自适应平均池化提取高度方向和宽度方向的特征。
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向池化
        # 计算通道缩减后的通道数 mip，并保证最小值为 8。
        mip = max(8, inp // reduction)
        # 定义 1x1 卷积层，用于减少通道数。
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # 定义批归一化层，跟随卷积层。
        self.bn1 = nn.BatchNorm2d(mip)
        # 定义 h-swish 激活函数。
        self.act = h_swish()
        # 定义 1x1 卷积层，用于从压缩的通道数恢复到目标输出通道数。
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 高度方向
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 宽度方向

    def forward(self, x):
        # 保存输入的张量，用于与注意力权重相乘。
        identity = x
        # 获取输入张量的尺寸 (batch_size, channels, height, width)。
        n, c, h, w = x.size()
        # 沿高度方向进行自适应平均池化，输出尺寸为 (batch_size, channels, height, 1)。
        x_h = self.pool_h(x)
        # 沿宽度方向进行自适应平均池化，并对宽度和高度进行转置，输出尺寸为 (batch_size, channels, width, 1)。
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # 将两个池化后的张量在宽度方向连接，形成一个新的张量。
        y = torch.cat([x_h, x_w], dim=2)
        # 对连接后的张量进行 1x1 卷积，减少通道数。
        y = self.conv1(y)
        # 进行批归一化。
        y = self.bn1(y)
        # 应用 h-swish 激活函数。
        y = self.act(y)
        # 将卷积后的张量重新按高度和宽度方向切分。
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 对宽度方向的张量进行转置，恢复原来的维度顺序。
        x_w = x_w.permute(0, 1, 3, 2)
        # 使用 sigmoid 函数对高度和宽度方向的张量进行注意力权重计算。
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # 将注意力权重应用到原始输入的张量上，进行逐元素相乘。
        out = identity * a_w * a_h
        # 返回输出张量，大小与输入张量相同。
        return out


# 输入 N C H W, 输出 N C H W
if __name__ == '__main__':
    # 实例化 CoordAtt 模块，输入和输出通道数为 64。
    block = CoordAtt(64, 64)
    # 生成一个随机输入张量，大小为 (1, 64, 64, 64)。
    input = torch.rand(1, 64, 64, 64)
    # 将输入张量传入 CoordAtt 模块，计算输出。
    output = block(input)
    # 打印输入和输出张量的大小。
    print(input.size(), output.size())
