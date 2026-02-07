import torch  # 导入 PyTorch
from torch import nn  # 从 PyTorch 导入神经网络模块

# 定义 ContextBlock 类，继承自 nn.Module
class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        
        # 验证 fusion_types 是否有效
        valid_fusion_types = ['channel_add', 'channel_mul']

        # 检查 pooling_type 是否在有效选项 ['avg', 'att'] 中
        assert pooling_type in ['avg', 'att']
        # 确认 fusion_types 是列表或元组
        assert isinstance(fusion_types, (list, tuple))
        # 确认 fusion_types 中的所有元素都在 valid_fusion_types 中
        assert all([f in valid_fusion_types for f in fusion_types])
        # 确保至少有一个 fusion 类型被指定
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes  # 输入通道数
        self.ratio = ratio  # 缩减比例
        self.planes = int(inplanes * ratio)  # 缩减后的通道数
        self.pooling_type = pooling_type  # 池化类型（'avg' 或 'att'）
        self.fusion_types = fusion_types  # 融合类型

        # 如果池化类型为 'att'，定义注意力池化的卷积层和 softmax
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)  # 在维度 2 上做 softmax
        else:
            # 如果池化类型为 'avg'，定义自适应平均池化层
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 定义 'channel_add' 融合类型的卷积层序列
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),  # 1x1 卷积
                nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
                nn.ReLU(inplace=True),  # 激活函数 ReLU
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)  # 1x1 卷积，恢复到原通道数
            )
        else:
            self.channel_add_conv = None

        # 定义 'channel_mul' 融合类型的卷积层序列
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),  # 1x1 卷积
                nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
                nn.ReLU(inplace=True),  # 激活函数 ReLU
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)  # 1x1 卷积，恢复到原通道数
            )
        else:
            self.channel_mul_conv = None

    # 定义空间池化方法
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()  # 获取输入张量的形状
        if self.pooling_type == 'att':  # 如果池化类型为 'att'
            input_x = x.view(batch, channel, height * width)  # 展平 H 和 W
            input_x = input_x.unsqueeze(1)  # 增加一个维度
            context_mask = self.conv_mask(x)  # 应用 1x1 卷积层
            context_mask = context_mask.view(batch, 1, height * width)  # 展平 H 和 W
            context_mask = self.softmax(context_mask)  # 在 H * W 上应用 softmax
            context_mask = context_mask.unsqueeze(-1)  # 增加一个维度
            context = torch.matmul(input_x, context_mask)  # 计算加权和
            context = context.view(batch, channel, 1, 1)  # 恢复形状
        else:
            context = self.avg_pool(x)  # 如果池化类型为 'avg'，直接应用平均池化

        return context  # 返回上下文张量

    # 定义前向传播方法
    def forward(self, x):
        context = self.spatial_pool(x)  # 获取上下文信息
        out = x  # 初始化输出
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))  # 应用通道乘法融合
            out = out * channel_mul_term  # 输出乘以融合结果

        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)  # 应用通道加法融合
            out = out + channel_add_term  # 输出加上融合结果

        return out  # 返回最终输出

# 测试代码块
if __name__ == "__main__":
    in_tensor = torch.ones((1, 64, 128, 128))  # 创建一个全1的输入张量，形状为 (1, 64, 128, 128)
    cb = ContextBlock(inplanes=64, ratio=0.25, pooling_type='att')  # 创建 ContextBlock 实例
    out_tensor = cb(in_tensor)  # 传递输入张量进行前向传播
    print(in_tensor.shape)  # 打印输入张量的形状
    print(out_tensor.shape)  # 打印输出张量的形状
