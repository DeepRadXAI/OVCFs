from torch import nn  # 导入 PyTorch 的神经网络模块
import torch  # 导入 PyTorch
from einops import rearrange  # 从 einops 导入 rearrange 函数，用于重排列张量的维度

# 定义一个带有 LayerNorm 的预处理模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)  # 定义 LayerNorm 层
        self.fn = fn  # 传入的函数，可以是注意力或前馈网络

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)  # 对输入先进行 LayerNorm，再通过传入的函数处理

# 定义一个前馈神经网络（MLP）模块
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # 第一个线性变换，输入维度为 dim，输出维度为 mlp_dim
            nn.SiLU(),  # 使用 SiLU 激活函数
            nn.Dropout(dropout),  # 添加 Dropout 以防止过拟合
            nn.Linear(mlp_dim, dim),  # 第二个线性变换，输入维度为 mlp_dim，输出维度为 dim
            nn.Dropout(dropout)  # 再次添加 Dropout
        )

    def forward(self, x):
        return self.net(x)  # 通过顺序模块处理输入

# 定义一个注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim  # 内部维度为头数乘以每个头的维度
        project_out = not (heads == 1 and head_dim == dim)  # 判断是否需要输出投影

        self.heads = heads  # 注意力头的数量
        self.scale = head_dim ** -0.5  # 缩放因子，用于缩放点积结果

        self.attend = nn.Softmax(dim=-1)  # 使用 Softmax 函数来计算注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 线性变换，生成查询、键和值

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出投影，将内部维度变换回原始维度
            nn.Dropout(dropout)  # 添加 Dropout
        ) if project_out else nn.Identity()  # 如果不需要输出投影，则使用 Identity 模块

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 计算查询、键和值，并沿着最后一个维度分割为三部分
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)  # 重排列查询、键和值的维度
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算点积并进行缩放
        attn = self.attend(dots)  # 计算注意力权重
        out = torch.matmul(attn, v)  # 根据注意力权重加权求和值
        out = rearrange(out, 'b p h n d -> b p n (h d)')  # 重排列输出的维度
        return self.to_out(out)  # 进行输出投影并返回

# 定义一个 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 使用 ModuleList 存储多层 Transformer
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),  # 添加带有 LayerNorm 的注意力层
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))  # 添加带有 LayerNorm 的前馈层
            ]))

    def forward(self, x):
        out = x  # 初始化输出为输入
        for att, ffn in self.layers:  # 遍历每一层
            out = out + att(out)  # 残差连接注意力层的输出
            out = out + ffn(out)  # 残差连接前馈层的输出
        return out  # 返回最终输出

# 定义 MobileViTAttention 模块
class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size  # 设置 patch 的高和宽
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)  # 第一层卷积
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)  # 第二层卷积，用于通道数变换

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)  # Transformer 模块，进行全局特征的提取

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)  # 第三层卷积，用于通道数还原
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)  # 第四层卷积，用于特征融合

    def forward(self, x):
        y = x.clone()  # 复制输入张量，保持原始数据不变

        ## 局部表示 Local Representation
        y = self.conv2(self.conv1(x))  # 通过前两层卷积提取局部特征

        ## 全局表示 Global Representation
        _, _, h, w = y.shape  # 获取特征图的形状
        # 重新排列特征图，将其拆分成多个 patch
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)
        y = self.trans(y)  # 通过 Transformer 处理提取全局特征
        # 将特征图重排列回原来的形状
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph, nw=w // self.pw)

        ## 融合 Fusion
        y = self.conv3(y)  # 通过第三层卷积将通道数还原
        y = torch.cat([x, y], 1)  # 将原始输入和全局特征连接在一起
        y = self.conv4(y)  # 通过第四层卷积融合特征

        return y  # 返回最终输出

# 测试代码块
if __name__ == '__main__':
    m = MobileViTAttention(in_channel=512)  # 创建 MobileViTAttention 实例，输入通道数为 512
    input = torch.randn(1, 512, 49, 49)  # 生成一个随机张量，形状为 (1, 512, 49, 49)
    output = m(input)  # 通过 MobileViTAttention 模块进行前向传播
    print(output.shape)  # 打印输出张量的形状，验证输出是否正确
