import torch  # 导入 PyTorch
import torch.nn as nn  # 从 PyTorch 导入神经网络模块

# 定义 SimAM 模块类
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()  # 使用 Sigmoid 激活函数
        self.e_lambda = e_lambda  # 设置正则化项的系数

    # 重写 __repr__ 方法，提供模块的可读表示
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)  # 显示 e_lambda 的值
        return s

    # 静态方法，返回模块的名称
    @staticmethod
    def get_module_name():
        return "simam"

    # 定义前向传播方法
    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入张量的形状

        n = w * h - 1  # 计算每个通道的像素数减一，用于归一化

        # 计算每个像素点减去均值的平方差
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算 y，公式为 x 减去均值的平方差，除以方差的 4 倍加上正则项，然后加上 0.5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 使用 Sigmoid 函数激活 y，并与输入 x 相乘
        return x * self.activation(y)

# 测试代码块
if __name__ == '__main__':
    input = torch.randn(3, 64, 7, 7)  # 生成一个随机输入张量，形状为 (3, 64, 7, 7)
    model = SimAM()  # 创建 SimAM 模块实例
    outputs = model(input)  # 通过 SimAM 模块进行前向传播
    print(outputs.shape)  # 打印输出张量的形状，验证输出是否正确
