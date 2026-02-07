import torch
import numpy as np
from typing import List, Tuple, Union

def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def kan_conv2d(matrix: Union[List[List[float]], np.ndarray], #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernel, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    matrix_out = torch.zeros((batch_size,n_channels,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)


    for channel in range(n_channels):
        #print(matrix[:,channel,:,:].unsqueeze(1).shape)
        conv_groups = unfold(matrix[:,channel,:,:].unsqueeze(1)).transpose(1, 2)
        #print("conv",conv_groups.shape)
        for k in range(batch_size):
            matrix_out[k,channel,:,:] = kernel.forward(conv_groups[k,:,:]).reshape((h_out,w_out))
    return matrix_out

def multiple_convs_kan_conv2d(matrix, #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernels, 
             kernel_side,
             stride= (1, 1), 
             dilation= (1, 1), 
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    h_out, w_out,batch_size,n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    matrix_out = torch.zeros((batch_size,n_channels*n_convs,h_out,w_out)).to(device)#estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side,kernel_side), dilation=dilation, padding=padding, stride=stride)
    conv_groups = unfold(matrix[:,:,:,:]).view(batch_size, n_channels,  kernel_side*kernel_side, h_out*w_out).transpose(2, 3)#reshape((batch_size,n_channels,h_out,w_out))
    for channel in range(n_channels):
        for kern in range(n_convs):
            matrix_out[:,kern  + channel*n_convs,:,:] = kernels[kern].conv.forward(conv_groups[:,channel,:,:].flatten(0,1)).reshape((batch_size,h_out,w_out))
    return matrix_out


def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix. 

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix

import torch
import math
from KAN import KANLinear

#Script que contiene la implementación del kernel con funciones de activación.
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            n_convs: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu"
        ):
        """
        KAN 卷积层，支持多个卷积操作

        参数:
            n_convs (int): 卷积层的数量
            kernel_size (tuple): 卷积核的大小
            stride (tuple): 卷积操作的步幅
            padding (tuple): 卷积操作的填充
            dilation (tuple): 卷积核元素之间的间距
            grid_size (int): 网格的大小
            spline_order (int): 样条的阶数
            scale_noise (float): 噪声的比例
            scale_base (float): 基础尺度
            scale_spline (float): 样条的尺度
            base_activation (torch.nn.Module): 基础激活函数
            grid_eps (float): 网格的 epsilon 值
            grid_range (tuple): 网格的范围
            device (str): 使用的设备
        """


        super(KAN_Convolutional_Layer, self).__init__()  # 调用父类的初始化方法
        self.grid_size = grid_size # 网格大小，用于定义卷积操作的网格
        self.spline_order = spline_order # 样条顺序，用于定义卷积核的阶数
        self.kernel_size = kernel_size  # 卷积核的大小
        self.device = device # 设备（如 'cuda' 或 'cpu'），用于模型计算
        self.dilation = dilation  # 膨胀系数，用于膨胀卷积操作
        self.padding = padding # 填充大小，用于卷积操作的边界填充
        self.convs = torch.nn.ModuleList() # 卷积层列表，用于存储多个卷积操作
        self.n_convs = n_convs  # 卷积层的数量
        self.stride = stride # 步幅，用于卷积操作的步长


        # 创建 n_convs 个 KAN_Convolution 对象，并将它们添加到卷积层列表中
        for _ in range(n_convs):
            self.convs.append(
                KAN_Convolution(
                    kernel_size= kernel_size,  # 卷积核的大小
                    stride = stride,  # 卷积操作的步幅
                    padding=padding,  # 卷积操作的边界填充
                    dilation = dilation,  # 卷积操作的膨胀系数
                    grid_size=grid_size, # 用于定义卷积操作的网格大小
                    spline_order=spline_order,  
                    scale_noise=scale_noise,  # 噪声尺度
                    scale_base=scale_base, # 基础尺度
                    scale_spline=scale_spline, # 样条尺度
                    base_activation=base_activation, # 基础激活函数
                    grid_eps=grid_eps,  # 网格的误差
                    grid_range=grid_range,  # 网格范围
                    device = device  # 用于模型计算的设备（如 'cuda' 或 'cpu'）
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        # 如果有多个卷积层，应用所有卷积层
        if self.n_convs>1:
            # 使用 multiple_convs_kan_conv2d 函数进行多个卷积操作
            return multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)
         # 如果只有一个卷积层，应用单个卷积层 
        return self.convs[0].forward(x)  # 使用第一个（也是唯一一个）卷积层进行卷积操作
        

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device = "cpu"
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor, update_grid=False):
        return kan_conv2d(x, self.conv,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)



