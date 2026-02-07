import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random

class NpyDataLoad(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path 
        self.transform = transform

        random.seed(0)  # 保证随机结果可复现
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

        # 遍历文件夹，一个文件夹对应一个类别
        data_classes = [cla for cla in os.listdir(os.path.join(data_path))] 
        self.num_class = len(data_classes)
        print(self.num_class)
        # 排序，保证顺序一致
        data_classes.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((cla, idx) for idx, cla in enumerate(data_classes))

        self.data_paths = []  # 存储训练集的所有.npy文件路径
        self.data_labels = []  # 存储训练集.npy文件对应的索引信息 
        self.data_nums = []  # 存储每个类别的样本总数
        supported = [".npy"]  # 支持的文件后缀类型
        # 遍历每个文件夹下的文件
        for cla in data_classes:
            # print(cla)
            # if cla=="cqNHE2":
            #     data_path=self.data_path
            # print("datapath",data_path)
            # data_path="/home/un/桌面/QC/图像分类/2d图像分类/data4/"
            cla_path1 = os.path.join(self.data_path, cla)
            print("cla_path",cla_path1)
            # 遍历获取支持的所有.npy文件路径
            data_files = [os.path.join(self.data_path, cla, i) for i in os.listdir(cla_path1) if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            data_class = class_indices[cla]
            # 记录该类别的样本数量
            self.data_nums.append(len(data_files))
            # 写入列表
            for data_path in data_files:
                self.data_paths.append(data_path)
                self.data_labels.append(data_class)

        print("{} samples were found in the dataset.".format(sum(self.data_nums)))

    def __len__(self):
        return sum(self.data_nums)
    
    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        label = self.data_labels[idx]
        # print(label)
        if self.transform is None:
            data.resize((3,128,128))
            # print(data.shape)
            # print(type(data))
            # data = 
        # else:
        #     raise ValueError('Data is not preprocessed')

        return torch.tensor(data, dtype=torch.float32), label
    
    @staticmethod
    def collate_fn(batch):
        data, labels = tuple(zip(*batch))
        data = torch.stack(data, dim=0) 
        labels = torch.as_tensor(labels)  
        return data, labels
