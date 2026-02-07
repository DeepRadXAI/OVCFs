############################################################################################################
# 增添了以下功能：
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine warm up：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。 
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型 
# 5. 使用amp包实现半精度训练，在保证准确率的同时尽可能的减小训练成本
# 6. 实现了数据加载类的自定义实现
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir= log_path"来启动，
#    结果通过网页 http://localhost:6006/ 查看可视化结果
############################################################################################################
# --model 可选的超参如下：
# alexnet   zfnet   vgg   vgg_tiny   vgg_small   vgg_big   googlenet   xception   resnet_small   resnet   resnet_big   resnext   resnext_big  
# densenet_tiny   densenet_small   densenet   densenet_big   mobilenet_v3   mobilenet_v3_large   shufflenet_small   shufflenet
# efficient_v2_small   efficient_v2   efficient_v2_large   convnext_tiny   convnext_small   convnext   convnext_big   convnext_huge
# vision_transformer_small   vision_transformer   vision_transformer_big   swin_transformer_tiny   swin_transformer_small   swin_transformer 
#
# 训练命令示例： 
#   python train_vit_weight.py --model vision_transformer1 --num_classes 2
############################################################################################################

import os 
import argparse 
import math
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler 

import classic_models 
from utils.lr_methods import warmup 
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate 


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')
parser.add_argument('--epochs', type=int, default=100, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate') 
parser.add_argument('--seed', default=21, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--use_amp', default=False, action='store_true', help=' training with mixed precision') 
parser.add_argument('--data_path', type=str, default=r"data/OVCFs_JPG")
parser.add_argument('--model', type=str, default="vision_transformer1", help=' select a model for training') 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument(
    '--weights',
    type=str,
    default=r'model_pth/vit_base_patch16_224.pth',
    help='initial weights path'
)


opt = parser.parse_args()  

if opt.seed:
    def seed_torch(seed=10):
        random.seed(seed)  # Python random module.	
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。
        # 如需完全可复现，可打开：
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        print('random seed has been fixed')
    seed_torch() 


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    # ================= TensorBoard =================
    if opt.tensorboard:
        # 这是存放你要使用tensorboard显示的数据的绝对路径
        log_path = os.path.join('./results/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path)  # 当log文件存在时删除文件夹

        # 实例化一个tensorboard
        tb_writer = SummaryWriter(log_path)
    else:
        tb_writer = None
    
    # ================= 数据增强 =================
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    } 
 
    # ================= 数据集 =================
    # 对标 pytorch 的 ImageFolder，自定义 Five_Flowers_Load
    train_dataset = Five_Flowers_Load(
        os.path.join(args.data_path, 'train'),
        transform=data_transform["train"]
    )
    val_dataset = Five_Flowers_Load(
        os.path.join(args.data_path, 'val'),
        transform=data_transform["val"]
    ) 
 
    if args.num_classes != train_dataset.num_class:
        raise ValueError("dataset have {} classes, but input {}".format(
            train_dataset.num_class, args.num_classes
        ))
    
    nw = 0
    # nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
 
    # ================= 创建模型 =================
    model = classic_models.find_model_using_name(
        opt.model, num_classes=opt.num_classes
    )  # 先在 CPU 上创建
 
    # ================= 预训练权重加载（按参考代码方式加入 ViT 权重加载） =================
    if args.weights != "":
        assert os.path.exists(args.weights), \
            "weights file: '{}' not exist.".format(args.weights)
        
        print(f"=> Loading pretrained weights from: {args.weights}")
        # 建议统一先加载到 CPU
        state_dict = torch.load(args.weights, map_location=torch.device('cpu'))

        # 兼容几种常见封装：{'state_dict': ...} / {'model': ...}
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # 判断是否为 ViT / Swin 这类带 head 的 transformer
        is_vit_like = ("head.weight" in state_dict) or ("head.bias" in state_dict)

        model_dict = model.state_dict()
        filtered_dict = {}

        if is_vit_like:
            print("=> Detected Vision Transformer-like weights, removing classification head ...")
            # 删除 ViT/Swin 中的 head 层
            del_keys = ['head.weight', 'head.bias']
            for k in del_keys:
                if k in state_dict:
                    print(f"  - drop: {k}")
                    del state_dict[k]

            # 只加载形状完全匹配的参数
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v

            print(f"=> Loaded {len(filtered_dict)} transformer layers (strict=False)")
            # 直接用 strict=False 加载（参考你原来的 VIT 写法）
            load_info = model.load_state_dict(filtered_dict, strict=False)
            print(load_info)

        else:
            print("=> Detected CNN-like weights (ResNet / DenseNet / etc.), only loading matched layers ...")
            # 只加载 shape 匹配的层（避免 fc、classifier 维度不一致报错）
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v

            print(f"=> Loaded {len(filtered_dict)} CNN layers (strict=False)")
            model_dict.update(filtered_dict)
            load_info = model.load_state_dict(model_dict, strict=False)
            print(load_info)

    else:
        print("⚠ 未提供预训练权重，将使用随机初始化参数训练。")
    # =====================================================================

    # 权重加载完再整体搬到 GPU / device
    model = model.to(device)
        
    pg = [p for p in model.parameters() if p.requires_grad] 
    optimizer = optim.Adam(pg, lr=args.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.
    
    # ================= 保存权重的目录 =================
    save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    log_txt = os.path.join(save_path, "transformer1_log.txt")
    weight_path = os.path.join(save_path, "transformer1.pth")

    # ================= 训练循环 =================
    for epoch in range(args.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            lr_method=warmup
        )
        scheduler.step()

        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)

        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f'
              % (epoch + 1, mean_loss, train_acc, val_acc))
        with open(log_txt, 'a') as f: 
            f.writelines(
                '[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f\n'
                % (epoch + 1, mean_loss, train_acc, val_acc)
            )

        if opt.tensorboard and tb_writer is not None:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # 判断当前验证集的准确率是否是最大的，如果是，则更新之前保存的权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            print(f"=> New best val_acc: {best_acc:.4f}, model saved to {weight_path}") 

        
main(opt)
