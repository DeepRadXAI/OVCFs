############################################################################################################
# 增添了以下功能：
# 1. 使用argparse类实现可以在训练的启动命令中指定超参数
# 2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
# 3. 使用了更高级的学习策略 cosine warm up：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。 
# 4. 可以通过在启动命令中指定 --model 来选择使用的模型 
# 5. 使用amp包实现半精度训练，在保证准确率的同时尽可能的减小训练成本
# 6. 实现了数据加载类的自定义实现
# 7. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
#    注意，使用tensorboad之前需要使用命令 "tensorboard --logdir= log_path" 来启动，
#    结果通过网页 http://localhost:6006/ 查看可视化结果
############################################################################################################
# --model 可选的超参如下：
# alexnet   zfnet   vgg   vgg_tiny   vgg_small   vgg_big   googlenet   xception   resnet_small   resnet   resnet_big   resnext   resnext_big  
# densenet_tiny   densenet_small   densenet   densenet_big   mobilenet_v3   mobilenet_v3_large   shufflenet_small   shufflenet
# efficient_v2_small   efficient_v2   efficient_v2_large   convnext_tiny   convnext_small   convnext   convnext_big   convnext_huge
# vision_transformer_small   vision_transformer   vision_transformer_big   swin_transformer_tiny   swin_transformer_small   swin_transformer 
#
# 训练命令示例： 
#   python train_densenet121_weight.py --model densenet_big --num_classes 2
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
parser.add_argument('--lr', type=float, default=0.0002, help='start learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate')
parser.add_argument('--seed', default=21, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help='use tensorboard for visualization')
parser.add_argument('--use_amp', default=False, action='store_true', help='training with mixed precision')
parser.add_argument('--data_path', type=str, default=r"data/OVCFs_JPG")
parser.add_argument('--model', type=str, default="densenet_big", help='select a model for training')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

parser.add_argument(
    '--weights',
    type=str,
    default=r"/model_pth/densenet121.pth",
    help='initial weights path'
)

opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)  # Python random module.
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # current GPU
        torch.cuda.manual_seed_all(seed)  # all GPU
        # 如需完全可复现，可再打开下面两行（会稍微慢一点）
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        print('random seed has been fixed')
    seed_torch()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    # ================= TensorBoard =================
    if opt.tensorboard:
        log_path = os.path.join('./results/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path))

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path)

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
    train_dataset = Five_Flowers_Load(
        os.path.join(args.data_path, 'train'),
        transform=data_transform["train"]
    )
    val_dataset = Five_Flowers_Load(
        os.path.join(args.data_path, 'val'),
        transform=data_transform["val"]
    )

    if args.num_classes != train_dataset.num_class:
        raise ValueError(
            "dataset have {} classes, but input {}".format(
                train_dataset.num_class, args.num_classes)
        )

    nw = 0
    print('Using {} dataloader workers every process'.format(nw))

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

    # ================= 创建模型（先在 CPU 上），方便安全加载权重 =================
    model = classic_models.find_model_using_name(
        opt.model, num_classes=opt.num_classes
    )  # 先不 .to(device)

    # ====================== DenseNet121 预训练权重加载 ======================
    if args.weights != "":
        assert os.path.exists(args.weights), \
            f"weights file: '{args.weights}' not exist."

        print(f"=> Loading DenseNet121 pretrained weights from: {args.weights}")
        # 一律先加载到 CPU，避免 CPU / CUDA storage 冲突
        state_dict = torch.load(args.weights, map_location='cpu')

        # 有些权重会包一层 {'state_dict': ...} 或 {'model': ...}
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # 删除 densenet 的分类层（避免 num_classes 不一致）
        del_keys = ["classifier.weight", "classifier.bias"]
        for k in del_keys:
            if k in state_dict:
                print(f"Delete head param from pretrained dict: {k}")
                del state_dict[k]

        model_dict = model.state_dict()
        filtered_dict = {}

        # 只加载形状完全匹配的层
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_dict[k] = v

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered_dict)}/{len(model_dict)} layers from DenseNet121 pretrained weights.")
    else:
        print("⚠ 未提供预训练权重，将使用随机初始化参数训练。")
    # ==========================================================================

    # 权重加载完再整体搬到 GPU / device
    model = model.to(device)

    # 只优化 requires_grad=True 的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)

    # cosine lr 调度
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.

    # ================= 保存权重的目录 =================
    save_path = os.path.join(os.getcwd(), 'results/weights', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    log_txt = os.path.join(save_path, "densenet121_log.txt")
    weight_path = os.path.join(save_path, "densenet121.pth")

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

        # 保存当前最优权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            print(f"=> New best val_acc: {best_acc:.4f}, model saved to {weight_path}")


main(opt)
