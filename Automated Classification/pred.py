import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import classic_models  # 你自己的模型库
import logging
import csv

# ================== 日志设置 ==================
logging.basicConfig(
    filename='xxxx.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log_and_print(message: str):
    print(message)
    logging.info(message)


# ================== 加载模型 ==================
def load_model(model_name: str, num_classes: int, weights_path: str, device: str):
    """
    根据模型名和权重路径加载模型
    """
    model = classic_models.find_model_using_name(
        model_name, num_classes=num_classes
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ================== 单张图预测 ==================
def predict(model, data, device: str, num_classes: int):
    """
    对单个 batch（这里 batch_size=1）做预测
    返回: pred_label(int), prob_list(list[float])

    重要修正：
    - 若模型输出通道数 != num_classes（比如 1000 vs 2），
      则只取前 num_classes 个通道，再重新 softmax，保证预测类别 ∈ [0, num_classes-1]
    """
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)  # [1, C]
        C = outputs.shape[1]

        if C != num_classes:
            # 仅提示一次也行，这里简单每次都打印，方便你确认
            log_and_print(
                f"警告: 模型输出通道数={C} 与 num_classes={num_classes} 不一致，"
                f"仅取前 {num_classes} 个通道参与二分类。"
            )
            outputs = outputs[:, :num_classes]  # [1, num_classes]

        soft_max_predict = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(soft_max_predict, dim=1)

    return predicted.item(), soft_max_predict.cpu().squeeze().tolist()


# ================== 主函数 ==================
def main_folder(model_name: str,
                num_classes: int,
                weights: str,
                root_folder: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_and_print(f"Using device: {device}")
    log_and_print(f"Root folder: {root_folder}")

    # 1) 加载模型
    model = load_model(model_name, num_classes, weights, device)

    # 2) 图像预处理（需与训练阶段保持一致）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ============ 新增：用于导出 CSV 的列表 ============
    results = []

    # 3) 找到所有子文件夹（类别）
    class_dirs = []
    for name in sorted(os.listdir(root_folder)):
        sub_dir = os.path.join(root_folder, name)
        if not os.path.isdir(sub_dir):
            continue

        try:
            label = int(name)
        except ValueError:
            log_and_print(f"跳过非数字类别文件夹: {sub_dir}")
            continue

        if label < 0 or label >= num_classes:
            log_and_print(f"警告: 文件夹 {sub_dir} 的类别 {label} 超出范围 0~{num_classes-1}")
            continue

        class_dirs.append((label, sub_dir))

    if not class_dirs:
        log_and_print("没有找到有效的类别子文件夹，程序结束。")
        return

    log_and_print(f"将评估的类别文件夹: {class_dirs}")

    # 混淆矩阵
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    total_files = 0
    total_correct = 0

    # 4) 遍历每个类别文件夹
    for true_label, folder_path in class_dirs:
        log_and_print(f"开始遍历类别 {true_label} 的文件夹: {folder_path}")

        for fname in os.listdir(folder_path):
            img_path = os.path.join(folder_path, fname)

            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                log_and_print(f"跳过无法打开的文件: {img_path}, error: {e}")
                continue

            data = transform(image).unsqueeze(0)

            pred_label, prob_list = predict(model, data, device, num_classes)
            img_name = os.path.basename(img_path)

            # 写日志
            log_and_print(
                f"样本名称: {img_name}\n"
                f"真实标签: {true_label}\n"
                f"预测概率值: {prob_list}\n"
                f"预测标签: {pred_label}\n"
            )

            # 保险起见再检查一次（正常情况下已经保证 0/1）
            if pred_label < 0 or pred_label >= num_classes:
                log_and_print(
                    f"⚠ 预测类别 {pred_label} 超出范围 [0, {num_classes-1}]，该样本不计入混淆矩阵。"
                )
                continue

            # ============ 存入 results 列表 ============
            # 这里假设 num_classes=2，对应 prob0/ prob1，如果是多类可以改成动态写入
            if len(prob_list) >= 2:
                prob0, prob1 = prob_list[0], prob_list[1]
            else:
                # 理论上不会发生
                prob0, prob1 = prob_list[0], None

            results.append({
                "img_name": img_name,
                "true": true_label,
                "pred": pred_label,
                "prob0": prob0,
                "prob1": prob1,
            })

            # 混淆矩阵
            confusion[true_label][pred_label] += 1

            total_files += 1
            if pred_label == true_label:
                total_correct += 1

    # ================== 导出 CSV ==================
    csv_path = f"xxxx.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img_name", "true", "pred", "prob0", "prob1"]
        )
        writer.writeheader()
        writer.writerows(results)

    log_and_print(f"预测结果已保存到 CSV：{csv_path}")
    print(f"预测结果已保存到 CSV：{csv_path}")

    # ================== 计算指标 ==================
    if total_files == 0:
        log_and_print("没有成功预测的图像，无法计算指标。")
        return

    overall_acc = total_correct / total_files

    log_and_print("========== Confusion Matrix (行=真实, 列=预测) ==========")
    for i in range(num_classes):
        log_and_print(f"True {i}: {confusion[i]}")

    log_and_print(f"总样本数: {total_files}")
    log_and_print(f"总体准确率 Accuracy: {overall_acc:.4f}")

    print("========== Confusion Matrix (行=真实, 列=预测) ==========")
    for i in range(num_classes):
        print(f"True {i}: {confusion[i]}")
    print(f"总样本数: {total_files}")
    print(f"总体准确率 Accuracy: {overall_acc:.4f}")

    # ================== Per-Class Metrics ==================
    log_and_print("========== Per-Class Metrics ==========")
    print("========== Per-Class Metrics ==========")
    for c in range(num_classes):
        tp = confusion[c][c]
        fn = sum(confusion[c]) - tp
        fp = sum(confusion[r][c] for r in range(num_classes)) - tp
        tn = total_files - tp - fp - fn

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0.0

        msg = (
            f"Class {c}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )

        log_and_print(msg)
        print(msg)


# ================== 脚本入口 ==================
if __name__ == "__main__":
    train_root = "/test"
    weights = "/xxxxx.pth"

    main_folder(
        model_name="xxxx",
        num_classes=2,
        weights=weights,
        root_folder=train_root
    )
