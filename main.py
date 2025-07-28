import os
import shutil
import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")




# 创建项目目录结构
def create_project_structure():
    project_dirs = [
        'datasets/mnist_yolo/train/images',
        'datasets/mnist_yolo/train/labels',
        'datasets/mnist_yolo/val/images',
        'datasets/mnist_yolo/val/labels',
        'runs',
        'saved_models',
        'test_images',
        'results'
    ]
    
    for directory in project_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录: {directory}")
    
    print("\n项目目录结构创建完成!")
    
    
    

# 下载并转换MNIST数据集为YOLO格式
def prepare_mnist_dataset():
    print("\n准备MNIST数据集...")
    
    # 下载MNIST数据集
    train_set = torchvision.datasets.MNIST(
        root='./Original_MNIST_Data', 
        train=True,
        download=True
    )
    
    test_set = torchvision.datasets.MNIST(
        root='./Original_MNIST_Data', 
        train=False,
        download=True
    )
    
    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")
    
    # 转换训练集
    print("转换训练集...")
    for idx, (img, label) in enumerate(train_set):
        # 保存图像
        img_path = f"datasets/mnist_yolo/train/images/{idx}.jpg"
        img.save(img_path)
        
        # 创建YOLO标签文件（全图边界框）
        label_path = f"datasets/mnist_yolo/train/labels/{idx}.txt"
        with open(label_path, "w") as f:
            # 格式: class_id center_x center_y width height
            f.write(f"{label} 0.5 0.5 1.0 1.0")
    
    # 转换测试集
    print("转换测试集...")
    for idx, (img, label) in enumerate(test_set):
        img_path = f"datasets/mnist_yolo/val/images/{idx}.jpg"
        img.save(img_path)
        
        label_path = f"datasets/mnist_yolo/val/labels/{idx}.txt"
        with open(label_path, "w") as f:
            f.write(f"{label} 0.5 0.5 1.0 1.0")
    
    print("数据集转换完成!")
    
    # 创建样本图像
    create_sample_images(train_set)

# 创建样本图像用于展示
def create_sample_images(dataset):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        img, label = dataset[i]
        plt.subplot(5, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    
    plt.savefig('results/mnist_samples.png')
    print("已保存样本图像: results/mnist_samples.png")

# 创建数据集配置文件
def create_dataset_config():
    config = """
    path: ./datasets/mnist_yolo
    train: train/images
    val: val/images
    
    # 类别数 (0-9)
    nc: 10
    
    # 类别名称
    names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """
    
    with open('mnist.yaml', 'w') as f:
        f.write(config)
    
    print("\n数据集配置文件创建完成: mnist.yaml")

# 训练YOLOv8模型
def train_yolov8_model():
    print("\n开始训练YOLOv8模型...")
    
    # 加载预训练模型
    model = YOLO('yolov8l.pt')
    
    # 训练参数
    train_args = {
        'data': 'mnist.yaml',
        'epochs': 128,
        'imgsz': 28,
        'batch': 256,  # 充分利用RTX 4060的8GB显存
        'device': 0,   # 使用GPU
        'optimizer': 'Adam',
        'lr0': 0.001,
        'weight_decay': 0.0001,
        'name': 'mnist_detection',
        'amp': True,   # 启用混合精度训练
        'seed': 42,
        'patience': 32,
        'dropout': 0.1,
        'translate': 0.1,  # 数据增强：小幅平移
        'hsv_h': 0.0,      # 禁用色调增强
        'hsv_s': 0.0       # 禁用饱和度增强
    }
    
    # 开始训练
    results = model.train(**train_args)
    
    # 保存最佳模型
    best_model_path = f"runs/detect/{train_args['name']}/weights/best.pt"
    shutil.copy(best_model_path, 'saved_models/yolov8_mnist_best.pt')
    print(f"\n最佳模型已保存至: saved_models/yolov8_mnist_best.pt")
    
    return results



def validate_model():
    print("\n验证模型性能...")
    model = YOLO('saved_models/yolov8_mnist_best.pt')
    val_args = {
        'data': 'mnist.yaml',
        'split': 'val',
        'batch': 128,
        'device': 0,
        'plots': True
    }
    metrics = model.val(**val_args)
    
    # 获取关键指标
    nc = metrics.box.nc  # 类别数量
    p = metrics.box.p    # 各类别精确率列表
    r = metrics.box.r    # 各类别召回率列表
    ap50_list = metrics.box.ap50  # 各类别AP@0.5（现在是属性，不是方法）
    maps_list = metrics.box.maps  # 各类别mAP@0.5:0.95

    print("\n验证结果:")
    print("-- 平均精度指标 --")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"各类别平均精度: {[round(x, 4) for x in maps_list]}")

    print("\n-- 精确率与召回率 --")
    print(f"平均精确率: {metrics.box.mp:.4f}")
    print(f"平均召回率: {metrics.box.mr:.4f}")

    print("\n-- 置信度阈值相关指标 --")
    print(f"类别0精确率: {p[0]:.4f}")
    print(f"类别0召回率: {r[0]:.4f}")

    print("\n-- 速度与效率指标 --")
    print(f"预处理时间: {metrics.speed['preprocess']:.2f} ms/图像")
    print(f"推理时间: {metrics.speed['inference']:.2f} ms/图像")
    print(f"后处理时间: {metrics.speed['postprocess']:.2f} ms/图像")
    print(f"总处理时间: {sum(metrics.speed.values()):.2f} ms/图像")
    
    # 打印各类别详细指标
    print("\n-- 各类别详细指标 --")
    for class_id in range(nc):
        print(f"类别 {class_id}:")
        print(f"  AP@0.5: {ap50_list[class_id]:.4f}")
        print(f"  AP@0.5:0.95: {maps_list[class_id]:.4f}")
        print(f"  精确率: {p[class_id]:.4f}  召回率: {r[class_id]:.4f}")
    
    return metrics


def generate_test_images():
    print("\n生成测试图像...")
    os.makedirs('test_images', exist_ok=True)  # 确保目录存在
    
    digit_dir = Path('datasets/mnist_yolo/val/images')
    for i in range(100):
        canvas = np.zeros((100, 250), dtype=np.uint8)
        digits = np.random.randint(0, 10, 5)
        
        for j, digit in enumerate(digits):
            digit_files = list(digit_dir.glob(f'{digit}_*.jpg'))
            if digit_files:
                img_path = np.random.choice(digit_files)
                digit_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                x_offset = 10 + j * 50
                canvas[20:48, x_offset:x_offset+28] = digit_img
        
        cv2.imwrite(f'test_images/test_{i}.jpg', canvas)
    
    print(f"已生成{100}张测试图像到 test_images/ 目录")

def run_predictions():
    print("\n执行预测...")
    model = YOLO('saved_models/yolov8_mnist_best.pt')
    os.makedirs('results/predictions', exist_ok=True)
    
    test_images = [f for f in os.listdir('test_images') if f.endswith('.jpg')]
    
    fig = plt.figure(figsize=(15, 8))
    for i, img_name in enumerate(test_images[:min(5, len(test_images))]):  # 安全处理
        img_path = os.path.join('test_images', img_name)
        results = model.predict(img_path, conf=0.5, imgsz=100)  # 调整尺寸匹配生成图像
        
        # 处理预测结果
        if results and results[0]:
            res_img = results[0].plot()
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            save_path = f'results/predictions/{img_name}'
            cv2.imwrite(save_path, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
            
            ax = fig.add_subplot(2, 3, i+1)
            ax.imshow(res_img)
            ax.set_title(f'预测结果: {img_name}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/all_predictions.png')
    print("预测结果已保存至 results/predictions/")


# 主函数
def main():
    print("="*50)
    print("YOLOv8 MNIST手写数字识别项目")
    print("="*50)
    
    # 步骤1: 创建项目结构
    create_project_structure()
    
    # 步骤2: 准备数据集
    prepare_mnist_dataset()
    
    # 步骤3: 创建数据集配置
    create_dataset_config()
    
    # 步骤4: 训练模型
    train_yolov8_model()
    
    # 步骤5: 验证模型
    validate_model()
    
    # 步骤6: 生成测试图像
    generate_test_images()
    
    # 步骤7: 执行预测
    run_predictions()
    
    print("\n" + "="*50)
    print("项目执行完成!")
    print("="*50)

if __name__ == "__main__":
    main()
    
    
