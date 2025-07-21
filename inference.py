# 导入必要的库
import argparse
import yaml
import torch
from PIL import Image
import numpy as np
import os

# 从项目结构中导入自定义模块
from data.transforms import get_spatial_transforms
from models.comprehensive_model import ComprehensiveModel
from utils.checkpoint import load_checkpoint
from train import get_freq_domain_input # 确保与训练时使用相同的函数

def inference(image_path, config_path):
    """对单张图像进行伪造检测推理

    Args:
        image_path (str): 输入图像的路径。
        config_path (str): 训练时使用的配置文件的路径。
    """
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. 实例化模型
    model = ComprehensiveModel(
        backbone_name=config['model']['backbone'],
        num_identities=config['model']['num_identities'],
        pretrained=False # 不使用ImageNet预训练权重，因为我们将加载自己的检查点
    ).to(device)

    # 4. 加载训练好的模型权重
    checkpoint_path = config.get('eval_checkpoint')
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Error: No valid checkpoint specified. Please set 'eval_checkpoint' in the config.")
        return

    model.eval() # 设置为评估模式

    # 5. 加载并预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    transform = get_spatial_transforms(is_train=False)
    # unsqueeze(0) 是为了增加一个批次维度，将 [C, H, W] 变为 [1, C, H, W]
    spatial_img = transform(image).unsqueeze(0).to(device)

    # 6. 执行推理
    with torch.no_grad(): # 推理时不需要计算梯度
        freq_img = get_freq_domain_input(spatial_img).to(device)
        # 我们只需要伪造分类的输出
        forgery_logits, _, _, _ = model(spatial_img, freq_img)
        # 使用sigmoid函数将logits转换为[0, 1]范围内的概率
        probability = torch.sigmoid(forgery_logits).item()

    # 7. 显示结果
    print("\n--- Inference Result ---")
    print(f"Image: {image_path}")
    print(f"Predicted Probability of being FAKE: {probability:.4f}")
    if probability > 0.5:
        print("Prediction: FAKE")
    else:
        print("Prediction: REAL")
    print("------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a single face image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                      help='Path to the config file used for training.')
    args = parser.parse_args()

    inference(args.image, args.config)