# 导入必要的库
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

# 从项目结构中导入自定义模块
from data.dataset import FaceForgeryDataset
from data.transforms import get_spatial_transforms
from models.comprehensive_model import ComprehensiveModel
from utils.metrics import calculate_metrics
from utils.checkpoint import load_checkpoint
from train import get_freq_domain_input # 从训练脚本中导入相同的频域处理函数

def evaluate(config):
    """主评估函数

    在测试集上加载训练好的模型，并计算详细的性能指标。

    Args:
        config (dict): 配置字典，包含评估所需的参数。
    """
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 创建数据集和数据加载器
    # 使用与验证时相同的图像变换（通常不包含数据增强）
    spatial_trans = get_spatial_transforms(is_train=False)
    
    # 获取测试集标注文件路径
    test_txt_path = config['data'].get('test_txt', 'data/test.txt') 
    if not os.path.exists(test_txt_path):
        print(f"Error: Test file not found at {test_txt_path}. Please specify 'test_txt' in your config.")
        return

    test_dataset = FaceForgeryDataset(txt_path=test_txt_path, transform=spatial_trans)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], 
                           shuffle=False, num_workers=config['train']['num_workers'])

    # 3. 实例化模型
    model = ComprehensiveModel(
        backbone_name=config['model']['backbone'],
        num_identities=config['model']['num_identities'],
        pretrained=False # 从检查点加载权重，因此这里设为False
    ).to(device)

    # 4. 加载训练好的模型权重
    checkpoint_path = config.get('eval_checkpoint')
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Error: No valid checkpoint specified. Please set 'eval_checkpoint' in the config.")
        return

    # 开始评估
    model.eval() # 设置为评估模式，关闭dropout和BN的训练行为
    all_labels = []
    all_preds = []
    progress_bar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad(): # 禁用梯度计算以加速并节省内存
        for spatial_imgs, _, labels, _ in progress_bar:
            spatial_imgs = spatial_imgs.to(device)
            
            # 生成频域输入
            freq_imgs = get_freq_domain_input(spatial_imgs).to(device)
            # 在评估时，我们只关心伪造预测的输出
            forgery_logits, _, _, _ = model(spatial_imgs, freq_imgs)

            # 计算概率并收集结果
            preds = torch.sigmoid(forgery_logits).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 5. 计算并打印指标
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    print("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        print(f"{key.capitalize():<12}: {value:.4f}")
    print("------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a face forgery detector.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(config)