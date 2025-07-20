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

def get_freq_domain_input(spatial_imgs):
    """一个辅助函数，用于从空间域图像批量生成频域输入"""
    # 在评估时，也需要与训练时一致的频域处理
    return torch.randn_like(spatial_imgs)[:, :1, :, :] # 示例

def evaluate(config):
    """主评估函数

    在测试集上加载训练好的模型，并计算详细的性能指标。

    Args:
        config (dict): 配置字典，包含评估参数
    """
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 创建数据集和数据加载器
    image_size = config['model'].get('image_size', 224)
    # 使用验证/测试变换（无随机增强）
    spatial_trans = get_spatial_transforms(image_size=image_size, is_train=False)
    
    # 获取测试集标注文件路径
    test_txt_path = config['data']['test_txt'] 
    if not os.path.exists(test_txt_path):
        print(f"Test file not found at {test_txt_path}. Please specify 'test_txt' in your config.")
        return

    test_dataset = FaceForgeryDataset(txt_path=test_txt_path, transform=spatial_trans)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], 
                           shuffle=False, num_workers=config['train']['num_workers'])

    # 3. 实例化模型
    model = ComprehensiveModel(
        num_classes=1, 
        num_identities=config['model']['num_identities'],
        backbone_name=config['model']['backbone'],
        pretrained=False # 加载检查点时不需要预训练权重
    ).to(device)

    # 4. 加载训练好的模型权重
    if config.get('eval_checkpoint'):
        load_checkpoint(config['eval_checkpoint'], model)
        print(f"Loaded checkpoint from {config['eval_checkpoint']}")
    else:
        print("No checkpoint specified for evaluation. Please set 'eval_checkpoint' in the config.")
        return

    # 开始评估
    model.eval() # 设置为评估模式
    all_labels = []
    all_preds = []
    progress_bar = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad(): # 禁用梯度计算
        for spatial_imgs, _, labels, _ in progress_bar:
            spatial_imgs = spatial_imgs.to(device)
            
            # 评估时，我们只关心伪造预测的输出
            freq_imgs = get_freq_domain_input(spatial_imgs).to(device)
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

    # 在运行前，请确保在配置文件中指定了测试数据路径和模型检查点路径
    # 例如:
    # data:
    #   test_txt: 'path/to/your/test.txt'
    # eval_checkpoint: 'path/to/your/model_best.pth.tar'
    
    evaluate(config)