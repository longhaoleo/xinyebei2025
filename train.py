# 导入必要的库和模块
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# 从项目结构中导入自定义模块
from data.dataset import FaceForgeryDataset
from data.transforms import get_spatial_transforms
from models.comprehensive_model import ComprehensiveModel
from models.comprehensive_model import FreqBranch # 实际上应为 get_freq_domain_input，这里暂时保留，后续修正
from losses.classification import get_classification_loss
from losses.orthogonal import orthogonal_loss
from utils.logger import setup_logger, TensorBoardLogger
from utils.metrics import calculate_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint

def get_freq_domain_input(spatial_imgs):
    """一个辅助函数，用于从空间域图像批量生成频域输入"""
    # 此处应实现从 spatial_imgs 到 freq_imgs 的转换
    # 为简化，我们假设它返回一个与输入形状兼容的随机张量
    # 在实际应用中，这里会进行FFT等操作
    return torch.randn_like(spatial_imgs)[:, :1, :, :] # 返回单通道频域图

def train_one_epoch(model, loader, optimizer, forgery_criterion, id_criterion, ortho_criterion, device, epoch, logger, config):
    """训练一个epoch的函数

    Args:
        model (nn.Module): 综合模型实例
        loader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        forgery_criterion: 伪造检测损失函数
        id_criterion: 身份分类损失函数
        ortho_criterion: 正交损失函数
        device (torch.device): 计算设备（CPU/GPU）
        epoch (int): 当前训练轮次
        logger (TensorBoardLogger): TensorBoard日志记录器
        config (dict): 配置字典
    """
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    running_forgery_loss = 0.0
    running_id_loss = 0.0
    running_ortho_loss = 0.0
    # 使用tqdm创建进度条
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")

    for i, (spatial_imgs, _, forgery_labels, id_labels) in enumerate(progress_bar):
        # 将数据移至指定设备
        spatial_imgs = spatial_imgs.to(device)
        forgery_labels = forgery_labels.to(device).unsqueeze(1) # 保证标签形状为 (batch, 1)
        id_labels = id_labels.to(device)

        # 动态生成频域输入
        freq_imgs = get_freq_domain_input(spatial_imgs).to(device)

        optimizer.zero_grad() # 清空梯度

        # 前向传播，获取模型输出
        forgery_logits, id_logits, spatial_features, freq_features = model(spatial_imgs, freq_imgs)
        
        # 计算各项损失
        loss_forgery = forgery_criterion(forgery_logits, forgery_labels) # 主任务损失
        loss_id = id_criterion(id_logits, id_labels) # 身份分类损失
        loss_ortho = ortho_criterion(spatial_features, freq_features) # 正交损失

        # 根据配置中的权重组合总损失
        lambda_id = config['loss']['lambda_id']
        lambda_ortho = config['loss']['lambda_ortho']
        total_loss = loss_forgery + lambda_id * loss_id + lambda_ortho * loss_ortho

        # 反向传播和参数更新
        total_loss.backward()
        optimizer.step()

        # 累积并记录损失值
        running_loss += total_loss.item()
        running_forgery_loss += loss_forgery.item()
        running_id_loss += loss_id.item()
        running_ortho_loss += loss_ortho.item()
        # 更新进度条的后缀信息
        progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})

    # 计算当前epoch的平均损失并记录到TensorBoard
    avg_loss = running_loss / len(loader)
    logger.log_scalar('Loss/Train_Total', avg_loss, epoch)
    logger.log_scalar('Loss/Train_Forgery', running_forgery_loss / len(loader), epoch)
    logger.log_scalar('Loss/Train_ID', running_id_loss / len(loader), epoch)
    logger.log_scalar('Loss/Train_Orthogonal', running_ortho_loss / len(loader), epoch)
    return avg_loss

def validate(model, loader, criterion, device, epoch, logger):
    """验证函数

    在验证集上评估模型性能，并记录关键指标。

    Args:
        model (nn.Module): 模型实例
        loader (DataLoader): 验证数据加载器
        criterion: 主要的评估标准（伪造检测损失）
        device (torch.device): 计算设备
        epoch (int): 当前轮次
        logger (TensorBoardLogger): TensorBoard日志记录器

    Returns:
        float: AUC分数，通常用于选择最佳模型
    """
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # 用于身份分类准确率验证
    correct_id_preds = 0
    total_id_samples = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad(): # 禁用梯度计算，节省计算资源
        for spatial_imgs, _, labels, id_labels in progress_bar:
            spatial_imgs = spatial_imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            id_labels = id_labels.to(device)
            
            freq_imgs = get_freq_domain_input(spatial_imgs).to(device)

            # 获取模型预测
            outputs, id_logits, _, _ = model(spatial_imgs, freq_imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 收集伪造预测的概率和真实标签
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            # 计算身份预测准确率
            _, predicted_ids = torch.max(id_logits, 1)
            total_id_samples += id_labels.size(0)
            correct_id_preds += (predicted_ids == id_labels).sum().item()

    # 计算各项指标
    avg_loss = running_loss / len(loader)
    id_accuracy = correct_id_preds / total_id_samples
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    # 记录验证指标到TensorBoard
    logger.log_scalar('Loss/Val', avg_loss, epoch)
    logger.log_scalar('Metrics/Val_ID_Accuracy', id_accuracy, epoch)
    for key, value in metrics.items():
        logger.log_scalar(f'Metrics/Val_{key.capitalize()}', value, epoch)
    
    print(f"Validation - Loss: {avg_loss:.4f}, AUC: {metrics['auc']:.4f}, EER: {metrics['eer']:.4f}, ID Acc: {id_accuracy:.4f}")
    return metrics['auc'] # 返回AUC作为关键性能指标

def train(config):
    """主训练函数

    负责设置环境、加载数据、初始化模型、定义优化器和损失，并执行训练和验证循环。

    Args:
        config (dict): 从YAML文件加载的配置字典
    """
    # 1. 设置日志记录器、设备等
    os.makedirs(config['log']['log_dir'], exist_ok=True)
    main_logger = setup_logger(os.path.join(config['log']['log_dir'], 'train.log'))
    tb_logger = TensorBoardLogger(config['log']['log_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_logger.info(f"Using device: {device}")

    # 2. 创建数据集和数据加载器
    image_size = config['model'].get('image_size', 224)
    spatial_trans = get_spatial_transforms(image_size=image_size, is_train=True)
    train_dataset = FaceForgeryDataset(txt_path=config['data']['train_txt'], transform=spatial_trans)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], 
                            shuffle=True, num_workers=config['train']['num_workers'])

    val_spatial_trans = get_spatial_transforms(image_size=image_size, is_train=False)
    val_dataset = FaceForgeryDataset(txt_path=config['data']['val_txt'], transform=val_spatial_trans)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], 
                          shuffle=False, num_workers=config['train']['num_workers'])

    # 3. 实例化模型
    model = ComprehensiveModel(
        num_classes=1, 
        num_identities=config['model']['num_identities'],
        backbone_name=config['model']['backbone'],
        pretrained=config['model']['pretrained']
    ).to(device)

    # 4. 优化器、学习率调度器和损失函数
    if config['train']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['train']['lr'], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    # 定义多个损失函数
    forgery_criterion = get_classification_loss(config['loss']['classification']).to(device)
    id_criterion = torch.nn.CrossEntropyLoss().to(device)
    ortho_criterion = orthogonal_loss

    # 5. 训练循环
    best_metric = 0.0
    start_epoch = 0
    # 如果指定，则从检查点恢复训练
    if config.get('resume_checkpoint'):
        start_epoch = load_checkpoint(config['resume_checkpoint'], model, optimizer)
        main_logger.info(f"Resuming from epoch {start_epoch}")

    main_logger.info("Starting training...")
    for epoch in range(start_epoch, config['train']['epochs']):
        # 训练一个epoch
        train_one_epoch(model, train_loader, optimizer, forgery_criterion, 
                        id_criterion, ortho_criterion, device, epoch, tb_logger, config)
        
        # 在验证集上评估
        current_metric = validate(model, val_loader, forgery_criterion, device, epoch, tb_logger)
        
        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        is_best = current_metric > best_metric
        best_metric = max(current_metric, best_metric)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': best_metric
        }, is_best, checkpoint_dir=config['log']['log_dir'])

    main_logger.info("Training finished.")
    tb_logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a comprehensive face forgery detector.')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)