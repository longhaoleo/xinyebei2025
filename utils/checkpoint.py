# 导入必要的库
import os
import torch

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False):
    """
    保存模型检查点，包括模型状态、优化器状态、轮次和评估指标。

    参数:
        model (torch.nn.Module): 需要保存的模型。
        optimizer (torch.optim.Optimizer): 优化器实例。
        epoch (int): 当前的训练轮次。
        metrics (dict): 包含评估指标（如auc, f1, eer）的字典。
        save_path (str): 检查点保存的路径。
        is_best (bool): 如果为True，则额外保存一个名为 'best_model.pth' 的文件，代表当前最佳模型。
    """
    # 创建一个字典来保存所有需要的信息
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    # 保存检查点文件
    torch.save(state, save_path)
    # 如果是最佳模型，额外保存一个副本，方便快速找到最佳模型
    if is_best:
        best_model_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(state, best_model_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点，恢复模型和优化器的状态。

    参数:
        model (torch.nn.Module): 需要加载状态的模型。
        optimizer (torch.optim.Optimizer): 需要加载状态的优化器。
        checkpoint_path (str): 检查点文件的路径。

    返回:
        tuple: 返回加载的轮次和指标字典。
    """
    # 加载检查点文件到CPU或GPU
    checkpoint = torch.load(checkpoint_path)
    # 加载模型和优化器的状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 返回保存的轮次和指标
    return checkpoint['epoch'], checkpoint.get('metrics', {})