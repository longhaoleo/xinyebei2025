# 导入必要的库
import os
import torch

def save_checkpoint(model, optimizer, epoch, miou, auc, f1, eer, save_path, is_best=False):
    """
    保存模型检查点。

    参数:
        model (torch.nn.Module): 需要保存的模型。
        optimizer (torch.optim.Optimizer): 优化器实例。
        epoch (int): 当前的训练轮次。
        miou (float): 平均交并比 (Mean Intersection over Union)。
        auc (float): ROC曲线下面积 (Area Under the Curve)。
        f1 (float): F1分数。
        eer (float): 相等错误率 (Equal Error Rate)。
        save_path (str): 检查点保存的路径。
        is_best (bool): 是否为当前最佳模型。
    """
    # 创建一个字典来保存状态
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'miou': miou,
        'auc': auc,
        'f1': f1,
        'eer': eer
    }
    # 保存检查点文件
    torch.save(state, save_path)
    # 如果是最佳模型，额外保存一个名为 'best_model.pth' 的文件
    if is_best:
        best_model_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(state, best_model_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点。

    参数:
        model (torch.nn.Module): 需要加载状态的模型。
        optimizer (torch.optim.Optimizer): 需要加载状态的优化器。
        checkpoint_path (str): 检查点文件的路径。

    返回:
        tuple: 返回加载的轮次、miou、auc、f1和eer。
    """
    # 加载检查点文件
    checkpoint = torch.load(checkpoint_path)
    # 加载模型和优化器的状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 返回保存的指标
    return checkpoint['epoch'], checkpoint['miou'], checkpoint['auc'], checkpoint['f1'], checkpoint['eer']