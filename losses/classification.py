# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (焦点损失) 的实现，用于处理类别不平衡问题。
    出自论文 "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        初始化 FocalLoss。
        参数:
            alpha (float): 平衡正负样本权重的因子。
            gamma (float): 调制因子，用于降低已分类良好的样本的损失贡献。
            reduction (str): 指定应用于输出的规约方法: 'none' | 'mean' | 'sum'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        前向传播。
        参数:
            inputs (torch.Tensor): 模型预测的 logits，形状为 (N, C)，其中 C 是类别数。
            targets (torch.Tensor): 真实标签，形状为 (N,)。
        返回:
            torch.Tensor: 计算出的焦点损失。
        """
        # 计算二元交叉熵损失，但不进行规约
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-bce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        # 根据指定的 reduction 参数进行处理
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_classification_loss(loss_name='bce'):
    """
    根据名称获取分类损失函数。
    参数:
        loss_name (str): 损失函数的名称 ('bce' 或 'focal')。
    返回:
        nn.Module: 对应的损失函数实例。
    """
    if loss_name == 'bce':
        # 返回标准的二元交叉熵损失
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'focal':
        # 返回Focal Loss
        return FocalLoss()
    else:
        # 如果名称无效，则抛出错误
        raise ValueError(f"Unsupported loss function: {loss_name}")