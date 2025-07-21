import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    """梯度反转层（GRL）的实现。"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播时，将梯度乘以一个负的常数
        output = grad_output.neg() * ctx.alpha
        return output, None

class IDUnawareModel(nn.Module):
    """一个旨在学习身份无关特征的模型。"""
    def __init__(self, backbone, num_classes=1, num_identities=1000):
        super().__init__()
        self.backbone = backbone
        # 获取主干网络的输出特征维度
        feature_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity() # 移除原始分类器

        # 用于主任务（如伪造检测）的分类器
        self.classifier = nn.Linear(feature_dim, num_classes)
        # 用于身份分类的分类器（对抗性分支）
        self.identity_classifier = nn.Linear(feature_dim, num_identities)

    def forward(self, x, alpha=1.0):
        """前向传播"""
        # 提取特征
        features = self.backbone(x)
        
        # 分类分支：直接使用特征进行主任务分类
        cls_output = self.classifier(features)

        # 身份分支：将特征通过GRL层，然后进行身份分类
        # GRL会反转来自此分支的梯度，从而惩罚那些对身份敏感的特征
        reversed_features = GradientReversalLayer.apply(features, alpha)
        id_output = self.identity_classifier(reversed_features)

        # 返回主任务输出、身份任务输出和原始特征
        return cls_output, id_output, features