# 导入必要的库
import torch
import torch.nn as nn
from torch.autograd import Function
from .backbone import get_backbone

class GradientReversalLayer(Function):
    """梯度反转层（GRL）的实现

    在前向传播中，它是一个恒等变换。
    在反向传播中，它将输入的梯度乘以一个负的常数 alpha，从而反转梯度的方向。
    这在对抗训练中非常有用，例如在领域自适应或移除敏感信息（如身份）时，
    可以最大化身份分类器的损失，同时最小化主任务的损失。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """前向传播
        Args:
            ctx: 上下文对象，用于存储反向传播所需的信息
            x: 输入张量
            alpha: 梯度缩放因子
        """
        # 保存 alpha 值以供反向传播使用
        ctx.alpha = alpha
        # 返回原始输入，不进行任何修改
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播
        Args:
            ctx: 上下文对象
            grad_output: 来自后续层的梯度
        Returns:
            tuple: (反转后的梯度, None)，None对应于alpha参数，它不需要梯度
        """
        # 将梯度取反，并乘以 alpha 进行缩放
        output = grad_output.neg() * ctx.alpha
        return output, None

class FreqBranch(nn.Module):
    """频域分支的轻量级CNN

    用于从频域表示（如FFT幅度谱）中提取特征。
    通常频域图的结构比空间域图像更简单，因此可以使用一个较浅的网络。
    """
    def __init__(self, in_channels=1, out_features=512):
        """初始化频域分支
        Args:
            in_channels (int): 输入通道数（频域图通常是单通道）
            out_features (int): 输出特征维度
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 自适应平均池化层，将特征图降维到1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，将特征向量映射到指定维度
        self.fc = nn.Linear(256, out_features)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) # 展平为 (batch_size, features)
        x = self.fc(x)
        return x

class ComprehensiveModel(nn.Module):
    """综合模型

    集成了空域-频域双流结构，并包含一个身份抑制分支。
    1.  **空间主干 (Spatial Backbone)**: 使用强大的预训练网络（如ResNet）提取图像的深层语义特征。
    2.  **频域分支 (Frequency Branch)**: 使用一个轻量级CNN处理频域输入，捕捉伪造痕迹。
    3.  **特征融合 (Feature Fusion)**: 将空间和频率特征拼接并融合，以供最终分类。
    4.  **分类头 (Classification Head)**: 对融合后的特征进行二分类（真/假）。
    5.  **身份抑制分支 (Identity Suppression Branch)**: 对空间特征进行身份分类，但通过GRL层反转梯度，
        迫使空间主干学习与身份无关的特征。
    """
    def __init__(self, backbone_name='resnet50', pretrained=True, num_classes=1, num_identities=1000):
        super().__init__()
        # 1. 空间域主干网络
        self.spatial_backbone = get_backbone(backbone_name, pretrained)
        # 获取主干网络的输出特征维度
        backbone_out_features = self.spatial_backbone.fc.in_features
        # 移除原始分类器，我们只使用特征提取部分
        self.spatial_backbone.fc = nn.Identity()

        # 2. 频域分支
        # 频域特征维度可以设置得小一些，作为辅助信息
        self.freq_branch = FreqBranch(out_features=backbone_out_features // 4)

        # 3. 特征融合层
        # 输入维度是空间特征和频率特征维度之和
        fusion_in_features = backbone_out_features + (backbone_out_features // 4)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5) # 使用Dropout防止过拟合
        )

        # 4. 分类判别头 (真/假)
        self.classification_head = nn.Linear(512, num_classes)

        # 5. 身份抑制分支
        self.identity_head = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_identities) # 输出每个身份的logits
        )

    def forward(self, x_spatial, x_freq, grl_alpha=1.0):
        """模型的前向传播
        Args:
            x_spatial (torch.Tensor): 空间域的输入图像张量
            x_freq (torch.Tensor): 频域的输入图像张量
            grl_alpha (float): GRL层的梯度缩放因子
        Returns:
            tuple: (伪造分类logits, 身份分类logits, 空间特征, 频率特征)
        """
        # 空间流：提取空间特征
        spatial_features = self.spatial_backbone(x_spatial)

        # 频率流：提取频率特征
        freq_features = self.freq_branch(x_freq)

        # 特征融合
        fused_features = torch.cat((spatial_features, freq_features), dim=1)
        fused_features = self.fusion_fc(fused_features)

        # 分类输出 (伪造/真实)
        cls_output = self.classification_head(fused_features)

        # 身份抑制分支的输出
        # 将空间特征通过GRL层，反转其梯度
        reversed_spatial_features = GradientReversalLayer.apply(spatial_features, grl_alpha)
        # 使用反转后的特征进行身份分类
        id_output = self.identity_head(reversed_spatial_features)

        # 返回所有需要的输出，用于计算不同的损失
        return cls_output, id_output, spatial_features, freq_features