# 导入必要的库
import torch
import torch.nn.functional as F

def orthogonal_loss(features1, features2):
    """
    计算两组特征之间的正交损失。
    这个损失函数的目标是使两组特征向量解耦（即正交）。
    通过最小化它们之间余弦相似度的平方来实现。当两个向量正交时，它们的余弦相似度为0。

    参数:
        features1 (torch.Tensor): 第一组特征，形状为 (N, D)，其中 N 是批量大小，D 是特征维度。
        features2 (torch.Tensor): 第二组特征，形状为 (N, D)。

    返回:
        torch.Tensor: 计算出的正交损失，是一个标量。
    """
    # 对特征进行L2归一化，使其成为单位向量，这样点积就直接等于余弦相似度
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    
    # 逐元素相乘后在特征维度上求和，得到批量中每个样本对的点积（余弦相似度）
    # (N, D) * (N, D) -> (N, D) -> sum -> (N,)
    cosine_similarity = torch.sum(features1 * features2, dim=1)
    
    # 计算余弦相似度的平方，并取批量均值作为最终的损失
    # 这个值越小，表示两组特征越趋于正交
    loss = torch.mean(cosine_similarity**2)
    
    return loss