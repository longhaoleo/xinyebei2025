# 导入必要的库
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_metrics(labels, predictions):
    """
    计算并返回多种评估指标，用于全面评估二分类模型的性能。

    参数:
        labels (np.ndarray): 真实标签 (0或1)。
        predictions (np.ndarray): 模型输出的预测概率，范围在 [0, 1] 之间。

    返回:
        dict: 包含准确率、AUC、精确率、召回率、F1分数和EER的字典。
    """
    # 将概率转换为二进制预测 (通常使用0.5作为阈值)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # 计算准确率：预测正确的样本占总样本的比例
    accuracy = accuracy_score(labels, binary_predictions)
    
    # 计算AUC (Area Under the ROC Curve)：ROC曲线下的面积，衡量模型整体排序能力
    auc = roc_auc_score(labels, predictions)
    
    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_predictions, average='binary', zero_division=0)
    
    # 计算相等错误率 (EER): 错误拒绝率(FRR)和错误接受率(FAR)相等时的值。
    # 这是生物识别和安全领域常用的指标。
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    # EER 对应于 1 - TPR = FPR 的点
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # 将所有指标存入字典以便于记录和展示
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'eer': eer
    }
    
    return metrics