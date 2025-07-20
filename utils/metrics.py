# 导入必要的库
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_metrics(labels, predictions):
    """
    计算并返回多种评估指标。

    参数:
        labels (np.ndarray): 真实标签 (0或1)。
        predictions (np.ndarray): 模型输出的预测概率。

    返回:
        dict: 包含准确率、AUC、精确率、召回率、F1分数和EER的字典。
    """
    # 将概率转换为二进制预测 (阈值为0.5)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # 计算准确率
    accuracy = accuracy_score(labels, binary_predictions)
    
    # 计算AUC
    auc = roc_auc_score(labels, predictions)
    
    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_predictions, average='binary')
    
    # 计算相等错误率 (EER)
    # EER是错误拒绝率(FRR)和错误接受率(FAR)相等时的值
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # 将所有指标存入字典
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'eer': eer
    }
    
    return metrics

# 注意: `roc_curve` 通常从 `sklearn.metrics` 导入，这里假设它已在别处定义或导入
# 为了代码完整性，这里补充一个导入
from sklearn.metrics import roc_curve