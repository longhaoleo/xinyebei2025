
# 人脸伪造检测项目 (Face Forgery Detection)

这是一个基于 PyTorch 实现的深度学习项目，旨在检测单张
人脸图像中的伪造痕迹。项目采用了一种综合模型，结合了空间
域和频率域的特征，并通过身份解耦来提升模型的泛化能力和鲁
棒性。

## 核心思想与模型架构

本项目的核心在于 `ComprehensiveModel`，它集成了以下
几个关键部分：

1.  **空间域主干网络 (Spatial Backbone)**: 使用一
个强大的卷积神经网络（如 ResNet、EfficientNet 等）作
为主干，提取图像在空间域的深层语义特征。这些特征有助于捕
捉常规的视觉伪造线索。

2.  **频率域分支 (Frequency Branch)**: 伪造图像
（尤其是 Deepfakes）常常在频率域留下独特的痕迹。本项目
设计了一个轻量级的CNN分支 (`FreqBranch`)，专门处理图
像的快速傅里叶变换（FFT）结果，提取频域伪影特征。

3.  **特征融合**: 将空间域和频率域的特征进行拼接
（Concatenate），使模型能够同时利用两种模态的信息进行
决策。

4.  **身份抑制分支 (Identity-aware Branch)**: 为
了防止模型过度依赖人脸的身份信息（而不是伪造痕迹）来做判
断，我们引入了一个身份抑制机制。该分支通过一个梯度反转
层 (`GradientReversalLayer`) 连接，其任务是识别人
脸身份。在训练过程中，主干网络会学习生成与身份无关的伪造
特征，因为梯度反转会惩罚那些有助于身份识别的特征。这使得
模型更专注于伪造检测任务本身。

5.  **正交损失 (Orthogonal Loss)**: 为了进一步解耦
伪造特征和身份特征，我们引入了正交损失。该损失函数鼓励由
伪造检测分支和身份识别分支生成的两组特征向量在空间上相互
正交，从而确保它们学习到的是不相关的、独立的特征表示。

## 项目结构

```
├── configs/
│   └── default.yaml         # 默认配置文件，包含训练参数、模型设置等
├── data/
│   ├── dataset.py           # 定义人脸伪造数据集的Dataset类
│   └── transforms.py        # 定义空间域和频率域的图像预处理变换
├── losses/
│   ├── classification.py    # 定义分类损失函数（如BCE、Focal Loss）
│   └── orthogonal.py        # 定义正交损失函数
├── models/
│   └── comprehensive_model.py # 核心模型架构的实现
├── utils/
│   ├── checkpoint.py        # 保存和加载模型检查点的工具函数
│   ├── logger.py            # 配置日志和TensorBoard的工具
│   └── metrics.py           # 计算评估指标（AUC, F1, EER等）的函数
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── inference.py             # 单张图像推理脚本
└── requirements.txt         # 项目依赖

## 环境设置

1.  克隆本仓库。
2.  安装依赖项:
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

### 1. 准备数据

请根据 `data/dataset.py` 中的 `FaceForgeryDataset` 类的要求准备您的数据集，并创建相应的标注文件（通常是包含图像路径、伪造标签和身份标签的CSV文件）。

### 2. 配置

在 `configs/default.yaml` 文件中修改数据路径、模型参数、训练超参数等配置。

### 3. 训练

```bash
python train.py --config configs/default.yaml
```
训练日志和模型检查点将保存在配置文件中指定的目录。

### 4. 评估

使用训练好的模型在验证集或测试集上进行评估。

```bash
python evaluate.py --checkpoint_path /path/to/your/best_model.pth --config configs/default.yaml
```

### 5. 推理

对单张图像进行伪造检测。

```bash
python inference.py --checkpoint_path /path/to/your/best_model.pth --image_path /path/to/your/image.jpg
```

