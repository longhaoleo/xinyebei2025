# 导入必要的库
import logging
from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_file_path):
    """
    配置日志记录器。

    参数:
        log_file_path (str): 日志文件的保存路径。

    返回:
        logging.Logger: 配置好的日志记录器实例。
    """
    # 获取一个日志记录器实例
    logger = logging.getLogger()
    # 设置日志级别为INFO
    logger.setLevel(logging.INFO)
    
    # 创建一个文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file_path)
    # 创建一个格式化器，定义日志的输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 将文件处理器添加到日志记录器
    logger.addHandler(file_handler)
    
    return logger

class TensorBoardLogger:
    """
    用于集成TensorBoard的日志记录器类。
    """
    def __init__(self, log_dir):
        """
        初始化TensorBoardLogger。

        参数:
            log_dir (str): TensorBoard日志的保存目录。
        """
        # 创建一个SummaryWriter实例
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """
        记录标量值。

        参数:
            tag (str): 数据的标签。
            value (float): 要记录的标量值。
            step (int): 记录的步数（例如，训练轮次）。
        """
        self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        """
        记录图像。

        参数:
            tag (str): 图像的标签。
            image (torch.Tensor or numpy.ndarray): 要记录的图像。
            step (int): 记录的步数。
        """
        self.writer.add_image(tag, image, step)

    def log_pr_curve(self, tag, labels, predictions, step):
        """
        记录精确率-召回率（PR）曲线。

        参数:
            tag (str): PR曲线的标签。
            labels (torch.Tensor or numpy.ndarray): 真实标签。
            predictions (torch.Tensor or numpy.ndarray): 模型的预测概率。
            step (int): 记录的步数。
        """
        self.writer.add_pr_curve(tag, labels, predictions, step)

    def close(self):
        """
        关闭SummaryWriter。
        """
        self.writer.close()